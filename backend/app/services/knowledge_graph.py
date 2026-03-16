"""
Medical Knowledge Graph

Builds and queries a graph of medical entities and their relationships.

Nodes  — medical entities (procedure, condition, outcome, anatomy, technique,
          population, drug)
Edges  — typed relationships (treats, complicates, compared_to, requires,
          associated_with, contraindicates, part_of, predicts)

Pipeline
--------
Ingestion : extract (entity, relation, entity) triples from document chunks
            via a single batched LLM call per 4-chunk window. Triples are
            upserted into two SQLite tables (graph_nodes, graph_edges).
            Repeated mentions increment mention_count / weight.

Retrieval : match query text against node names → 1-hop graph traversal →
            return related entity names → caller appends them to the retrieval
            query string to enrich the embedding.

Tables are created with CREATE TABLE IF NOT EXISTS on first use, so they are
added to existing databases automatically without a migration.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm.base import LLMMessage

logger = logging.getLogger(__name__)

# How many chunks to process in one LLM call
_BATCH_SIZE = 4

_EXTRACT_SYSTEM = (
    "You extract medical knowledge graph triples. "
    "Output only valid JSON, nothing else."
)

_EXTRACT_TEMPLATE = """\
Extract medical entities and relationships from the text below.

Entity types  : procedure, condition, outcome, anatomy, technique, population, drug
Relation types: treats, complicates, compared_to, requires, associated_with,
                contraindicates, part_of, predicts

Output ONLY this JSON (no markdown, no extra text):
{{
  "entities": [
    {{"name": "DIEP flap", "type": "procedure"}},
    {{"name": "flap failure", "type": "outcome"}}
  ],
  "relationships": [
    {{"source": "DIEP flap", "relation": "complicates", "target": "flap failure"}}
  ]
}}

Rules:
- Maximum 12 entities and 15 relationships per batch
- Standard medical terminology, title case for entity names
- Only clinically meaningful relationships
- Only relationships between entities you listed above

Text:
{text}
"""

_CREATE_NODES = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id     INTEGER NOT NULL,
    entity_name      TEXT    NOT NULL,
    entity_type      TEXT    NOT NULL,
    normalized_name  TEXT    NOT NULL,
    mention_count    INTEGER DEFAULT 1,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workspace_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    UNIQUE (workspace_id, normalized_name)
)
"""

_CREATE_EDGES = """
CREATE TABLE IF NOT EXISTS graph_edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL,
    source_id    INTEGER NOT NULL,
    target_id    INTEGER NOT NULL,
    relationship TEXT    NOT NULL,
    weight       REAL    DEFAULT 1.0,
    document_id  INTEGER,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workspace_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    FOREIGN KEY (source_id)    REFERENCES graph_nodes(id)     ON DELETE CASCADE,
    FOREIGN KEY (target_id)    REFERENCES graph_nodes(id)     ON DELETE CASCADE,
    UNIQUE (source_id, target_id, relationship)
)
"""


class MedicalKnowledgeGraph:
    """Knowledge graph scoped to one workspace."""

    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self._tables_ensured = False

    # ── Schema ────────────────────────────────────────────────────────────────

    async def _ensure_tables(self, db: AsyncSession) -> None:
        if self._tables_ensured:
            return
        await db.execute(text(_CREATE_NODES))
        await db.execute(text(_CREATE_EDGES))
        # Migrate existing graph_edges tables that pre-date the document_id column
        try:
            await db.execute(text("ALTER TABLE graph_edges ADD COLUMN document_id INTEGER"))
            await db.commit()
        except Exception:
            pass  # Column already exists — safe to ignore
        await db.commit()
        self._tables_ensured = True

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def build_from_document(
        self,
        chunks: list[dict],
        db: AsyncSession,
        provider,
        document_id: Optional[int] = None,
    ) -> int:
        """
        Extract entities/relationships from document chunks and store them.

        chunks : list of chunk dicts with a "content" key (same format as
                 rag_service produces)
        provider : any BaseLLMProvider instance

        Returns the number of new graph edges created.
        """
        await self._ensure_tables(db)

        total_edges = 0
        for i in range(0, len(chunks), _BATCH_SIZE):
            batch_text = "\n\n---\n\n".join(
                c["content"] for c in chunks[i : i + _BATCH_SIZE]
            )
            data = await self._extract_triples(batch_text, provider)
            if data:
                n = await self._store_triples(data, db, document_id=document_id)
                total_edges += n

        logger.info(
            f"KG workspace={self.workspace_id}: "
            f"processed {len(chunks)} chunks → {total_edges} new edges"
        )
        return total_edges

    async def _extract_triples(
        self, text_: str, provider
    ) -> Optional[dict]:
        """Call the LLM to extract entity/relationship triples."""
        prompt = _EXTRACT_TEMPLATE.format(text=text_[:3000])
        try:
            raw = ""
            async for chunk in provider.astream(
                [LLMMessage(role="user", content=prompt)],
                system_prompt=_EXTRACT_SYSTEM,
                temperature=0.0,
                max_tokens=700,
            ):
                if chunk.type == "text":
                    raw += chunk.text

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            if "entities" in data and "relationships" in data:
                return data
        except Exception as exc:
            logger.debug(f"KG extraction error: {exc}")
        return None

    async def _store_triples(
        self, data: dict, db: AsyncSession, document_id: Optional[int] = None
    ) -> int:
        """Upsert nodes and edges. Returns number of new edges inserted."""
        entity_ids: dict[str, int] = {}  # normalized_name → row id

        # ── Upsert nodes ──────────────────────────────────────────────────────
        for entity in data.get("entities", []):
            name = (entity.get("name") or "").strip()
            etype = (entity.get("type") or "unknown").strip()
            if not name:
                continue
            norm = name.lower()

            row = (
                await db.execute(
                    text(
                        "SELECT id FROM graph_nodes "
                        "WHERE workspace_id=:ws AND normalized_name=:n"
                    ),
                    {"ws": self.workspace_id, "n": norm},
                )
            ).fetchone()

            if row:
                entity_ids[norm] = row[0]
                await db.execute(
                    text(
                        "UPDATE graph_nodes SET mention_count = mention_count + 1 "
                        "WHERE id=:id"
                    ),
                    {"id": row[0]},
                )
            else:
                result = await db.execute(
                    text(
                        "INSERT INTO graph_nodes "
                        "(workspace_id, entity_name, entity_type, normalized_name) "
                        "VALUES (:ws, :name, :type, :norm)"
                    ),
                    {
                        "ws": self.workspace_id,
                        "name": name,
                        "type": etype,
                        "norm": norm,
                    },
                )
                entity_ids[norm] = result.lastrowid

        # ── Upsert edges ──────────────────────────────────────────────────────
        new_edges = 0
        for rel in data.get("relationships", []):
            src_norm = (rel.get("source") or "").strip().lower()
            tgt_norm = (rel.get("target") or "").strip().lower()
            relation = (rel.get("relation") or "").strip()

            if not src_norm or not tgt_norm or not relation:
                continue
            if src_norm not in entity_ids or tgt_norm not in entity_ids:
                continue

            src_id = entity_ids[src_norm]
            tgt_id = entity_ids[tgt_norm]

            existing = (
                await db.execute(
                    text(
                        "SELECT id FROM graph_edges "
                        "WHERE source_id=:s AND target_id=:t AND relationship=:r"
                    ),
                    {"s": src_id, "t": tgt_id, "r": relation},
                )
            ).fetchone()

            if existing:
                await db.execute(
                    text(
                        "UPDATE graph_edges SET weight = weight + 0.1 "
                        "WHERE source_id=:s AND target_id=:t AND relationship=:r"
                    ),
                    {"s": src_id, "t": tgt_id, "r": relation},
                )
            else:
                await db.execute(
                    text(
                        "INSERT INTO graph_edges "
                        "(workspace_id, source_id, target_id, relationship, document_id) "
                        "VALUES (:ws, :s, :t, :r, :doc)"
                    ),
                    {
                        "ws": self.workspace_id,
                        "s": src_id,
                        "t": tgt_id,
                        "r": relation,
                        "doc": document_id,
                    },
                )
                new_edges += 1

        await db.commit()
        return new_edges

    # ── Document deletion ─────────────────────────────────────────────────────

    async def delete_document_edges(
        self, document_id: int, db: AsyncSession
    ) -> None:
        """
        Remove all graph edges contributed by a specific document, then prune
        any nodes that have become fully isolated (no remaining edges).

        Called by rag_service.delete_document() to keep graph state consistent
        with the vector store — deleted papers no longer influence KG expansion.
        """
        await self._ensure_tables(db)

        # Delete edges contributed by this document
        await db.execute(
            text(
                "DELETE FROM graph_edges "
                "WHERE workspace_id=:ws AND document_id=:doc"
            ),
            {"ws": self.workspace_id, "doc": document_id},
        )

        # Prune nodes with no remaining edges in either direction
        await db.execute(
            text(
                "DELETE FROM graph_nodes "
                "WHERE workspace_id=:ws "
                "AND id NOT IN ("
                "  SELECT source_id FROM graph_edges WHERE workspace_id=:ws "
                "  UNION "
                "  SELECT target_id FROM graph_edges WHERE workspace_id=:ws"
                ")"
            ),
            {"ws": self.workspace_id},
        )

        await db.commit()
        logger.info(
            f"KG workspace={self.workspace_id}: "
            f"removed edges for document {document_id}, pruned orphan nodes"
        )

    # ── Query expansion ───────────────────────────────────────────────────────

    async def _load_adjacency(
        self, db: AsyncSession
    ) -> tuple[dict[int, str], dict[int, list[tuple[int, float]]]]:
        """
        Load the entire workspace graph into memory as an adjacency list.

        Returns
        -------
        id_to_name : {node_id: entity_name}
        adjacency  : {node_id: [(neighbour_id, edge_weight), ...]}
                     Both directions are stored so traversal is undirected.
        """
        node_rows = (
            await db.execute(
                text(
                    "SELECT id, entity_name FROM graph_nodes "
                    "WHERE workspace_id=:ws"
                ),
                {"ws": self.workspace_id},
            )
        ).fetchall()

        edge_rows = (
            await db.execute(
                text(
                    "SELECT source_id, target_id, weight FROM graph_edges "
                    "WHERE workspace_id=:ws"
                ),
                {"ws": self.workspace_id},
            )
        ).fetchall()

        id_to_name: dict[int, str] = {r[0]: r[1] for r in node_rows}

        adjacency: dict[int, list[tuple[int, float]]] = {
            node_id: [] for node_id in id_to_name
        }
        for src, tgt, weight in edge_rows:
            if src in adjacency:
                adjacency[src].append((tgt, weight))
            if tgt in adjacency:
                adjacency[tgt].append((src, weight))

        return id_to_name, adjacency

    async def expand_query(
        self,
        query: str,
        db: AsyncSession,
        hops: int = 2,
        max_results: int = 15,
    ) -> list[str]:
        """
        Return related entity names for concepts mentioned in the query.

        Uses BFS up to `hops` depth from each matched seed node.
        Nodes are scored by cumulative edge weight decayed by hop distance
        (weight × 0.6^(hop-1)) so directly connected, strongly evidenced
        concepts rank highest. Nodes already mentioned in the query are
        excluded from the results.

        Parameters
        ----------
        hops        : traversal depth (default 2)
        max_results : maximum related terms to return (default 15)
        """
        try:
            await self._ensure_tables(db)
        except Exception:
            return []

        # Load all nodes and check graph is populated
        node_rows = (
            await db.execute(
                text(
                    "SELECT id, normalized_name FROM graph_nodes "
                    "WHERE workspace_id=:ws "
                    "ORDER BY mention_count DESC LIMIT 1000"
                ),
                {"ws": self.workspace_id},
            )
        ).fetchall()

        if not node_rows:
            return []

        # Find seed nodes — graph entities that appear in the query text
        query_lower = query.lower()
        seed_ids = [
            row[0]
            for row in node_rows
            if row[1] in query_lower
        ]

        if not seed_ids:
            return []

        # Load full adjacency list into memory (fast for typical graph sizes)
        id_to_name, adjacency = await self._load_adjacency(db)

        # BFS with distance-decayed scoring
        # scores[node_id] = best cumulative score seen so far
        scores: dict[int, float] = {}

        # frontier: list of (node_id, current_hop, accumulated_score)
        frontier = [(nid, 0, 1.0) for nid in seed_ids]
        visited: set[int] = set(seed_ids)

        while frontier:
            next_frontier = []
            for node_id, hop, score in frontier:
                if hop >= hops:
                    continue
                for neighbour_id, edge_weight in adjacency.get(node_id, []):
                    neighbour_score = score * edge_weight * (0.6 ** hop)
                    if neighbour_id not in visited:
                        visited.add(neighbour_id)
                        next_frontier.append(
                            (neighbour_id, hop + 1, neighbour_score)
                        )
                        scores[neighbour_id] = neighbour_score
                    elif neighbour_score > scores.get(neighbour_id, 0):
                        # Found a higher-scoring path — update but don't re-expand
                        scores[neighbour_id] = neighbour_score

            frontier = next_frontier

        # Exclude seed nodes and terms already in the query
        results: list[tuple[str, float]] = []
        for node_id, score in scores.items():
            name = id_to_name.get(node_id, "")
            if not name:
                continue
            if name.lower() in query_lower:
                continue
            results.append((name, score))

        # Sort by score descending, return top max_results names
        results.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in results[:max_results]]

    # ── Graph export ──────────────────────────────────────────────────────────

    async def get_graph_data(self, db: AsyncSession) -> dict:
        """Return the full graph as {nodes, edges} for API/visualisation."""
        await self._ensure_tables(db)

        node_rows = (
            await db.execute(
                text(
                    "SELECT id, entity_name, entity_type, mention_count "
                    "FROM graph_nodes WHERE workspace_id=:ws "
                    "ORDER BY mention_count DESC"
                ),
                {"ws": self.workspace_id},
            )
        ).fetchall()

        edge_rows = (
            await db.execute(
                text(
                    "SELECT source_id, target_id, relationship, weight "
                    "FROM graph_edges WHERE workspace_id=:ws "
                    "ORDER BY weight DESC"
                ),
                {"ws": self.workspace_id},
            )
        ).fetchall()

        return {
            "nodes": [
                {
                    "id": r[0],
                    "name": r[1],
                    "type": r[2],
                    "mention_count": r[3],
                }
                for r in node_rows
            ],
            "edges": [
                {
                    "source": r[0],
                    "target": r[1],
                    "relationship": r[2],
                    "weight": round(r[3], 2),
                }
                for r in edge_rows
            ],
        }


# ── Module-level factory ──────────────────────────────────────────────────────

_kg_cache: dict[int, MedicalKnowledgeGraph] = {}


def get_knowledge_graph(workspace_id: int) -> MedicalKnowledgeGraph:
    """Return a cached MedicalKnowledgeGraph instance for a workspace."""
    if workspace_id not in _kg_cache:
        _kg_cache[workspace_id] = MedicalKnowledgeGraph(workspace_id)
    return _kg_cache[workspace_id]
