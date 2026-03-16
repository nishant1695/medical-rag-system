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
            upserted into SQL tables and tracked per document so deletions can
            remove only the evidence contributed by the deleted paper.

Retrieval : match query text against node names → graph traversal → return
            related entity names → caller appends them to the retrieval query
            string to enrich the embedding.

Tables are created on first use without a formal migration layer so the graph
feature can be added to existing databases incrementally.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    UniqueConstraint,
    text,
)
from sqlalchemy.schema import CreateTable
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import Base
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

_GRAPH_METADATA = Base.metadata

_GRAPH_NODES = Table(
    "graph_nodes",
    _GRAPH_METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "workspace_id",
        Integer,
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("entity_name", String, nullable=False),
    Column("entity_type", String, nullable=False),
    Column("normalized_name", String, nullable=False),
    Column("mention_count", Integer, nullable=False, server_default=text("1")),
    Column("created_at", DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("workspace_id", "normalized_name", name="uq_graph_nodes_workspace_name"),
)

_GRAPH_EDGES = Table(
    "graph_edges",
    _GRAPH_METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "workspace_id",
        Integer,
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("source_id", Integer, ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False),
    Column("target_id", Integer, ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False),
    Column("relationship", String, nullable=False),
    Column("weight", Float, nullable=False, server_default=text("1.0")),
    # Legacy column retained for backward-compatible backfill from older DBs.
    Column("document_id", Integer, nullable=True),
    Column("created_at", DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("source_id", "target_id", "relationship", name="uq_graph_edges_relation"),
)

_GRAPH_EDGE_MENTIONS = Table(
    "graph_edge_mentions",
    _GRAPH_METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "workspace_id",
        Integer,
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("edge_id", Integer, ForeignKey("graph_edges.id", ondelete="CASCADE"), nullable=False),
    Column("document_id", Integer, nullable=False),
    Column("mention_count", Integer, nullable=False, server_default=text("1")),
    Column("created_at", DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("edge_id", "document_id", name="uq_graph_edge_mentions_edge_doc"),
)


def _weight_for_mentions(total_mentions: int) -> float:
    """Map relation mentions to the aggregate edge weight used for traversal."""
    if total_mentions <= 1:
        return 1.0
    return 1.0 + (total_mentions - 1) * 0.1


class MedicalKnowledgeGraph:
    """Knowledge graph scoped to one workspace."""

    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self._tables_ensured = False

    # ── Schema ────────────────────────────────────────────────────────────────

    async def _ensure_tables(self, db: AsyncSession) -> None:
        if self._tables_ensured:
            return
        await db.execute(CreateTable(_GRAPH_NODES, if_not_exists=True))
        await db.execute(CreateTable(_GRAPH_EDGES, if_not_exists=True))
        await db.execute(CreateTable(_GRAPH_EDGE_MENTIONS, if_not_exists=True))
        await db.commit()

        # Older graph_edges tables may predate the legacy document_id column.
        try:
            await db.execute(text("ALTER TABLE graph_edges ADD COLUMN document_id INTEGER"))
            await db.commit()
        except Exception:
            await db.rollback()

        # Backfill provenance for graph data that predates graph_edge_mentions.
        # Existing aggregate edge weight is converted into an approximate
        # mention_count for the stored document_id so delete semantics remain
        # consistent for already-ingested papers.
        await db.execute(
            text(
                "INSERT INTO graph_edge_mentions "
                "(workspace_id, edge_id, document_id, mention_count) "
                "SELECT e.workspace_id, e.id, e.document_id, "
                "       CASE "
                "         WHEN e.weight IS NULL OR e.weight <= 1 THEN 1 "
                "         ELSE CAST(ROUND((e.weight - 1.0) / 0.1) AS INTEGER) + 1 "
                "       END "
                "FROM graph_edges e "
                "WHERE e.document_id IS NOT NULL "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM graph_edge_mentions gem "
                "  WHERE gem.edge_id = e.id AND gem.document_id = e.document_id"
                ")"
            )
        )
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

            await db.execute(
                text(
                    "INSERT INTO graph_nodes "
                    "(workspace_id, entity_name, entity_type, normalized_name, mention_count) "
                    "VALUES (:ws, :name, :type, :norm, 0) "
                    "ON CONFLICT (workspace_id, normalized_name) DO NOTHING"
                ),
                {
                    "ws": self.workspace_id,
                    "name": name,
                    "type": etype,
                    "norm": norm,
                },
            )
            row = (
                await db.execute(
                    text(
                        "SELECT id FROM graph_nodes "
                        "WHERE workspace_id=:ws AND normalized_name=:n"
                    ),
                    {"ws": self.workspace_id, "n": norm},
                )
            ).fetchone()
            if not row:
                continue

            entity_ids[norm] = row[0]
            await db.execute(
                text(
                    "UPDATE graph_nodes SET mention_count = mention_count + 1 "
                    "WHERE id=:id"
                ),
                {"id": row[0]},
            )

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
                edge_id = existing[0]
            else:
                await db.execute(
                    text(
                        "INSERT INTO graph_edges "
                        "(workspace_id, source_id, target_id, relationship, weight, document_id) "
                        "VALUES (:ws, :s, :t, :r, 1.0, :doc) "
                        "ON CONFLICT (source_id, target_id, relationship) DO NOTHING"
                    ),
                    {
                        "ws": self.workspace_id,
                        "s": src_id,
                        "t": tgt_id,
                        "r": relation,
                        "doc": document_id,
                    },
                )
                edge_row = (
                    await db.execute(
                        text(
                            "SELECT id FROM graph_edges "
                            "WHERE source_id=:s AND target_id=:t AND relationship=:r"
                        ),
                        {"s": src_id, "t": tgt_id, "r": relation},
                    )
                ).fetchone()
                if not edge_row:
                    continue
                edge_id = edge_row[0]
                new_edges += 1

            if document_id is not None:
                await db.execute(
                    text(
                        "INSERT INTO graph_edge_mentions "
                        "(workspace_id, edge_id, document_id, mention_count) "
                        "VALUES (:ws, :edge_id, :doc, 0) "
                        "ON CONFLICT (edge_id, document_id) DO NOTHING"
                    ),
                    {
                        "ws": self.workspace_id,
                        "edge_id": edge_id,
                        "doc": document_id,
                    },
                )
                await db.execute(
                    text(
                        "UPDATE graph_edge_mentions "
                        "SET mention_count = mention_count + 1 "
                        "WHERE edge_id=:edge_id AND document_id=:doc"
                    ),
                    {
                        "edge_id": edge_id,
                        "doc": document_id,
                    },
                )
                total_mentions = (
                    await db.execute(
                        text(
                            "SELECT COALESCE(SUM(mention_count), 0) "
                            "FROM graph_edge_mentions WHERE edge_id=:edge_id"
                        ),
                        {"edge_id": edge_id},
                    )
                ).scalar_one()
                await db.execute(
                    text(
                        "UPDATE graph_edges "
                        "SET weight=:weight, document_id=COALESCE(document_id, :doc) "
                        "WHERE id=:edge_id"
                    ),
                    {
                        "weight": _weight_for_mentions(int(total_mentions or 0)),
                        "doc": document_id,
                        "edge_id": edge_id,
                    },
                )
            elif existing:
                await db.execute(
                    text(
                        "UPDATE graph_edges SET weight = weight + 0.1 "
                        "WHERE id=:edge_id"
                    ),
                    {"edge_id": edge_id},
                )

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

        edge_ids = [
            row[0]
            for row in (
                await db.execute(
                    text(
                        "SELECT DISTINCT edge_id FROM graph_edge_mentions "
                        "WHERE workspace_id=:ws AND document_id=:doc"
                    ),
                    {"ws": self.workspace_id, "doc": document_id},
                )
            ).fetchall()
        ]

        # Delete edge-mention provenance contributed by this document.
        await db.execute(
            text(
                "DELETE FROM graph_edge_mentions "
                "WHERE workspace_id=:ws AND document_id=:doc"
            ),
            {"ws": self.workspace_id, "doc": document_id},
        )

        for edge_id in edge_ids:
            total_mentions = (
                await db.execute(
                    text(
                        "SELECT COALESCE(SUM(mention_count), 0) "
                        "FROM graph_edge_mentions WHERE edge_id=:edge_id"
                    ),
                    {"edge_id": edge_id},
                )
            ).scalar_one()
            if total_mentions:
                await db.execute(
                    text(
                        "UPDATE graph_edges "
                        "SET weight=:weight, "
                        "    document_id=("
                        "      SELECT MIN(document_id) FROM graph_edge_mentions "
                        "      WHERE edge_id=:edge_id"
                        "    ) "
                        "WHERE id=:edge_id"
                    ),
                    {
                        "weight": _weight_for_mentions(int(total_mentions)),
                        "edge_id": edge_id,
                    },
                )
            else:
                await db.execute(
                    text("DELETE FROM graph_edges WHERE id=:edge_id"),
                    {"edge_id": edge_id},
                )

        # Legacy cleanup for old rows that only tracked a single document_id and
        # do not yet have provenance rows.
        await db.execute(
            text(
                "DELETE FROM graph_edges "
                "WHERE workspace_id=:ws AND document_id=:doc "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM graph_edge_mentions gem "
                "  WHERE gem.edge_id = graph_edges.id"
                ")"
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
