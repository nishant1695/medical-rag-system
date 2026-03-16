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
        await db.commit()
        self._tables_ensured = True

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def build_from_document(
        self,
        chunks: list[dict],
        db: AsyncSession,
        provider,
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
                n = await self._store_triples(data, db)
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

    async def _store_triples(self, data: dict, db: AsyncSession) -> int:
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
                        "(workspace_id, source_id, target_id, relationship) "
                        "VALUES (:ws, :s, :t, :r)"
                    ),
                    {
                        "ws": self.workspace_id,
                        "s": src_id,
                        "t": tgt_id,
                        "r": relation,
                    },
                )
                new_edges += 1

        await db.commit()
        return new_edges

    # ── Query expansion ───────────────────────────────────────────────────────

    async def expand_query(
        self, query: str, db: AsyncSession
    ) -> list[str]:
        """
        Return related entity names for the concepts mentioned in the query.

        Strategy
        --------
        1. Load all node names for this workspace (capped at 500 by mention_count).
        2. Find which node names appear as substrings in the query.
        3. For each matched node, fetch its 1-hop neighbours (both directions).
        4. Return unique neighbour names not already in the query — up to 10.
        """
        try:
            await self._ensure_tables(db)
        except Exception:
            return []

        # Load candidate nodes
        rows = (
            await db.execute(
                text(
                    "SELECT id, normalized_name FROM graph_nodes "
                    "WHERE workspace_id=:ws "
                    "ORDER BY mention_count DESC LIMIT 500"
                ),
                {"ws": self.workspace_id},
            )
        ).fetchall()

        if not rows:
            return []

        query_lower = query.lower()
        matched_ids = [
            row[0]
            for row in rows
            if row[1] in query_lower
        ]

        if not matched_ids:
            return []

        # 1-hop traversal for each matched node
        related: list[str] = []
        seen_lower: set[str] = set()

        for node_id in matched_ids[:5]:  # limit traversal to top 5 matched
            # Outgoing edges
            out_rows = (
                await db.execute(
                    text(
                        "SELECT n.entity_name FROM graph_edges e "
                        "JOIN graph_nodes n ON e.target_id = n.id "
                        "WHERE e.workspace_id=:ws AND e.source_id=:nid "
                        "ORDER BY e.weight DESC LIMIT 8"
                    ),
                    {"ws": self.workspace_id, "nid": node_id},
                )
            ).fetchall()

            # Incoming edges
            in_rows = (
                await db.execute(
                    text(
                        "SELECT n.entity_name FROM graph_edges e "
                        "JOIN graph_nodes n ON e.source_id = n.id "
                        "WHERE e.workspace_id=:ws AND e.target_id=:nid "
                        "ORDER BY e.weight DESC LIMIT 8"
                    ),
                    {"ws": self.workspace_id, "nid": node_id},
                )
            ).fetchall()

            for (name,) in out_rows + in_rows:
                nl = name.lower()
                if nl not in query_lower and nl not in seen_lower:
                    seen_lower.add(nl)
                    related.append(name)

        return related[:10]

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
