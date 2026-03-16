"""
Bulk ingest papers from data/papers into the knowledge base (DB + vector store).

Assumes fetch_pubmed.py has saved data/papers/{pmid}.json and {pmid}.txt

Usage:
    python scripts/bulk_ingest.py --workspace 1 --limit 50
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

# When running as script, add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models import Document, KnowledgeBase, DocumentStatus
from app.services.rag_service import get_rag_service

logger = logging.getLogger("bulk_ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PAPERS_DIR = Path("data/papers")


async def ingest_one(session: AsyncSession, workspace_id: int, pmid: str) -> bool:
    json_path = PAPERS_DIR / f"{pmid}.json"
    txt_path = PAPERS_DIR / f"{pmid}.txt"

    if not json_path.exists() and not txt_path.exists():
        logger.warning(f"No files for PMID {pmid}")
        return False

    # Load metadata from json if available, else minimal info
    meta = {}
    ingest_path = json_path if json_path.exists() else txt_path
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {json_path}: {e}")

    title = meta.get("title") or f"PMID {pmid}"
    abstract = meta.get("abstract", "")

    # Check if PMID already indexed (avoid duplicates)
    from sqlalchemy import select
    existing = await session.execute(
        select(Document).where(Document.pmid == pmid)
    )
    if existing.scalar_one_or_none():
        logger.info(f"PMID {pmid} already indexed — skipping")
        return True

    # Create Document record
    doc = Document(
        workspace_id=workspace_id,
        original_filename=ingest_path.name,
        file_type="pubmed-json" if json_path.exists() else "pubmed-txt",
        file_size=ingest_path.stat().st_size,
        file_path=str(ingest_path),
        status=DocumentStatus.PENDING.value,
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=meta.get("authors", []),
        journal=meta.get("journal"),
        publication_year=meta.get("year"),
        mesh_terms=meta.get("mesh_terms", []),
    )
    session.add(doc)
    await session.flush()  # get document.id without committing

    # Process document (parse + embed + index)
    rag = get_rag_service(session, workspace_id)
    try:
        n_chunks = await rag.process_document(
            document_id=doc.id,
            file_path=str(ingest_path),
            pmid=pmid,
        )
        logger.info(f"Indexed PMID {pmid} (doc_id={doc.id}, {n_chunks} chunks)")
        return True
    except Exception as e:
        logger.error(f"Failed to index PMID {pmid}: {e}")
        return False


async def bulk_ingest(workspace_id: int, limit: int = 50):
    async with AsyncSessionLocal() as session:
        # Ensure workspace exists
        ws = await session.get(KnowledgeBase, workspace_id)
        if not ws:
            logger.error(f"Workspace {workspace_id} not found")
            return

        # Gather PMIDs from JSON files (prefer JSON over txt)
        json_pmids = {p.stem for p in PAPERS_DIR.glob("*.json")}
        txt_pmids = {p.stem for p in PAPERS_DIR.glob("*.txt")}
        all_pmids = sorted(json_pmids | txt_pmids)[:limit]

        logger.info(f"Found {len(all_pmids)} papers to ingest (limit={limit})")

        success = 0
        for i, pmid in enumerate(all_pmids):
            logger.info(f"[{i+1}/{len(all_pmids)}] Processing PMID {pmid}")
            ok = await ingest_one(session, workspace_id, pmid)
            if ok:
                success += 1
            # Commit after each successful doc to avoid holding long transactions
            try:
                await session.commit()
            except Exception as e:
                logger.warning(f"Commit failed after PMID {pmid}: {e}")
                await session.rollback()

        logger.info(f"Bulk ingest complete: {success}/{len(all_pmids)} succeeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=int, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    import asyncio
    asyncio.run(bulk_ingest(args.workspace, args.limit))
