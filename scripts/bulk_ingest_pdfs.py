"""
Bulk ingest PDF files into the knowledge base (DB + vector store).

Usage:
    python scripts/bulk_ingest_pdfs.py --workspace 1 --dir /path/to/pdfs
    python scripts/bulk_ingest_pdfs.py --workspace 1 --dir ./data/pdfs --limit 20
    python scripts/bulk_ingest_pdfs.py --workspace 1 --dir ./data/pdfs --pattern "*.pdf"

The script:
  1. Scans the target directory for PDF files
  2. Creates a Document record for each
  3. Runs the full parser pipeline (Docling + metadata extraction + DOI/CrossRef lookup)
  4. Embeds and indexes chunks in ChromaDB
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models import Document, KnowledgeBase, DocumentStatus
from app.services.rag_service import get_rag_service

logger = logging.getLogger("bulk_ingest_pdfs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def ingest_pdf(
    session: AsyncSession,
    workspace_id: int,
    pdf_path: Path,
    subspecialty: Optional[str] = None,
) -> bool:
    """Ingest a single PDF file. Returns True on success."""
    filename = pdf_path.name

    # Skip if already indexed by filename
    existing = await session.execute(
        select(Document).where(
            Document.workspace_id == workspace_id,
            Document.original_filename == filename,
        )
    )
    if existing.scalar_one_or_none():
        logger.info(f"'{filename}' already indexed — skipping")
        return True

    # Create Document record
    doc = Document(
        workspace_id=workspace_id,
        original_filename=filename,
        file_type="pdf",
        file_size=pdf_path.stat().st_size,
        file_path=str(pdf_path.resolve()),
        status=DocumentStatus.PENDING.value,
        subspecialty=subspecialty or "",
    )
    session.add(doc)
    await session.flush()  # populate doc.id

    # Run full parse + embed + index pipeline (no PMID — parser will extract metadata)
    rag = get_rag_service(session, workspace_id)
    try:
        n_chunks = await rag.process_document(
            document_id=doc.id,
            file_path=str(pdf_path.resolve()),
            subspecialty=subspecialty,
        )
        paper_url = doc.paper_url  # set by rag_service after parsing
        logger.info(
            f"Indexed '{filename}' (doc_id={doc.id}, {n_chunks} chunks"
            + (f", subspecialty={subspecialty}" if subspecialty else "")
            + (f", url={paper_url}" if paper_url else "")
            + ")"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to index '{filename}': {e}")
        return False


async def bulk_ingest_pdfs(
    workspace_id: int,
    pdf_dir: Path,
    pattern: str,
    limit: int,
    subspecialty: Optional[str] = None,
):
    async with AsyncSessionLocal() as session:
        ws = await session.get(KnowledgeBase, workspace_id)
        if not ws:
            logger.error(f"Workspace {workspace_id} not found. Create it via the UI first.")
            return

        pdf_files = sorted(pdf_dir.glob(pattern))[:limit]
        if not pdf_files:
            logger.warning(f"No files matching '{pattern}' found in {pdf_dir}")
            return

        spec_label = f" [{subspecialty}]" if subspecialty else ""
        logger.info(
            f"Found {len(pdf_files)} PDFs to ingest into workspace "
            f"'{ws.name}' (id={workspace_id}){spec_label}"
        )

        success = 0
        for i, pdf in enumerate(pdf_files):
            logger.info(f"[{i+1}/{len(pdf_files)}] {pdf.name}")
            ok = await ingest_pdf(session, workspace_id, pdf, subspecialty=subspecialty)
            if ok:
                success += 1
            try:
                await session.commit()
            except Exception as e:
                logger.warning(f"Commit failed after '{pdf.name}': {e}")
                await session.rollback()

        logger.info(f"Done: {success}/{len(pdf_files)} ingested successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk ingest PDF files into the medical RAG system")
    parser.add_argument("--workspace", type=int, required=True, help="Workspace (knowledge base) ID")
    parser.add_argument("--dir", type=Path, required=True, help="Directory containing PDF files")
    parser.add_argument("--pattern", default="*.pdf", help="Glob pattern for files (default: *.pdf)")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of files to ingest")
    parser.add_argument(
        "--subspecialty",
        choices=["breast", "hand", "craniofacial", "microsurgery", "burns", "aesthetic", "lower_extremity"],
        default=None,
        help="Tag all ingested PDFs with this subspecialty (used for filtered retrieval)",
    )
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"Error: '{args.dir}' is not a directory")
        sys.exit(1)

    asyncio.run(bulk_ingest_pdfs(args.workspace, args.dir, args.pattern, args.limit, args.subspecialty))
