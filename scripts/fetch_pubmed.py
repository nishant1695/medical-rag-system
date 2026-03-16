"""
Fetch PubMed metadata and save as JSON + plain text for ingestion.

Usage:
    python scripts/fetch_pubmed.py --query "DIEP flap" --max-papers 50

Saves files under data/papers/{pmid}.json and {pmid}.txt
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

from Bio import Entrez, Medline

# Try to load environment from repo .env via pydantic-settings in config
try:
    # Importing config will read .env
    from app.core.config import settings
    if settings.PUBMED_EMAIL:
        Entrez.email = settings.PUBMED_EMAIL
    if settings.PUBMED_API_KEY:
        Entrez.api_key = settings.PUBMED_API_KEY
except Exception:
    # Fallback: rely on environment variables
    Entrez.email = os.environ.get("PUBMED_EMAIL", "your_email@example.com")
    Entrez.api_key = os.environ.get("PUBMED_API_KEY")

logger = logging.getLogger("fetch_pubmed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PAPERS_DIR = Path("data/papers")
PAPERS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_pmids(query: str, max_papers: int = 50) -> List[str]:
    logger.info(f"Searching PubMed for: {query} (max {max_papers})")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_papers, sort="relevance")
    record = Entrez.read(handle)
    pmids = record.get("IdList", [])
    logger.info(f"Found {len(pmids)} PMIDs")
    return pmids


def fetch_metadata(pmid: str) -> dict:
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = Medline.read(handle)

        # Extract DOI if present
        doi = ""
        for aid in record.get("AID", []) or []:
            if "doi" in aid.lower():
                doi = aid.split("[")[0].strip()
                break

        # Extract year
        year = None
        dp = record.get("DP", "")
        if dp:
            try:
                import re

                m = re.search(r"(\d{4})", dp)
                if m:
                    year = int(m.group(1))
            except Exception:
                year = None

        meta = {
            "pmid": pmid,
            "title": record.get("TI", ""),
            "abstract": record.get("AB", ""),
            "authors": record.get("AU", []),
            "journal": record.get("JT", ""),
            "year": year,
            "doi": doi,
            "mesh_terms": record.get("MH", []),
            "raw_medline": dict(record),
        }
        return meta
    except Exception as e:
        logger.warning(f"Failed to fetch metadata for {pmid}: {e}")
        return {}


def save_paper(meta: dict) -> Path:
    pmid = meta.get("pmid") or meta.get("PMID")
    if not pmid:
        raise ValueError("No PMID in metadata")

    json_path = PAPERS_DIR / f"{pmid}.json"
    txt_path = PAPERS_DIR / f"{pmid}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Create text file (title + abstract)
    title = meta.get("title", "")
    abstract = meta.get("abstract", "")
    journal = meta.get("journal", "")
    year = meta.get("year")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n")
        f.write(f"Journal: {journal} ({year})\n\n")
        f.write("Abstract:\n")
        f.write(abstract or "")

    return txt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="PubMed query (e.g., 'DIEP flap breast reconstruction')")
    parser.add_argument("--max-papers", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    pmids = fetch_pmids(args.query, args.max_papers + args.start)
    pmids = pmids[args.start : args.start + args.max_papers]

    saved = []
    for pmid in pmids:
        meta = fetch_metadata(pmid)
        if not meta:
            continue
        path = save_paper(meta)
        logger.info(f"Saved {pmid} -> {path}")
        saved.append(pmid)

    logger.info(f"Completed: saved {len(saved)} papers to {PAPERS_DIR}")


if __name__ == "__main__":
    main()
