"""
Medical Document Parser

Extends document parsing with medical domain enhancements:
- PubMed metadata fetching via Biopython
- Study design classification
- Evidence level grading
- Medical entity extraction using scispaCy
- Enhanced chunking with medical context
"""
import logging
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Any

from Bio import Entrez, Medline

from app.core.config import settings
from app.models import StudyDesign, EvidenceLevel

logger = logging.getLogger(__name__)

# Configure Entrez
if settings.PUBMED_EMAIL:
    Entrez.email = settings.PUBMED_EMAIL
if settings.PUBMED_API_KEY:
    Entrez.api_key = settings.PUBMED_API_KEY


class ParsedMedicalDocument:
    """Container for parsed medical document with enhanced metadata."""

    def __init__(self):
        self.document_id: int = 0
        self.original_filename: str = ""
        self.markdown: str = ""
        self.page_count: int = 0
        self.chunks: List[Any] = []
        self.images: List[Any] = []
        self.tables: List[Any] = []
        self.tables_count: int = 0

        # Medical metadata
        self.pmid: Optional[str] = None
        self.doi: Optional[str] = None
        self.title: str = ""
        self.abstract: str = ""
        self.authors: List[str] = []
        self.journal: str = ""
        self.publication_year: Optional[int] = None
        self.mesh_terms: List[str] = []
        self.study_design: str = StudyDesign.UNKNOWN.value
        self.evidence_level: str = EvidenceLevel.LEVEL_V.value
        self.sample_size: Optional[int] = None
        self.limitations: List[str] = []
        self.medical_entities: List[Dict[str, Any]] = []

        # Resolved online link (doi.org, PubMed, or CrossRef lookup)
        self.paper_url: Optional[str] = None


class MedicalDocumentParser:
    """
    Parses medical research papers with domain-specific enhancements.

    Uses Docling for structural parsing + medical metadata extraction.
    """

    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self.output_dir = settings.BASE_DIR / "data" / "docling" / f"kb_{workspace_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._spacy_nlp = None

    def _get_spacy_nlp(self):
        """Lazy load scispaCy model."""
        if self._spacy_nlp is None:
            import spacy
            try:
                self._spacy_nlp = spacy.load(settings.SCISPACY_MODEL)
            except OSError:
                logger.warning(
                    f"scispaCy model {settings.SCISPACY_MODEL} not found. "
                    "Run: python -m spacy download en_core_sci_md"
                )
                # Fallback to regular spacy
                self._spacy_nlp = spacy.load("en_core_web_sm")
        return self._spacy_nlp

    def parse(
        self,
        file_path: str | Path,
        document_id: int,
        original_filename: str,
        pmid: Optional[str] = None,
    ) -> ParsedMedicalDocument:
        """
        Parse medical research paper.

        Args:
            file_path: Path to PDF/document file
            document_id: Database document ID
            original_filename: Original filename
            pmid: Optional PubMed ID for metadata fetching

        Returns:
            ParsedMedicalDocument with full medical metadata
        """
        start_time = time.time()
        result = ParsedMedicalDocument()
        result.document_id = document_id
        result.original_filename = original_filename

        # Step 1: Fetch PubMed metadata if PMID provided
        if pmid:
            logger.info(f"Fetching PubMed metadata for PMID: {pmid}")
            pubmed_meta = self._fetch_pubmed_metadata(pmid)
            if pubmed_meta:
                result.pmid = pmid
                result.doi = pubmed_meta.get("doi", "")
                result.title = pubmed_meta.get("title", "")
                result.abstract = pubmed_meta.get("abstract", "")
                result.authors = pubmed_meta.get("authors", [])
                result.journal = pubmed_meta.get("journal", "")
                result.publication_year = pubmed_meta.get("year")
                result.mesh_terms = pubmed_meta.get("mesh_terms", [])

        # Step 2: Parse document with Docling (structure extraction)
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            parsed = self._parse_with_docling(path, document_id, original_filename)
        elif suffix == ".json":
            parsed = self._parse_pubmed_json(path, result)
        elif suffix == ".txt":
            parsed = self._parse_plain_text(path)
        else:
            parsed = self._parse_text_file(path)

        result.markdown = parsed.get("markdown", "")
        result.page_count = parsed.get("page_count", 0)
        result.images = parsed.get("images", [])
        result.tables = parsed.get("tables", [])
        result.tables_count = len(result.tables)

        # Step 2b: Extract metadata from PDF text when not populated by PubMed
        if not result.doi:
            result.doi = self._extract_doi_from_text(result.markdown) or ""
        if not result.authors:
            result.authors = self._extract_authors_from_pdf(result.markdown)
        if not result.publication_year:
            result.publication_year = self._extract_year_from_text(result.markdown)
        if not result.journal:
            result.journal = self._extract_journal_from_text(result.markdown) or ""

        # Step 3: Extract medical metadata from text if not from PubMed
        full_text = result.markdown or ""
        if result.title:
            full_text = result.title + "\n\n" + full_text
        if result.abstract:
            full_text = full_text[:5000] + "\n\n" + result.abstract

        if not result.title:
            result.title = self._extract_title(full_text)
        if not result.abstract:
            result.abstract = self._extract_abstract(result.markdown)

        # Step 4: Classify study design
        result.study_design = self._classify_study_design(full_text)

        # Step 5: Grade evidence level
        result.evidence_level = self._grade_evidence(result.study_design, result.publication_year)

        # Step 6: Extract sample size
        result.sample_size = self._extract_sample_size(full_text)

        # Step 7: Extract limitations
        result.limitations = self._extract_limitations(result.markdown)

        # Step 8: Extract medical entities
        result.medical_entities = self._extract_medical_entities(full_text)

        # Step 9: Create chunks with medical context
        result.chunks = self._create_medical_chunks(
            parsed.get("raw_chunks", []),
            result,
        )

        # Resolve online paper URL
        result.paper_url = self._resolve_paper_url(
            result.doi or None, result.pmid, result.title
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Parsed medical document {document_id} ({original_filename}) in {elapsed_ms}ms: "
            f"{result.page_count} pages, {len(result.chunks)} chunks, "
            f"{len(result.images)} images, {result.tables_count} tables, "
            f"study_design={result.study_design}, evidence_level={result.evidence_level}"
        )

        return result

    # ── PDF metadata extraction helpers ───────────────────────────────────────

    def _extract_doi_from_text(self, text: str) -> Optional[str]:
        """Extract DOI from document text via regex."""
        match = re.search(r'\b(10\.\d{4,9}/[^\s\]\[>"\'<,;)]+)', text)
        if match:
            return match.group(1).rstrip('.')
        return None

    def _extract_authors_from_pdf(self, markdown: str) -> List[str]:
        """Heuristic extraction of author names from PDF markdown header."""
        header = markdown[:3000]

        # Pattern 1: explicit "Author(s):" label
        authors_match = re.search(
            r'(?:authors?|by)[:\s]+([A-Z][a-z]+(?:[\s\-][A-Z][a-z]+)*'
            r'(?:,\s*[A-Z][a-z]+(?:[\s\-][A-Z][a-z]+)*){1,})',
            header, re.IGNORECASE,
        )
        if authors_match:
            raw = authors_match.group(1)
            parts = [a.strip() for a in re.split(r',\s*', raw) if a.strip()]
            if len(parts) >= 2:
                return parts[:10]

        # Pattern 2: line of comma-separated "Lastname Initial" entries
        for line in header.split('\n')[:30]:
            line = line.strip()
            if not 10 <= len(line) <= 300:
                continue
            if re.search(r'\d{4}', line):  # skip lines with years
                continue
            if re.match(
                r'^[A-Z][a-z]+\s+[A-Z][a-z]?\s*(?:,\s*[A-Z][a-z]+\s+[A-Z][a-z]?\s*){2,}',
                line,
            ):
                parts = [p.strip() for p in re.split(r',\s*', line) if p.strip()]
                if 2 <= len(parts) <= 15:
                    return parts[:10]

        return []

    def _extract_year_from_text(self, text: str) -> Optional[int]:
        """Extract most recent plausible publication year from document header."""
        header = text[:2000]
        matches = re.findall(r'\b(19[0-9]{2}|20[0-2][0-9])\b', header)
        if matches:
            years = [int(y) for y in matches if 1950 <= int(y) <= 2030]
            if years:
                return max(years)
        return None

    def _extract_journal_from_text(self, text: str) -> Optional[str]:
        """Heuristic extraction of journal name from PDF header/footer text."""
        header = text[:3000]
        match = re.search(
            r'\b((?:Journal|Annals|Archives|British|European|American|International|'
            r'Plastic|Aesthetic|Reconstructive|Microsurgery|Surgery)[^\n]{5,80})',
            header, re.IGNORECASE,
        )
        if match:
            candidate = match.group(1).strip()
            if len(candidate) <= 80 and not re.search(
                r'\b(the|this|study|patients?|results?|we|were|was)\b',
                candidate, re.I,
            ):
                return candidate
        return None

    def _resolve_paper_url(
        self,
        doi: Optional[str],
        pmid: Optional[str],
        title: str,
    ) -> Optional[str]:
        """
        Resolve an online URL for the paper.

        Priority:
          1. DOI  → https://doi.org/{doi}
          2. PMID → https://pubmed.ncbi.nlm.nih.gov/{pmid}/
          3. Title → CrossRef API lookup → DOI URL
        """
        if doi:
            return f"https://doi.org/{doi}"
        if pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if title:
            try:
                import json as _json
                import urllib.parse
                import urllib.request

                encoded = urllib.parse.quote(title[:200])
                url = (
                    f"https://api.crossref.org/works"
                    f"?query.bibliographic={encoded}&rows=1&select=DOI"
                )
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": (
                            f"MedicalRAG/1.0 (mailto:{settings.PUBMED_EMAIL})"
                        )
                    },
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = _json.loads(resp.read())
                items = data.get("message", {}).get("items", [])
                if items and items[0].get("DOI"):
                    return f"https://doi.org/{items[0]['DOI']}"
            except Exception as e:
                logger.debug(f"CrossRef lookup failed for '{title[:50]}': {e}")
        return None

    # ── PubMed fetch ────────────────────────────────────────────────────────────

    def _fetch_pubmed_metadata(self, pmid: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata from PubMed API."""
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
            record = Medline.read(handle)

            # Extract year from publication date
            year = None
            pub_date = record.get("DP", "")
            if pub_date:
                year_match = re.search(r"\d{4}", pub_date)
                if year_match:
                    year = int(year_match.group())

            # Extract DOI from article identifiers
            doi = ""
            for aid in record.get("AID", []):
                if "doi" in aid.lower():
                    doi = aid.split("[")[0].strip()
                    break

            return {
                "pmid": pmid,
                "title": record.get("TI", ""),
                "abstract": record.get("AB", ""),
                "authors": record.get("AU", []),
                "journal": record.get("JT", ""),
                "year": year,
                "doi": doi,
                "mesh_terms": record.get("MH", []),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch PubMed metadata for {pmid}: {e}")
            return None

    def _parse_with_docling(
        self, file_path: Path, document_id: int, original_filename: str
    ) -> Dict[str, Any]:
        """Parse PDF using Docling for structure preservation."""
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling_core.transforms.chunker import HybridChunker

            # Configure Docling pipeline
            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_picture_images = settings.NEXUSRAG_ENABLE_IMAGE_EXTRACTION
            pipeline_options.images_scale = settings.NEXUSRAG_DOCLING_IMAGES_SCALE
            pipeline_options.do_formula_enrichment = settings.NEXUSRAG_ENABLE_FORMULA_ENRICHMENT

            converter = DocumentConverter(
                format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
            )

            # Convert document
            conv_result = converter.convert(str(file_path))
            doc = conv_result.document

            # Export to markdown
            markdown = doc.export_to_markdown()

            # Get page count
            page_count = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 0

            # Extract images (simplified for now)
            images = []
            # TODO: Implement image extraction with captioning

            # Extract tables
            tables = []
            # TODO: Implement table extraction

            # Chunk document
            chunker = HybridChunker(
                max_tokens=settings.NEXUSRAG_CHUNK_MAX_TOKENS,
                merge_peers=True,
            )
            raw_chunks = []
            for i, chunk in enumerate(chunker.chunk(doc)):
                chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)

                # Extract metadata
                page_no = 0
                heading_path = []
                if hasattr(chunk, "meta") and chunk.meta:
                    if hasattr(chunk.meta, "page"):
                        page_no = chunk.meta.page or 0
                    if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                        heading_path = list(chunk.meta.headings)

                raw_chunks.append({
                    "content": chunk_text,
                    "chunk_index": i,
                    "page_no": page_no,
                    "heading_path": heading_path,
                })

            return {
                "markdown": markdown,
                "page_count": page_count,
                "images": images,
                "tables": tables,
                "raw_chunks": raw_chunks,
            }

        except Exception as e:
            logger.error(f"Docling parsing failed for {file_path}: {e}")
            # Fallback to simple text extraction
            return self._parse_text_file(file_path)

    def _parse_pubmed_json(self, file_path: Path, result: "ParsedMedicalDocument") -> Dict[str, Any]:
        """
        Parse a PubMed JSON file produced by fetch_pubmed.py.
        Populates result metadata from the JSON and creates a rich text body.
        """
        import json as _json
        try:
            with open(file_path, encoding="utf-8") as f:
                meta = _json.load(f)

            # Fill result metadata from stored JSON (avoids redundant Entrez call)
            if not result.title:
                result.title = meta.get("title", "")
            if not result.abstract:
                result.abstract = meta.get("abstract", "")
            if not result.authors:
                result.authors = meta.get("authors", [])
            if not result.journal:
                result.journal = meta.get("journal", "")
            if result.publication_year is None:
                result.publication_year = meta.get("year")
            if not result.mesh_terms:
                result.mesh_terms = meta.get("mesh_terms", [])
            if not result.doi:
                result.doi = meta.get("doi", "")

            # Build a structured plain-text representation
            authors_str = ", ".join(result.authors[:6])
            if len(result.authors) > 6:
                authors_str += f" et al."
            year_str = str(result.publication_year) if result.publication_year else "n.d."
            mesh_str = "; ".join((result.mesh_terms or [])[:10])

            markdown = (
                f"# {result.title}\n\n"
                f"**Authors:** {authors_str}\n"
                f"**Journal:** {result.journal} ({year_str})\n"
                f"**PMID:** {meta.get('pmid', '')}\n"
            )
            if result.doi:
                markdown += f"**DOI:** {result.doi}\n"
            if mesh_str:
                markdown += f"**MeSH Terms:** {mesh_str}\n"
            markdown += f"\n## Abstract\n\n{result.abstract}\n"

            return self._chunk_plain_text(markdown, page_count=1)

        except Exception as e:
            logger.error(f"PubMed JSON parsing failed for {file_path}: {e}")
            return {"markdown": "", "page_count": 0, "images": [], "tables": [], "raw_chunks": []}

    def _parse_plain_text(self, file_path: Path) -> Dict[str, Any]:
        """Parse a plain .txt file (e.g. from fetch_pubmed.py)."""
        try:
            text = file_path.read_text(encoding="utf-8")
            return self._chunk_plain_text(text, page_count=1)
        except Exception as e:
            logger.error(f"Plain text parsing failed for {file_path}: {e}")
            return {"markdown": "", "page_count": 0, "images": [], "tables": [], "raw_chunks": []}

    def _chunk_plain_text(self, text: str, page_count: int = 1) -> Dict[str, Any]:
        """Chunk a plain text / markdown string into overlapping windows."""
        words = text.split()
        chunk_size = 400   # ~512 tokens at ~0.75 tokens/word
        overlap = 50
        raw_chunks = []
        stride = max(1, chunk_size - overlap)
        for i in range(0, max(1, len(words)), stride):
            chunk_words = words[i: i + chunk_size]
            if not chunk_words:
                break
            raw_chunks.append({
                "content": " ".join(chunk_words),
                "chunk_index": len(raw_chunks),
                "page_no": 0,
                "heading_path": [],
            })
        return {
            "markdown": text,
            "page_count": page_count,
            "images": [],
            "tables": [],
            "raw_chunks": raw_chunks,
        }

    def _parse_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Fallback: try PyMuPDF for PDFs, else plain text."""
        try:
            import pymupdf

            doc = pymupdf.open(str(file_path))
            text_pages = [page.get_text() for page in doc]
            markdown = "\n\n".join(text_pages)
            return self._chunk_plain_text(markdown, page_count=len(doc))
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {file_path}: {e} — falling back to plain text")
            return self._parse_plain_text(file_path)

    def _extract_title(self, text: str) -> str:
        """Extract title from document text (first significant line)."""
        lines = text.split("\n")
        for line in lines[:20]:
            line = line.strip()
            if len(line) > 20 and len(line) < 300 and not line.startswith("#"):
                return line
        return ""

    def _extract_abstract(self, markdown: str) -> str:
        """Extract abstract section from markdown."""
        # Look for Abstract heading
        abstract_match = re.search(
            r"#+\s*Abstract\s*\n+(.*?)(?=\n#+|\Z)", markdown, re.DOTALL | re.IGNORECASE
        )
        if abstract_match:
            return abstract_match.group(1).strip()[:2000]
        return ""

    def _classify_study_design(self, text: str) -> str:
        """Classify study design from text."""
        text_lower = text.lower()[:5000]

        if any(
            term in text_lower
            for term in ["meta-analysis", "meta analysis", "systematic review"]
        ):
            return StudyDesign.META_ANALYSIS.value
        elif any(
            term in text_lower for term in ["randomized controlled trial", "rct", "randomized"]
        ):
            return StudyDesign.RCT.value
        elif "prospective" in text_lower and "cohort" in text_lower:
            return StudyDesign.PROSPECTIVE_COHORT.value
        elif "retrospective" in text_lower:
            return StudyDesign.RETROSPECTIVE_COHORT.value
        elif "case-control" in text_lower or "case control" in text_lower:
            return StudyDesign.CASE_CONTROL.value
        elif "case series" in text_lower or "case report" in text_lower:
            return StudyDesign.CASE_SERIES.value
        else:
            return StudyDesign.UNKNOWN.value

    def _grade_evidence(self, study_design: str, publication_year: Optional[int]) -> str:
        """Grade evidence level (Oxford Centre for Evidence-Based Medicine)."""
        if study_design == StudyDesign.META_ANALYSIS.value:
            return EvidenceLevel.LEVEL_I.value
        elif study_design == StudyDesign.RCT.value:
            return EvidenceLevel.LEVEL_I.value
        elif study_design == StudyDesign.PROSPECTIVE_COHORT.value:
            return EvidenceLevel.LEVEL_II.value
        elif study_design in [
            StudyDesign.RETROSPECTIVE_COHORT.value,
            StudyDesign.CASE_CONTROL.value,
        ]:
            return EvidenceLevel.LEVEL_III.value
        elif study_design == StudyDesign.CASE_SERIES.value:
            return EvidenceLevel.LEVEL_IV.value
        else:
            return EvidenceLevel.LEVEL_V.value

    def _extract_sample_size(self, text: str) -> Optional[int]:
        """Extract sample size (n=XXX) from text."""
        # Look for n = XXX patterns
        n_match = re.search(r"\bn\s*[=:]\s*(\d+)", text.lower())
        if n_match:
            return int(n_match.group(1))

        # Look for "XXX patients/subjects/participants"
        patients_match = re.search(r"(\d+)\s+(patients|subjects|participants)", text.lower())
        if patients_match:
            return int(patients_match.group(1))

        return None

    def _extract_limitations(self, markdown: str) -> List[str]:
        """Extract study limitations from markdown."""
        limitations = []

        # Look for Limitations section
        limitations_match = re.search(
            r"#+\s*Limitations?\s*\n+(.*?)(?=\n#+|\Z)", markdown, re.DOTALL | re.IGNORECASE
        )
        if limitations_match:
            limitations_text = limitations_match.group(1).strip()[:1000]

            # Common limitations
            if "retrospective" in limitations_text.lower():
                limitations.append("Retrospective design")
            if "single-center" in limitations_text.lower() or "single center" in limitations_text.lower():
                limitations.append("Single-center study")
            if "small sample" in limitations_text.lower():
                limitations.append("Small sample size")
            if "selection bias" in limitations_text.lower():
                limitations.append("Potential selection bias")

        return limitations

    def _extract_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical entities using scispaCy."""
        try:
            nlp = self._get_spacy_nlp()
            # Limit text for performance
            doc = nlp(text[:50000])

            entities = []
            seen = set()
            for ent in doc.ents:
                if ent.label_ in ["DISEASE", "CHEMICAL", "PROCEDURE"] and ent.text not in seen:
                    entities.append({
                        "text": ent.text,
                        "type": ent.label_.lower(),
                        "start": ent.start_char,
                        "end": ent.end_char,
                    })
                    seen.add(ent.text)

            return entities[:100]  # Limit to avoid huge lists
        except Exception as e:
            logger.warning(f"Medical entity extraction failed: {e}")
            return []

    def _create_medical_chunks(
        self,
        raw_chunks: List[Dict[str, Any]],
        doc_result: ParsedMedicalDocument,
    ) -> List[Dict[str, Any]]:
        """Enrich chunks with medical context headers."""
        enriched_chunks = []

        for chunk_data in raw_chunks:
            # Build medical context header
            context_header = f"""Study: {doc_result.title or 'Unknown'}
PMID: {doc_result.pmid or 'N/A'}
Study Design: {doc_result.study_design} (Evidence Level: {doc_result.evidence_level})
Sample Size: n={doc_result.sample_size or 'N/A'}
Authors: {', '.join(doc_result.authors[:3])}{'...' if len(doc_result.authors) > 3 else ''}
Journal: {doc_result.journal} ({doc_result.publication_year or 'N/A'})

Section: {' > '.join(chunk_data.get('heading_path', [])) or 'Body'}

Content:
"""

            enriched_content = context_header + chunk_data.get("content", "")

            enriched_chunks.append({
                "content": enriched_content,
                "content_raw": chunk_data.get("content", ""),
                "chunk_index": chunk_data.get("chunk_index", 0),
                "source_file": doc_result.original_filename,
                "document_id": doc_result.document_id,
                "page_no": chunk_data.get("page_no", 0),
                "heading_path": chunk_data.get("heading_path", []),
            })

        return enriched_chunks
