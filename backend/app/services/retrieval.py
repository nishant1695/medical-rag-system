"""
Medical Retrieval Service

Hybrid retrieval combining vector search with cross-encoder reranking.
Medical domain enhancements include evidence level filtering and PMID tracking.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.core.config import settings
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.reranker import get_reranker_service

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Retrieved document chunk with metadata."""

    content: str
    chunk_id: str
    document_id: int
    page_no: int
    heading_path: List[str]
    score: float

    # Medical metadata
    pmid: Optional[str] = None
    evidence_level: Optional[str] = None
    study_design: Optional[str] = None


@dataclass
class Citation:
    """Citation information for a retrieved chunk."""

    source_file: str
    document_id: int
    page_no: int
    heading_path: List[str]
    pmid: Optional[str] = None
    evidence_level: Optional[str] = None

    def format(self) -> str:
        """Format citation as string."""
        parts = [self.source_file]
        if self.page_no:
            parts.append(f"p.{self.page_no}")
        if self.heading_path:
            parts.append(" > ".join(self.heading_path))
        if self.pmid:
            parts.append(f"PMID:{self.pmid}")
        if self.evidence_level:
            parts.append(f"Level {self.evidence_level}")
        return " | ".join(parts)


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""

    chunks: List[RetrievedChunk]
    citations: List[Citation]
    context: str
    query: str
    evidence_summary: Dict[str, int]  # Count by evidence level


class MedicalRetriever:
    """
    Hybrid retrieval for medical documents.

    Pipeline:
    1. Vector search (over-fetch)
    2. Cross-encoder reranking
    3. Evidence level filtering (optional)
    """

    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self.embedder = get_embedding_service()
        self.vector_store = get_vector_store(workspace_id)
        self.reranker = get_reranker_service()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[int]] = None,
        min_evidence_level: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of final results
            document_ids: Optional filter to specific documents
            min_evidence_level: Optional minimum evidence level (I, II, III, IV, V)

        Returns:
            RetrievalResult with chunks and citations
        """
        # Step 1: Vector search (over-fetch for reranking)
        prefetch_k = max(settings.NEXUSRAG_VECTOR_PREFETCH, top_k * 3)

        query_embedding = self.embedder.embed_query(query)

        where_filter = None
        if document_ids:
            where_filter = {"document_id": {"$in": document_ids}}

        vector_results = self.vector_store.query(
            query_embedding=query_embedding.tolist(),
            n_results=prefetch_k,
            where=where_filter,
        )

        if not vector_results["documents"]:
            logger.warning(f"No results found for query: {query[:100]}")
            return RetrievalResult(
                chunks=[],
                citations=[],
                context="No relevant documents found.",
                query=query,
                evidence_summary={},
            )

        # Step 2: Cross-encoder reranking
        documents = vector_results["documents"]
        reranked = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=settings.NEXUSRAG_RERANKER_TOP_K,
            min_score=settings.NEXUSRAG_MIN_RELEVANCE_SCORE,
        )

        if not reranked:
            logger.warning(
                f"No results passed reranking threshold "
                f"({settings.NEXUSRAG_MIN_RELEVANCE_SCORE})"
            )
            # Fallback: use top vector results
            reranked = self.reranker.rerank(
                query=query,
                documents=documents[:min(5, len(documents))],
                top_k=5,
                min_score=None,
            )

        # Step 3: Build chunks and citations
        chunks = []
        citations = []
        evidence_summary = {}

        for ranked in reranked[:top_k]:
            idx = ranked.index
            metadata = vector_results["metadatas"][idx]

            # Extract heading path
            heading_path = []
            if metadata.get("heading_path"):
                heading_path_str = metadata["heading_path"]
                if isinstance(heading_path_str, str):
                    heading_path = heading_path_str.split(" > ")

            # Get evidence level (from metadata or None)
            evidence_level = metadata.get("evidence_level")
            if evidence_level:
                evidence_summary[evidence_level] = evidence_summary.get(evidence_level, 0) + 1

            chunk = RetrievedChunk(
                content=ranked.text,
                chunk_id=vector_results["ids"][idx],
                document_id=metadata.get("document_id", 0),
                page_no=metadata.get("page_no", 0),
                heading_path=heading_path,
                score=ranked.score,
                pmid=metadata.get("pmid"),
                evidence_level=evidence_level,
                study_design=metadata.get("study_design"),
            )
            chunks.append(chunk)

            citation = Citation(
                source_file=metadata.get("source", "Unknown"),
                document_id=metadata.get("document_id", 0),
                page_no=metadata.get("page_no", 0),
                heading_path=heading_path,
                pmid=metadata.get("pmid"),
                evidence_level=evidence_level,
            )
            citations.append(citation)

        # Step 4: Build context string
        context = self._build_context(chunks, citations)

        logger.info(
            f"Retrieved {len(chunks)} chunks for query (evidence: {evidence_summary})"
        )

        return RetrievalResult(
            chunks=chunks,
            citations=citations,
            context=context,
            query=query,
            evidence_summary=evidence_summary,
        )

    def _build_context(
        self, chunks: List[RetrievedChunk], citations: List[Citation]
    ) -> str:
        """Build formatted context string for LLM."""
        parts = []

        for i, (chunk, citation) in enumerate(zip(chunks, citations)):
            citation_str = citation.format()
            parts.append(f"[{i + 1}] {citation_str}\n{chunk.content}")

        return "\n\n---\n\n".join(parts)


def get_retriever(workspace_id: int) -> MedicalRetriever:
    """Get retriever instance for a workspace."""
    return MedicalRetriever(workspace_id)
