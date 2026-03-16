"""
Medical RAG Service

Orchestrates document processing and querying for medical RAG system.
Integrates medical document parser, embeddings, vector store, and retrieval.
"""
import logging
from pathlib import Path
from typing import Optional, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Document, DocumentStatus
from app.services.medical_document_parser import MedicalDocumentParser
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.retrieval import get_retriever, RetrievalResult
from app.core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


class MedicalRAGService:
    """
    Medical RAG service for document processing and querying.
    """

    def __init__(self, db: AsyncSession, workspace_id: int):
        self.db = db
        self.workspace_id = workspace_id
        self.parser = MedicalDocumentParser(workspace_id)
        self.embedder = get_embedding_service()
        self.vector_store = get_vector_store(workspace_id)
        self.retriever = get_retriever(workspace_id)

    async def process_document(
        self,
        document_id: int,
        file_path: str,
        pmid: Optional[str] = None,
        subspecialty: Optional[str] = None,
    ) -> int:
        """
        Process a document through the full pipeline.

        Args:
            document_id: Database document ID
            file_path: Path to document file
            pmid: Optional PubMed ID

        Returns:
            Number of chunks created
        """
        # Get document from database
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        if not document:
            raise DocumentProcessingError(f"Document {document_id} not found")

        try:
            # Update status
            document.status = DocumentStatus.PARSING
            await self.db.commit()

            # Parse document with medical enhancements
            logger.info(f"Parsing document {document_id}: {file_path}")
            parsed = self.parser.parse(
                file_path=file_path,
                document_id=document_id,
                original_filename=document.original_filename,
                pmid=pmid,
            )

            # Update document metadata
            document.markdown_content = parsed.markdown
            document.page_count = parsed.page_count
            document.chunk_count = len(parsed.chunks)
            document.table_count = parsed.tables_count
            document.image_count = len(parsed.images)

            # Medical metadata
            document.pmid = parsed.pmid
            document.doi = parsed.doi
            document.title = parsed.title
            document.abstract = parsed.abstract
            document.authors = parsed.authors
            document.journal = parsed.journal
            document.publication_year = parsed.publication_year
            document.mesh_terms = parsed.mesh_terms
            document.study_design = parsed.study_design
            document.evidence_level = parsed.evidence_level
            document.sample_size = parsed.sample_size
            document.limitations = parsed.limitations
            document.paper_url = parsed.paper_url
            document.subspecialty = subspecialty or ""

            document.parser_version = "medical_v1"
            await self.db.commit()

            # Indexing phase
            document.status = DocumentStatus.INDEXING
            await self.db.commit()

            # Generate embeddings
            if parsed.chunks:
                logger.info(f"Embedding {len(parsed.chunks)} chunks")
                chunk_texts = [c["content"] for c in parsed.chunks]
                embeddings = self.embedder.embed_texts(chunk_texts, show_progress=True)

                # Prepare for vector store
                ids = [
                    f"doc_{document_id}_chunk_{i}" for i in range(len(parsed.chunks))
                ]

                metadatas = []
                for chunk in parsed.chunks:
                    metadata = {
                        "document_id": document_id,
                        "chunk_index": chunk["chunk_index"],
                        "source": chunk["source_file"],
                        "page_no": chunk["page_no"],
                        "heading_path": " > ".join(chunk.get("heading_path", [])),
                        # Medical metadata
                        "pmid": parsed.pmid or "",
                        "evidence_level": parsed.evidence_level or "",
                        "study_design": parsed.study_design or "",
                        "paper_url": parsed.paper_url or "",
                        "subspecialty": subspecialty or "",
                    }
                    metadatas.append(metadata)

                # Add to vector store
                logger.info(f"Indexing {len(parsed.chunks)} chunks in vector store")
                self.vector_store.add_documents(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=chunk_texts,
                    metadatas=metadatas,
                )

            # Mark as indexed
            document.status = DocumentStatus.INDEXED
            await self.db.commit()

            # Build knowledge graph (errors here don't fail the overall ingest)
            try:
                from app.services.knowledge_graph import get_knowledge_graph
                from app.services.llm import get_llm_provider
                if parsed.chunks:
                    kg = get_knowledge_graph(self.workspace_id)
                    provider = get_llm_provider()
                    n_edges = await kg.build_from_document(
                        parsed.chunks, self.db, provider, document_id=document_id
                    )
                    logger.info(
                        f"Knowledge graph: {n_edges} new edges for doc {document_id}"
                    )
            except Exception as kg_exc:
                logger.warning(f"Knowledge graph extraction skipped: {kg_exc}")

            logger.info(
                f"Successfully processed document {document_id}: "
                f"{len(parsed.chunks)} chunks, "
                f"PMID={parsed.pmid}, "
                f"evidence_level={parsed.evidence_level}"
            )

            return len(parsed.chunks)

        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)[:500]
            await self.db.commit()
            raise DocumentProcessingError(f"Document processing failed: {e}")

    async def query(
        self,
        question: str,
        top_k: int = 5,
        document_ids: Optional[List[int]] = None,
        subspecialty: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Query the knowledge base.

        Args:
            question: User question
            top_k: Number of results
            document_ids: Optional filter to specific documents
            subspecialty: Optional subspecialty filter (e.g. "breast", "hand")

        Returns:
            RetrievalResult with chunks and citations
        """
        return await self.retriever.retrieve(
            query=question,
            top_k=top_k,
            document_ids=document_ids,
            subspecialty=subspecialty,
        )

    async def get_subspecialty_coverage(self) -> dict[str, int]:
        """Return chunk counts per subspecialty in this workspace."""
        return self.vector_store.count_by_subspecialty()

    async def delete_document(self, document_id: int) -> None:
        """Delete document from vector store and knowledge graph."""
        self.vector_store.delete_by_document_id(document_id)
        logger.info(f"Deleted document {document_id} from vector store")
        try:
            from app.services.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph(self.workspace_id)
            await kg.delete_document_edges(document_id, self.db)
        except Exception as kg_exc:
            logger.warning(f"KG cleanup skipped for document {document_id}: {kg_exc}")

    def get_chunk_count(self) -> int:
        """Get total number of chunks in vector store."""
        return self.vector_store.count()


def get_rag_service(db: AsyncSession, workspace_id: int) -> MedicalRAGService:
    """Get RAG service instance."""
    return MedicalRAGService(db, workspace_id)
