"""
Vector Store Service

Provides vector storage and similarity search using ChromaDB.
Workspace-isolated collections for different subspecialties.
"""
import logging
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.core.exceptions import RetrievalError

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store wrapper for ChromaDB."""

    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self.collection_name = f"workspace_{workspace_id}"
        self._client = None
        self._collection = None

    def _get_client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb

                # Use embedded/persistent client if local path is configured
                if settings.CHROMADB_LOCAL_PATH:
                    import os
                    os.makedirs(settings.CHROMADB_LOCAL_PATH, exist_ok=True)
                    self._client = chromadb.PersistentClient(
                        path=settings.CHROMADB_LOCAL_PATH
                    )
                    logger.info(f"Using embedded ChromaDB at {settings.CHROMADB_LOCAL_PATH}")
                elif settings.VECTOR_STORE_TYPE == "chromadb":
                    self._client = chromadb.HttpClient(
                        host=settings.CHROMADB_HOST,
                        port=settings.CHROMADB_PORT,
                    )
                    logger.info(
                        f"Connected to ChromaDB at {settings.CHROMADB_HOST}:{settings.CHROMADB_PORT}"
                    )
                else:
                    # Fallback to persistent client
                    self._client = chromadb.PersistentClient(
                        path=str(settings.BASE_DIR / "data" / "chromadb")
                    )
                    logger.info("Using persistent ChromaDB client")

            except Exception as e:
                logger.error(f"Failed to connect to ChromaDB: {e}")
                raise RetrievalError(f"Failed to connect to vector store: {e}")

        return self._client

    def _get_collection(self):
        """Get or create collection for this workspace."""
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"workspace_id": self.workspace_id},
                )
                logger.info(f"Using collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to get/create collection: {e}")
                raise RetrievalError(f"Failed to access collection: {e}")

        return self._collection

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            ids: Unique IDs for each document
            embeddings: List of embedding vectors
            documents: List of document texts
            metadatas: List of metadata dicts
        """
        try:
            collection = self._get_collection()
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"Added {len(ids)} documents to {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise RetrievalError(f"Failed to add documents: {e}")

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filters

        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        try:
            collection = self._get_collection()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )

            # Flatten results (ChromaDB returns lists of lists)
            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
            }
        except Exception as e:
            logger.error(f"Failed to query vector store: {e}")
            raise RetrievalError(f"Failed to query vector store: {e}")

    def delete_by_document_id(self, document_id: int) -> None:
        """Delete all chunks for a document."""
        try:
            collection = self._get_collection()
            # Query to find all IDs for this document
            results = collection.get(where={"document_id": document_id})
            if results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info(
                    f"Deleted {len(results['ids'])} chunks for document {document_id}"
                )
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            raise RetrievalError(f"Failed to delete document: {e}")

    def count(self) -> int:
        """Get total number of documents in collection."""
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0

    def count_by_subspecialty(self) -> dict[str, int]:
        """Return chunk counts grouped by subspecialty metadata tag."""
        try:
            collection = self._get_collection()
            all_meta = collection.get(include=["metadatas"])["metadatas"] or []
            counts: dict[str, int] = {}
            for m in all_meta:
                spec = (m.get("subspecialty") or "untagged").strip() or "untagged"
                counts[spec] = counts.get(spec, 0) + 1
            return counts
        except Exception as e:
            logger.error(f"Failed to count by subspecialty: {e}")
            return {}


def get_vector_store(workspace_id: int) -> VectorStore:
    """Get vector store instance for a workspace."""
    return VectorStore(workspace_id)
