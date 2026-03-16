"""
Embedding Service

Provides text embedding functionality using PubMedBERT for medical domain.
Supports batch processing and caching for performance.
"""
import logging
from typing import List, Optional
import numpy as np

from app.core.config import settings
from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from text."""

    def __init__(self):
        self._model = None
        self._device = settings.EMBEDDING_DEVICE

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
                self._model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    device=self._device,
                )
                logger.info(f"Embedding model loaded successfully on {self._device}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise EmbeddingError(f"Failed to load embedding model: {e}")

        return self._model

    def embed_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Embed multiple texts.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        try:
            model = self._load_model()
            embeddings = model.encode(
                texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise EmbeddingError(f"Failed to embed texts: {e}")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query text.

        Args:
            query: Query text

        Returns:
            numpy array of shape (embedding_dim,)
        """
        embeddings = self.embed_texts([query])
        return embeddings[0]

    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


# Global instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
