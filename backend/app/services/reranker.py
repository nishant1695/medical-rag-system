"""
Reranker Service

Cross-encoder reranking for improved retrieval precision.
Uses BGE reranker v2 m3 for joint query-document scoring.
"""
import logging
from typing import List, Optional
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """Reranked result with score."""

    index: int  # Original index in input list
    score: float  # Reranker score
    text: str  # Document text


class RerankerService:
    """Service for reranking retrieved documents."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = settings.RERANKER_DEVICE

    def _load_model(self):
        """Lazy load reranker model."""
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")

                self._tokenizer = AutoTokenizer.from_pretrained(settings.RERANKER_MODEL)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    settings.RERANKER_MODEL
                )

                # Move to device
                if self._device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.cuda()
                elif self._device == "mps" and torch.backends.mps.is_available():
                    self._model = self._model.to("mps")

                self._model.eval()
                logger.info(f"Reranker model loaded on {self._device}")

            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                # Continue without reranking
                self._model = None
                self._tokenizer = None

        return self._model

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[RankedResult]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Query text
            documents: List of document texts to rerank
            top_k: Number of top results to return (None = all)
            min_score: Minimum score threshold (None = no filter)

        Returns:
            List of RankedResult sorted by score (descending)
        """
        if not documents:
            return []

        # Try to load model
        model = self._load_model()
        if model is None:
            # Fallback: return original order with dummy scores
            logger.warning("Reranker model not available, returning original order")
            return [
                RankedResult(index=i, score=1.0 - (i * 0.01), text=doc)
                for i, doc in enumerate(documents)
            ]

        try:
            import torch

            # Prepare pairs
            pairs = [[query, doc] for doc in documents]

            # Tokenize
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )

                # Move to device
                if self._device == "cuda" and torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                elif self._device == "mps" and torch.backends.mps.is_available():
                    inputs = {k: v.to("mps") for k, v in inputs.items()}

                # Get scores
                outputs = model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()

            # Create ranked results
            results = [
                RankedResult(index=i, score=float(score), text=doc)
                for i, (score, doc) in enumerate(zip(scores, documents))
            ]

            # Sort by score (descending)
            results.sort(key=lambda x: x.score, reverse=True)

            # Apply filters
            if min_score is not None:
                results = [r for r in results if r.score >= min_score]

            if top_k is not None:
                results = results[:top_k]

            logger.debug(
                f"Reranked {len(documents)} docs, returned {len(results)} "
                f"(scores: {results[0].score:.3f} to {results[-1].score:.3f})"
                if results
                else "No results after filtering"
            )

            return results

        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original order")
            return [
                RankedResult(index=i, score=1.0 - (i * 0.01), text=doc)
                for i, doc in enumerate(documents)
            ]


# Global instance
_reranker_service: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    """Get or create global reranker service instance."""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
