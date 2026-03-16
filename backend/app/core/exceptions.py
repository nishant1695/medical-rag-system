"""Custom exceptions for the application."""


class MedicalRAGException(Exception):
    """Base exception for Medical RAG System."""

    pass


class DocumentProcessingError(MedicalRAGException):
    """Raised when document processing fails."""

    pass


class RetrievalError(MedicalRAGException):
    """Raised when retrieval fails."""

    pass


class SafetyViolation(MedicalRAGException):
    """Raised when a query violates safety policies."""

    pass


class EmbeddingError(MedicalRAGException):
    """Raised when embedding generation fails."""

    pass
