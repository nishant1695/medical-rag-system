"""Pydantic schemas for API requests and responses."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Workspace schemas
class WorkspaceCreate(BaseModel):
    """Create workspace request."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    subspecialty: Optional[str] = None
    system_prompt: Optional[str] = None


class WorkspaceResponse(BaseModel):
    """Workspace response."""

    id: int
    name: str
    description: Optional[str]
    subspecialty: Optional[str]
    created_at: datetime
    document_count: int = 0

    class Config:
        from_attributes = True


# Document schemas
class DocumentUpload(BaseModel):
    """Document upload metadata."""

    pmid: Optional[str] = None


class DocumentResponse(BaseModel):
    """Document response."""

    id: int
    workspace_id: int
    original_filename: str
    file_type: Optional[str]
    status: str
    page_count: int
    chunk_count: int

    # Medical metadata
    pmid: Optional[str]
    title: Optional[str]
    authors: Optional[List[str]]
    journal: Optional[str]
    publication_year: Optional[int]
    study_design: Optional[str]
    evidence_level: Optional[str]
    sample_size: Optional[int]

    created_at: datetime

    class Config:
        from_attributes = True


# Chat schemas
class ChatMessage(BaseModel):
    """Chat message."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request."""

    message: str
    history: List[ChatMessage] = []
    enable_thinking: bool = False
    force_search: bool = False


class SourceChunk(BaseModel):
    """Source chunk in response."""

    index: str  # Citation ID (e.g., "a3x9")
    chunk_id: str
    content: str
    document_id: int
    page_no: int
    heading_path: List[str]
    score: float
    pmid: Optional[str] = None
    evidence_level: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response."""

    answer: str
    sources: List[SourceChunk]
    safety_classification: str
    evidence_summary: Dict[str, int]
    thinking: Optional[str] = None


# Search schemas
class SearchRequest(BaseModel):
    """Search request."""

    query: str
    top_k: int = Field(5, ge=1, le=20)
    document_ids: Optional[List[int]] = None


class SearchResponse(BaseModel):
    """Search response."""

    query: str
    chunks: List[SourceChunk]
    evidence_summary: Dict[str, int]
