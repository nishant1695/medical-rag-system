"""Database models for Medical RAG System."""
from datetime import datetime
from enum import Enum
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
    BigInteger,
    Float,
    JSON,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PARSING = "parsing"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class StudyDesign(str, Enum):
    """Study design types."""

    META_ANALYSIS = "meta_analysis"
    RCT = "rct"
    PROSPECTIVE_COHORT = "prospective_cohort"
    RETROSPECTIVE_COHORT = "retrospective_cohort"
    CASE_CONTROL = "case_control"
    CASE_SERIES = "case_series"
    EXPERT_OPINION = "expert_opinion"
    UNKNOWN = "unknown"


class EvidenceLevel(str, Enum):
    """Evidence levels (Oxford CEBM)."""

    LEVEL_I = "I"
    LEVEL_II = "II"
    LEVEL_III = "III"
    LEVEL_IV = "IV"
    LEVEL_V = "V"


class KnowledgeBase(Base):
    """Workspace/subspecialty knowledge base."""

    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    system_prompt = Column(Text)
    subspecialty = Column(String(100))  # craniofacial, breast, hand, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="knowledge_base", cascade="all, delete-orphan")


class Document(Base):
    """Research paper document."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50))
    file_size = Column(BigInteger)
    file_path = Column(Text)

    # Processing status
    status = Column(String(50), default=DocumentStatus.PENDING.value)
    error_message = Column(Text)

    # Docling parsing results
    markdown_content = Column(Text)
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    table_count = Column(Integer, default=0)
    image_count = Column(Integer, default=0)
    parser_version = Column(String(50))
    processing_time_ms = Column(Integer)

    # Medical metadata
    pmid = Column(String(50), unique=True, index=True)
    doi = Column(String(255))
    title = Column(Text)
    abstract = Column(Text)
    authors = Column(JSON)        # stored as JSON array
    journal = Column(String(255))
    publication_year = Column(Integer)
    publication_date = Column(DateTime)
    mesh_terms = Column(JSON)     # stored as JSON array

    # Online link (doi.org, PubMed, or CrossRef-resolved URL)
    paper_url = Column(Text)

    # Subspecialty tag (breast, hand, craniofacial, microsurgery, burns, aesthetic, lower_extremity)
    subspecialty = Column(String(100))

    # Study characteristics
    study_design = Column(String(100))
    evidence_level = Column(String(10))
    sample_size = Column(Integer)
    study_population = Column(Text)
    limitations = Column(JSON)    # stored as JSON array

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    images = relationship("DocumentImage", back_populates="document", cascade="all, delete-orphan")
    tables = relationship("DocumentTable", back_populates="document", cascade="all, delete-orphan")


class DocumentImage(Base):
    """Extracted image from document."""

    __tablename__ = "document_images"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    image_id = Column(String(255), unique=True, index=True, nullable=False)
    page_no = Column(Integer)
    file_path = Column(Text)
    caption = Column(Text)
    width = Column(Integer)
    height = Column(Integer)
    mime_type = Column(String(50), default="image/png")

    # Medical image classification
    image_type = Column(String(50))  # diagram, chart, photo, before_after, surgical, microscopy

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="images")


class DocumentTable(Base):
    """Extracted table from document."""

    __tablename__ = "document_tables"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    table_id = Column(String(255), unique=True, index=True, nullable=False)
    page_no = Column(Integer)
    content_markdown = Column(Text)
    caption = Column(Text)
    num_rows = Column(Integer)
    num_cols = Column(Integer)

    # Medical table classification
    table_type = Column(String(50))  # demographics, outcomes, complications, statistics

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="tables")


class ChatMessage(Base):
    """Chat message with medical enhancements."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    message_id = Column(String(255), unique=True, index=True, nullable=False)
    role = Column(String(50), nullable=False)  # user, assistant
    content = Column(Text, nullable=False)

    # Sources and citations
    sources = Column(JSON)      # Array of source chunks with PMIDs
    image_refs = Column(JSON)   # Array of image references
    thinking = Column(Text)
    agent_steps = Column(JSON)

    # Medical enhancements
    safety_classification = Column(String(50))   # literature, clinical_query, emergency
    related_entities = Column(JSON)              # Extracted medical entities list
    evidence_quality = Column(JSON)              # Summary: {level_I: 3, level_II: 2, ...}

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="chat_messages")


class MedicalEntity(Base):
    """Cached medical entities extracted from documents."""

    __tablename__ = "medical_entities"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    entity_text = Column(String(255), nullable=False)
    entity_type = Column(String(50), nullable=False)  # condition, procedure, treatment, outcome, anatomy
    umls_cui = Column(String(50))  # UMLS Concept Unique Identifier (optional)
    frequency = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
