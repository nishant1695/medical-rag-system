"""
Core configuration settings for Medical RAG System.
Loads from environment variables.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Application
    APP_NAME: str = "Medical RAG System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    BASE_DIR: Path = Path(__file__).parent.parent.parent

    # API
    API_V1_PREFIX: str = "/api/v1"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://medrag:medrag@localhost:5432/medrag"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_SESSION_TTL: int = 3600  # 1 hour

    # Vector Store
    VECTOR_STORE_TYPE: str = "chromadb"  # or "qdrant"
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 8001
    CHROMADB_LOCAL_PATH: Optional[str] = "./data/chromadb"  # Embedded by default; set to "" to use HTTP server
    QDRANT_URL: str = "http://localhost:6333"
    VECTOR_DIM: int = 768  # PubMedBERT dimension

    # Embeddings
    EMBEDDING_MODEL: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    EMBEDDING_DEVICE: str = "cpu"  # or "cuda" or "mps"
    EMBEDDING_BATCH_SIZE: int = 32

    # LLM
    LLM_PROVIDER: str = "anthropic"  # anthropic, openai, gemini, ollama
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    LLM_MODEL: str = "claude-3-5-sonnet-20241022"  # or gpt-4, gemini-2.0-flash-exp
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_OUTPUT_TOKENS: int = 4000

    # Document Processing
    NEXUSRAG_ENABLE_IMAGE_EXTRACTION: bool = True
    NEXUSRAG_ENABLE_IMAGE_CAPTIONING: bool = True
    NEXUSRAG_ENABLE_TABLE_CAPTIONING: bool = True
    NEXUSRAG_ENABLE_FORMULA_ENRICHMENT: bool = True
    NEXUSRAG_DOCLING_IMAGES_SCALE: float = 2.0
    NEXUSRAG_MAX_IMAGES_PER_DOC: int = 20
    NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS: int = 2000

    # Chunking
    NEXUSRAG_CHUNK_MAX_TOKENS: int = 512

    # Retrieval
    NEXUSRAG_VECTOR_PREFETCH: int = 20  # Over-fetch for reranking
    NEXUSRAG_RERANKER_TOP_K: int = 8
    NEXUSRAG_MIN_RELEVANCE_SCORE: float = 0.3
    NEXUSRAG_ENABLE_KG: bool = True
    NEXUSRAG_KG_QUERY_TIMEOUT: int = 10  # seconds

    # Reranker
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_DEVICE: str = "cpu"

    # PubMed
    PUBMED_EMAIL: Optional[str] = None
    PUBMED_API_KEY: Optional[str] = None
    PUBMED_BATCH_SIZE: int = 100

    # Medical NER
    SCISPACY_MODEL: str = "en_core_sci_md"

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]


settings = Settings()
