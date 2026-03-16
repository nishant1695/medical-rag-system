# Medical RAG System
## Clinical Decision Support for Healthcare Professionals

A production-ready RAG (Retrieval-Augmented Generation) system designed for healthcare professionals. Ask complex clinical questions and receive structured, evidence-graded answers with inline citations sourced directly from peer-reviewed literature.

[![Status](https://img.shields.io/badge/status-active-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Overview

This system ingests research papers (PDF), indexes them by subspecialty, and answers clinical questions using a multi-specialist agentic pipeline. It is intended for **qualified healthcare professionals** seeking rapid evidence synthesis — not a replacement for clinical judgment.

**Typical use cases:**
- "What is the evidence for immediate vs. delayed breast reconstruction after mastectomy?"
- "A 55-year-old with cutaneous melanoma presents with a new thigh lesion. What staging workup and treatment options does the literature support?"
- "Compare DIEP flap vs. TRAM flap outcomes in obese patients."

---

## Key Features

### Multi-Specialist Architecture
- **8 subspecialties** in a single vector store, tagged with metadata: Aesthetic Surgery, Pediatric Plastic Surgery, Microsurgery, Craniofacial Surgery, Supermicrosurgery, Burn Surgery, Wound Care, Hand Surgery
- **Keyword-based classifier** routes queries to the relevant specialty pool(s) — no extra LLM call
- **Parallel retrieval** across matched specialties via `asyncio.gather`
- **Multi-specialist synthesis** when a query spans multiple domains — the answer is structured with per-specialty findings, areas of agreement/disagreement, and a final verdict

### Query Decomposition + HyDE
- **Complexity detection** identifies clinical vignettes, comparative queries, multi-aspect questions, and long queries (>60 words)
- **Query decomposition** breaks complex queries into 2–4 focused sub-questions (one cheap LLM call, JSON output)
- **HyDE (Hypothetical Document Embeddings)** generates a 2–3 sentence hypothetical abstract per sub-question; embedding this paragraph instead of the raw query improves recall for complex clinical questions
- **Chunk deduplication** across all parallel retrievals — chunks are deduplicated by `chunk_id` before cid assignment, keeping the highest-scoring copy

### Clinical Decision Support Mode
- **Safety classifier** routes queries to one of three classes:
  - `literature` — general research questions, evidence summaries
  - `clinical_query` — clinical vignettes, treatment planning, diagnosis assistance (the primary use-case)
  - `emergency` — life-threatening situations (hard-blocked, directs to emergency services)
- **Structured clinical output** for `clinical_query`: Clinical Summary → Evidence-Based Options → Key Considerations → Evidence Quality
- No intrusive warnings for clinical queries — the system is designed for qualified clinicians

### Hybrid Retrieval Pipeline
- **PubMedBERT embeddings** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`, 768-dim) for medical domain semantic search
- **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to score candidates against the query
- **Over-fetch + rerank**: prefetches 3× candidates, reranks, returns top-K
- **Subspecialty filtering** via ChromaDB metadata `where` filters

### Evidence Grading
- Oxford CEBM **Level I–V** classification for all papers and individual claims
- Evidence level displayed alongside every citation
- Evidence quality summary in every response (count by level)

### Conversation Memory
- **Rolling history**: last 8 turns sent verbatim to the LLM
- **Condensed summary** of older turns (up to 40 messages) appended to the system prompt — no extra LLM call, zero extra latency
- **Context-aware query rewriting** for follow-up questions ("what about complications?" → "what about complications of DIEP flap breast reconstruction?")
- **Persistent history** stored in the database, restored on page load

### PDF Parsing & Metadata Extraction
- **Docling** with `HybridChunker` for structure-aware chunking (preserves headings, tables, sections)
- Automatic extraction of DOI, authors, publication year, journal from PDF text
- **CrossRef API** lookup to resolve paper URLs when no DOI/PMID is present
- All metadata stored per-chunk in ChromaDB for retrieval-time access

### Streaming Chat Interface
- **SSE (Server-Sent Events)** streaming — tokens appear as they are generated
- Step-by-step status events: classifying → decomposing → HyDE → retrieving → synthesizing
- Extended thinking support (when enabled in config)
- Abort mid-stream

### Frontend
- React 19 + TypeScript + Vite + TailwindCSS
- Per-message citation panel with evidence badges and paper links
- Specialist context panel for multi-specialty answers
- Workspace management, document upload, semantic search tab
- Persistent conversation with clear-history support

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        React Frontend                               │
│   Chat • Citation Panel • Specialist Context • Document Management  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  SSE streaming (POST /chat)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Medical Safety Classifier                   │   │
│  │   literature  │  clinical_query  │  emergency (blocked)      │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│  ┌───────────────────────────▼─────────────────────────────────┐   │
│  │                    Agentic Chat Pipeline                     │   │
│  │                                                              │   │
│  │  1. Follow-up query rewriting (context-aware)               │   │
│  │  2. Subspecialty classification (keyword, no LLM)           │   │
│  │  3. Complexity detection                                     │   │
│  │     └─ [complex] Decompose → HyDE per sub-question          │   │
│  │  4. Parallel retrieval (sub-queries × subspecialties)       │   │
│  │  5. Deduplicate chunks by chunk_id                          │   │
│  │  6. Synthesizer LLM (structured clinical or literature fmt) │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│  ┌───────────────────────────▼─────────────────────────────────┐   │
│  │                  Hybrid Retrieval                            │   │
│  │   PubMedBERT Embeddings → ChromaDB → Cross-Encoder Rerank   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │               Medical Document Parser                       │    │
│  │   Docling HybridChunker • DOI/CrossRef • Evidence Grading   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  SQLite (aiosqlite)          ChromaDB (embedded PersistentClient)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Subspecialties

| Key | Display Name | Example Keywords |
|---|---|---|
| `aesthetic` | Aesthetic Surgery | rhinoplasty, facelift, liposuction, abdominoplasty |
| `pediatric_plastic` | Pediatric Plastic Surgery | cleft lip/palate, hemangioma, syndactyly, microtia |
| `microsurgery` | Microsurgery | free flap, DIEP, anastomosis, replantation, ALT flap |
| `craniofacial` | Craniofacial Surgery | mandible, orthognathic, Le Fort, distraction osteogenesis |
| `supermicrosurgery` | Supermicrosurgery | lymphedema, LVA, VLNT, ICG lymphography |
| `burn_surgery` | Burn Surgery | TBSA, escharotomy, inhalation injury, Parkland formula |
| `wound_care` | Wound Care | NPWT/VAC, pressure ulcer, diabetic foot, wound bed preparation |
| `hand` | Hand Surgery | Dupuytren, carpal tunnel, flexor/extensor tendon, nerve graft |

Queries that span multiple subspecialties trigger parallel retrieval from each matched pool and a multi-specialist synthesis response.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Anthropic API key (or OpenAI / local Ollama)

### Backend

```bash
cd medical-rag-system/backend

# Create and activate virtualenv
python -m venv ../venv
source ../venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp ../.env.example ../.env
# Edit .env — set LLM_PROVIDER, ANTHROPIC_API_KEY, DATABASE_URL, etc.

# Initialise database (SQLite by default)
DATABASE_URL="sqlite+aiosqlite:///./data/medrag.db" PYTHONPATH=. python ../scripts/init_db.py

# Start API server
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd medical-rag-system/frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Access

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| API | http://localhost:8000 |
| Interactive API docs | http://localhost:8000/docs |

---

## Ingesting Papers

### Single paper via the UI

1. Create a workspace (name it anything — it's your knowledge base)
2. Go to the **Documents** tab
3. Upload a PDF, optionally provide a PMID for automatic metadata enrichment

### Bulk ingest from a directory

```bash
# Ingest all PDFs in a folder, tagged with a subspecialty
python scripts/bulk_ingest_pdfs.py \
    --workspace 1 \
    --dir /path/to/pdfs \
    --subspecialty microsurgery

# Options
#   --workspace   Workspace (knowledge base) ID (required)
#   --dir         Directory containing PDF files (required)
#   --subspecialty  One of: aesthetic, pediatric_plastic, microsurgery,
#                           craniofacial, supermicrosurgery, burn_surgery,
#                           wound_care, hand
#   --pattern     Glob pattern, default *.pdf
#   --limit       Max files to ingest, default 1000
```

The script:
- Skips files already indexed (by filename + workspace)
- Runs the full Docling parse pipeline
- Extracts DOI, authors, year, journal from PDF text
- Looks up paper URL via CrossRef API if no DOI/PMID found
- Embeds and indexes all chunks in ChromaDB with subspecialty metadata

---

## Configuration

### Key environment variables (`.env`)

```bash
# LLM
LLM_PROVIDER=anthropic          # anthropic | openai | ollama
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-opus-4-6       # override model
LLM_MAX_OUTPUT_TOKENS=8000

# Database (SQLite for local, PostgreSQL for production)
DATABASE_URL=sqlite+aiosqlite:///./data/medrag.db

# Vector store (embedded ChromaDB, no server needed)
CHROMADB_LOCAL_PATH=./data/chromadb

# Embeddings
EMBEDDING_MODEL=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
EMBEDDING_DEVICE=cpu             # cpu | cuda | mps

# Retrieval tuning
NEXUSRAG_VECTOR_PREFETCH=20      # candidates fetched before reranking
NEXUSRAG_RERANKER_TOP_K=8        # results after reranking
NEXUSRAG_MIN_RELEVANCE_SCORE=0.3 # minimum reranker score
```

---

## API Reference

### Workspaces

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/workspaces` | List all workspaces |
| `POST` | `/api/v1/workspaces` | Create workspace |
| `DELETE` | `/api/v1/workspaces/{id}` | Delete workspace |

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/workspaces/{id}/documents` | List documents |
| `POST` | `/api/v1/workspaces/{id}/documents/upload` | Upload PDF |
| `DELETE` | `/api/v1/workspaces/{id}/documents/{doc_id}` | Delete document |

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/workspaces/{id}/chat` | SSE streaming chat |
| `POST` | `/api/v1/workspaces/{id}/search` | Semantic search |
| `GET` | `/api/v1/workspaces/{id}/history` | Conversation history |
| `DELETE` | `/api/v1/workspaces/{id}/history` | Clear history |
| `GET` | `/api/v1/workspaces/{id}/stats` | Workspace stats |

### Chat request body

```json
{
  "message": "What is the evidence for DIEP flap in obese patients?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "enable_thinking": false,
  "force_search": true
}
```

### SSE event stream

The chat endpoint streams the following events:

| Event | Payload | Description |
|---|---|---|
| `status` | `{ step, detail, subspecialties? }` | Pipeline progress |
| `thinking` | `{ text }` | Extended thinking tokens (if enabled) |
| `token` | `{ text }` | Answer tokens |
| `sources` | `{ sources[] }` | Retrieved sources (sent before answer) |
| `complete` | `{ answer, sources, safety_classification, evidence_summary, specialist_contexts }` | Final event |
| `error` | `{ message }` | Error |

---

## Project Structure

```
medical-rag-system/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat.py          # Chat, search, history endpoints
│   │   │   ├── workspaces.py    # Workspace CRUD
│   │   │   └── documents.py     # Document upload / management
│   │   ├── core/
│   │   │   ├── config.py        # Settings (pydantic-settings)
│   │   │   └── database.py      # Async SQLAlchemy setup
│   │   ├── models/              # SQLAlchemy ORM models
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   └── services/
│   │       ├── chat_agent.py        # Agentic pipeline (decomp, HyDE, retrieval, synthesis)
│   │       ├── medical_safety_classifier.py  # literature / clinical_query / emergency
│   │       ├── medical_document_parser.py    # Docling parser + metadata extraction
│   │       ├── rag_service.py       # Document processing orchestration
│   │       ├── retrieval.py         # Hybrid retrieval (vector + reranker)
│   │       ├── embeddings.py        # PubMedBERT embedding service
│   │       ├── vector_store.py      # ChromaDB wrapper
│   │       ├── reranker.py          # Cross-encoder reranker
│   │       └── llm/
│   │           ├── base.py          # LLM provider abstraction
│   │           ├── anthropic_provider.py
│   │           ├── openai_provider.py
│   │           └── ollama_provider.py
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── ChatPage.tsx     # Chat interface with streaming
│       │   └── WorkspacePage.tsx
│       ├── components/
│       │   ├── ChatMessage.tsx  # Message bubble + citation + specialist panels
│       │   ├── ChatInput.tsx
│       │   ├── CitationPanel.tsx
│       │   └── DocumentsTab.tsx
│       ├── api.ts               # API client + SSE stream handler
│       └── types.ts             # TypeScript interfaces
├── scripts/
│   ├── bulk_ingest_pdfs.py      # Batch PDF ingestion with subspecialty tagging
│   └── init_db.py               # Database initialisation
└── data/
    ├── medrag.db                # SQLite database
    └── chromadb/                # Embedded ChromaDB vector store
```

---

## Safety & Disclaimer

**This system is a clinical decision support tool for qualified healthcare professionals.**

It is NOT:
- A substitute for clinical judgment
- Approved for direct patient care decisions without professional oversight
- A medical device
- Appropriate for emergency use

The safety classifier routes queries to three classes:

| Class | Behaviour |
|---|---|
| `literature` | General evidence summary — standard response |
| `clinical_query` | Clinical vignette / treatment planning — structured Clinical Decision Support response with disclaimer footer |
| `emergency` | Hard-blocked — directs to emergency services (911) immediately |

All responses include inline citations with evidence levels so every claim is traceable to a source.

---

## Acknowledgements

- **Docling** — Document parsing and structure-aware chunking
- **PubMedBERT** — Medical domain embeddings (Microsoft Research)
- **ChromaDB** — Embedded vector store
- **cross-encoder/ms-marco-MiniLM** — Cross-encoder reranking
- **Oxford CEBM** — Evidence level classification framework
- **CrossRef** — DOI and paper URL resolution API
