# Medical RAG System
## AI-Powered Medical Research Assistant with Evidence-Based Citations

A production-ready RAG (Retrieval-Augmented Generation) system specifically designed for medical research, combining NexusRAG's hybrid retrieval architecture with medical domain enhancements.

[![Status](https://img.shields.io/badge/status-in--development-yellow)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 🌟 Key Features

### Medical Domain Specialization
- **PubMed Integration** - Automatic fetching of research papers with metadata (PMID, authors, journal)
- **Evidence Grading** - Oxford CEBM Level I-V classification for all papers
- **Study Design Detection** - Automatic classification (RCT, meta-analysis, cohort, case series)
- **Medical Entity Extraction** - scispaCy-based extraction of conditions, procedures, treatments
- **Safety Classification** - Detects and blocks patient-specific or emergency queries

### Advanced Retrieval
- **Hybrid Search** - Combines Knowledge Graph + Vector Search + Cross-Encoder Reranking
- **Medical Embeddings** - PubMedBERT for domain-specific semantic search
- **Structure-Aware Chunking** - Preserves document hierarchy and medical context
- **Image & Table Extraction** - Surgical diagrams and outcome tables become searchable

### Citation & Verification
- **PMID-Based Citations** - Every claim links to PubMed source
- **Evidence Transparency** - Level I-V displayed for all citations
- **Inline References** - 4-character citation IDs in answers
- **Verifiable Sources** - Direct links to original papers

### Medical Safety
- **Patient-Specific Detection** - Warns when queries appear to be about specific patients
- **Emergency Blocking** - Refuses to answer emergency medical questions
- **Conservative Language** - "Studies suggest" rather than "You should"
- **Educational Disclaimers** - Clear messaging about system limitations

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Frontend (React)                        │
│  Chat Interface • Citation Display • Evidence Badges        │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│              Medical Safety Classifier                     │
│  Emergency Detection • Patient-Specific Filtering          │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│              Hybrid Retrieval Pipeline                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ Knowledge    │ │ Vector       │ │ Cross-Encoder│      │
│  │ Graph        │ │ Search       │ │ Reranking    │      │
│  │ (LightRAG)   │ │ (ChromaDB)   │ │ (BGE-v2-m3)  │      │
│  └──────────────┘ └──────────────┘ └──────────────┘      │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│            Medical Document Parser                         │
│  Docling • PubMed API • scispaCy • Evidence Grading        │
└────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose**
- **Node.js 18+** (for frontend)
- **PubMed API Key** (free from NCBI)
- **Anthropic API Key** (for Claude) OR local Ollama

### Installation

```bash
# 1. Clone repository
cd ~/GitHub
git clone <repository-url> medical-rag-system
cd medical-rag-system

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Install backend dependencies
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Download medical NLP models
python -m spacy download en_core_sci_md

# 5. Start infrastructure (PostgreSQL, ChromaDB, Redis)
cd ..
docker-compose up -d

# 6. Initialize database
python scripts/init_db.py

# 7. Start backend
cd backend
uvicorn app.main:app --reload --port 8000

# 8. Start frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Access

- **Frontend:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## 📚 Usage

### Basic Query

```python
# Via Python client
from medical_rag_client import MedicalRAG

rag = MedicalRAG(api_url="http://localhost:8000")

# Create a workspace (subspecialty)
workspace = rag.create_workspace(
    name="Breast Surgery",
    subspecialty="breast"
)

# Upload a paper
rag.upload_paper(
    workspace_id=workspace.id,
    file_path="paper.pdf",
    pmid="12345678"  # Optional
)

# Ask a question
response = rag.query(
    workspace_id=workspace.id,
    question="What are the outcomes of DIEP flap breast reconstruction?"
)

# Response includes:
# - answer: Full answer with inline citations
# - sources: List of sources with PMID, evidence level
# - evidence_quality: Distribution of evidence levels
# - safety_classification: "literature" | "patient_specific" | "emergency"
```

### Via Web Interface

1. Create a workspace (e.g., "Breast Surgery")
2. Upload research papers (PDF with optional PMID)
3. Wait for processing (parsing + indexing)
4. Start chatting with citations and evidence levels

### Safety Features

```python
# Literature query (safe) ✅
"What are the complications of DIEP flap surgery?"
→ Returns evidence-based answer with citations

# Patient-specific query (warned) ⚠️
"Should I perform DIEP flap on my 45-year-old patient with BMI 32?"
→ Warns user, provides only general evidence, no recommendations

# Emergency query (blocked) 🚫
"Patient with severe bleeding after surgery, what should I do?"
→ Blocks query, advises calling emergency services
```

---

## 🔧 Configuration

### Environment Variables

```bash
# LLM Provider
LLM_PROVIDER=anthropic  # or openai, gemini, ollama
ANTHROPIC_API_KEY=sk-...

# Database
DATABASE_URL=postgresql+asyncpg://medrag:medrag@localhost:5432/medrag

# PubMed
PUBMED_EMAIL=your-email@example.com
PUBMED_API_KEY=your-pubmed-api-key

# Vector Store
VECTOR_STORE_TYPE=chromadb  # or qdrant
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Embeddings
EMBEDDING_MODEL=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
EMBEDDING_DEVICE=cpu  # or cuda, mps

# Retrieval
NEXUSRAG_VECTOR_PREFETCH=20
NEXUSRAG_RERANKER_TOP_K=8
NEXUSRAG_MIN_RELEVANCE_SCORE=0.3
```

See `.env.example` for full configuration options.

---

## 📊 Database Schema

### Core Tables

- **knowledge_bases** - Workspaces (subspecialties)
- **documents** - Research papers with medical metadata
- **document_images** - Extracted images with captions
- **document_tables** - Extracted tables with structure
- **chat_messages** - Conversation history with citations
- **medical_entities** - Cached medical entities

### Medical Enhancements

Each document includes:
- PMID, DOI, authors, journal
- Study design (RCT, cohort, meta-analysis, etc.)
- Evidence level (I-V)
- Sample size, study population
- Limitations
- MeSH terms

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test specific component
pytest tests/test_medical_parser.py
pytest tests/test_safety_classifier.py

# Test with medical queries
python scripts/test_queries.py --workspace breast_surgery
```

---

## 📈 Evaluation

### Test Queries

```python
# Run evaluation on test set
python scripts/evaluate.py \
    --test-set data/evaluation/breast_surgery_queries.json \
    --workspace breast_surgery \
    --output results.json
```

### Metrics

- **Citation Accuracy**: % of claims with correct PMID
- **Evidence Level Accuracy**: Correct Level I-V classification
- **Safety Recall**: % of patient-specific queries detected
- **Retrieval Quality**: Precision@5, Recall@5

---

## 🔒 Medical Safety & Legal

### Important Disclaimers

⚠️ **FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

This system is designed to help medical professionals explore research literature. It is NOT:
- A substitute for clinical judgment
- Approved for patient care decisions
- A medical device
- Intended for emergency use

**All clinical decisions must be made by qualified healthcare professionals.**

### Safety Features

1. **Patient-Specific Detection** - Blocks queries about specific patients
2. **Emergency Blocking** - Refuses emergency medical questions
3. **Evidence Grading** - Transparent Level I-V classification
4. **PMID Verification** - All claims linked to verifiable sources
5. **Conservative Language** - "Studies suggest" vs. "You should"

### Limitations

- Information may be incomplete or outdated
- RAG systems can occasionally make errors
- Citations should always be verified
- Not a substitute for systematic reviews
- May miss relevant papers not in corpus

---

## 🛠️ Development

### Project Structure

```
medical-rag-system/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI endpoints
│   │   ├── core/          # Config, database, exceptions
│   │   ├── models/        # SQLAlchemy models
│   │   ├── schemas/       # Pydantic schemas
│   │   └── services/      # Business logic
│   │       ├── medical_document_parser.py
│   │       ├── medical_safety_classifier.py
│   │       ├── embeddings.py
│   │       ├── vector_store.py
│   │       └── ...
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── hooks/         # Custom hooks
│   │   └── utils/         # Utilities
│   └── package.json
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── docker-compose.yml
├── .env.example
└── README.md
```

### Adding a New Subspecialty

```python
# 1. Create workspace
rag.create_workspace(
    name="Hand Surgery",
    subspecialty="hand",
    system_prompt="You are a hand surgery research assistant..."
)

# 2. Define PubMed query
query = """
    (hand surgery OR hand reconstruction)
    AND (outcomes OR complications)
    AND ("2015"[Date - Publication] : "2026"[Date - Publication])
"""

# 3. Fetch papers
python scripts/fetch_pubmed.py \
    --workspace hand_surgery \
    --query "$query" \
    --max-papers 200

# 4. Process papers
python scripts/process_papers.py --workspace hand_surgery
```

---

## 📝 API Documentation

### Endpoints

#### Workspaces
- `POST /api/v1/workspaces` - Create workspace
- `GET /api/v1/workspaces` - List workspaces
- `GET /api/v1/workspaces/{id}` - Get workspace

#### Documents
- `POST /api/v1/workspaces/{id}/documents` - Upload document
- `GET /api/v1/workspaces/{id}/documents` - List documents
- `DELETE /api/v1/documents/{id}` - Delete document

#### Chat
- `POST /api/v1/workspaces/{id}/chat` - SSE streaming chat
- `POST /api/v1/workspaces/{id}/search` - Search documents
- `GET /api/v1/workspaces/{id}/chat/history` - Get chat history

See http://localhost:8000/docs for interactive API documentation.

---

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests
5. Run tests (`pytest`)
6. Commit (`git commit -m 'Add amazing feature'`)
7. Push (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📜 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **NexusRAG** - Base architecture and hybrid retrieval
- **Docling** - Document parsing and structure preservation
- **PubMedBERT** - Medical domain embeddings
- **scispaCy** - Medical named entity recognition
- **LightRAG** - Knowledge graph implementation
- **NCBI** - PubMed API access

---

## 📞 Support

- **Documentation**: See IMPLEMENTATION_STATUS.md and POC_COMPREHENSIVE_SOLUTION.md
- **Issues**: GitHub Issues
- **Email**: support@medical-rag.example.com

---

## 🔄 Roadmap

### ✅ Phase 1: Foundation (Current)
- [x] Project structure
- [x] Database models
- [x] Medical document parser
- [x] Safety classifier
- [ ] Hybrid retrieval pipeline

### 🚧 Phase 2: Core Features
- [ ] Agentic chat with streaming
- [ ] Frontend UI
- [ ] PubMed integration
- [ ] Docker deployment

### 📋 Phase 3: Advanced Features
- [ ] Multi-specialist consultation
- [ ] Conversation memory
- [ ] Image/table understanding
- [ ] Evidence synthesis

### 🎯 Phase 4: Production
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Security hardening
- [ ] Deployment documentation

---

**Status**: 🟡 Phase 1 (Foundation) - 60% Complete

**Built with ❤️ for advancing medical research accessibility**
