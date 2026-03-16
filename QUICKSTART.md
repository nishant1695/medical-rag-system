# Quick Start Guide - Medical RAG System

## 🚀 Get Started in 5 Minutes

### Step 1: Setup

```bash
cd ~/GitHub/medical-rag-system
./setup.sh
```

This will:
- Create Python virtual environment
- Install all dependencies
- Download medical NLP models
- Start Docker services (PostgreSQL, ChromaDB, Redis)
- Initialize database

### Step 2: Configure API Keys

Edit `.env` file:

```bash
nano .env
```

**Required:**
- `ANTHROPIC_API_KEY=sk-...` (or OpenAI/Gemini)
- `PUBMED_EMAIL=your@email.com`

**Optional:**
- `PUBMED_API_KEY=...` (free from NCBI)

### Step 3: Start Backend

```bash
cd backend
source ../venv/bin/activate
uvicorn app.main:app --reload
```

**API will be available at:**
- 📚 API Docs: http://localhost:8000/docs
- 💚 Health: http://localhost:8000/health

### Step 4: Test the API

**Create a workspace:**

```bash
curl -X POST http://localhost:8000/api/v1/workspaces \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Breast Surgery",
    "subspecialty": "breast",
    "description": "Breast reconstruction research"
  }'
```

**Upload a paper:**

```bash
curl -X POST http://localhost:8000/api/v1/workspaces/1/documents \
  -F "file=@paper.pdf" \
  -F "pmid=12345678"
```

**Search:**

```bash
curl -X POST http://localhost:8000/api/v1/workspaces/1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the outcomes of DIEP flap surgery?",
    "top_k": 5
  }'
```

## 🎯 Features Available

✅ **Core Features (Implemented):**
- Workspace management (subspecialties)
- Document upload with PMID
- Medical document parsing (Docling + PubMed)
- Evidence grading (Level I-V)
- Study design classification
- Medical entity extraction (scispaCy)
- Safety classification (patient-specific detection)
- Hybrid retrieval (vector + reranking)
- PubMedBERT embeddings
- Search API with citations

🚧 **Coming Soon:**
- SSE streaming chat
- Frontend UI
- Multi-specialist consultation
- Knowledge graph integration

## 📊 Architecture

```
User Query
    ↓
Safety Classifier (block patient-specific/emergency)
    ↓
Medical Retriever
  ├─ PubMedBERT Embeddings
  ├─ ChromaDB Vector Search
  └─ Cross-Encoder Reranking
    ↓
Response with:
  ├─ Relevant chunks
  ├─ PMID citations
  ├─ Evidence levels
  └─ Safety warnings
```

## 🔍 Example Queries

**Good (Literature) ✅**
- "What are the complications of DIEP flap surgery?"
- "Outcomes of breast reconstruction in obese patients"
- "Evidence for TRAM flap vs DIEP flap"

**Blocked (Patient-Specific) ⚠️**
- "Should I perform DIEP flap on my 45-year-old patient?"
- "What dose should I give this patient?"

**Blocked (Emergency) 🚫**
- "Patient with severe bleeding, what to do?"
- "Emergency treatment for..."

## 📁 Directory Structure

```
medical-rag-system/
├── backend/
│   ├── app/
│   │   ├── api/          # ✅ Workspaces, documents, chat
│   │   ├── core/         # ✅ Config, database
│   │   ├── models/       # ✅ Database models
│   │   ├── schemas/      # ✅ Pydantic schemas
│   │   └── services/     # ✅ RAG pipeline
│   └── requirements.txt
├── scripts/
│   └── init_db.py        # ✅ Database setup
├── docker-compose.yml    # ✅ Services
├── setup.sh              # ✅ One-command setup
└── README.md
```

## 🐛 Troubleshooting

**Database connection failed:**
```bash
docker-compose up -d postgres
docker-compose ps  # Check status
```

**ChromaDB connection failed:**
```bash
docker-compose up -d chromadb
curl http://localhost:8000/api/v1/heartbeat
```

**Model download failed:**
```bash
python -m spacy download en_core_sci_md
```

**Import errors:**
```bash
source venv/bin/activate
cd backend
pip install -r requirements.txt
```

## 📝 Next Steps

1. **Add more papers:** Upload PDFs with PMIDs
2. **Test queries:** Use the search endpoint
3. **Check evidence:** Review evidence levels in responses
4. **Monitor safety:** Test with patient-specific queries

## 🔗 Resources

- **Full Docs:** README.md
- **Implementation:** IMPLEMENTATION_STATUS.md
- **Design:** POC_COMPREHENSIVE_SOLUTION.md
- **API Docs:** http://localhost:8000/docs

---

**Status:** ✅ Core features ready for testing!
