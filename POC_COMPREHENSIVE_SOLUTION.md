# Comprehensive POC Solution: Medical RAG System
## Leveraging NexusRAG Architecture for Medical Research

**Date:** March 16, 2026
**Objective:** Build a production-ready POC medical RAG system by adapting NexusRAG's proven architecture for medical research papers

---

## Executive Summary

After evaluating both **NexusRAG** (a sophisticated hybrid RAG system) and the **medical-rag-system design document**, this document proposes a comprehensive POC that combines the best of both:

### Key Innovations from NexusRAG to Adopt:
1. **Deep Document Parsing with Docling** - Preserves structure, headings, tables, images
2. **Hybrid Retrieval** - Vector search + Knowledge Graph + Cross-encoder reranking
3. **Image & Table Extraction** - Visual content becomes semantically searchable
4. **Structure-Aware Chunking** - Maintains document hierarchy and context
5. **Agentic Streaming Chat** - Real-time responses with inline citations
6. **Workspace Isolation** - Multiple knowledge bases for different subspecialties

### Medical Domain Enhancements:
1. **PubMed Integration** - Automated research paper acquisition
2. **Medical NER** - Extract medical entities, conditions, treatments
3. **Evidence Grading** - Level I-V evidence classification
4. **Safety Classification** - Patient-specific vs literature queries
5. **Multi-Specialist Consultation** - Parallel agent system for subspecialties
6. **Citation Provenance** - PMID-level tracing for medical accuracy

### Architecture Decision:

**Option 1 (RECOMMENDED): NexusRAG-Based Medical System**
- Fork NexusRAG, adapt for medical domain
- Add PubMed integration, medical NER, evidence grading
- Keep proven architecture: Docling + ChromaDB + LightRAG + Cross-encoder
- **Timeline:** 3-4 weeks for full POC

**Option 2: Medical-First with NexusRAG Components**
- Build from medical-rag-system design
- Integrate NexusRAG's Docling parser, hybrid retriever, agentic chat
- Custom multi-agent orchestration for subspecialties
- **Timeline:** 5-6 weeks for full POC

---

## Part 1: Recommended Architecture (NexusRAG-Based Medical System)

### 1.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE (Streamlit/React)                       │
│  - Query Interface | Safety Warnings | Citation Display | Evidence Grading  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MEDICAL QUERY CLASSIFIER & ROUTER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │   Safety     │  │   Medical    │  │ Subspecialty │                     │
│  │  Classifier  │  │     NER      │  │   Router     │                     │
│  └──────────────┘  └──────────────┘  └──────────────┘                     │
│  - Patient-specific detection                                               │
│  - Condition/treatment extraction                                           │
│  - Route to: Craniofacial | Hand | Breast | Burn | Reconstructive         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEXUSRAG HYBRID RETRIEVAL ENGINE                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  1. Knowledge Graph Query (LightRAG)                               │    │
│  │     - Entity: "DIEP flap"                                          │    │
│  │     - Relationships: complications, outcomes, techniques           │    │
│  │     - Medical ontology traversal                                   │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  2. Vector Search Over-Fetch (ChromaDB)                            │    │
│  │     - Embedding: PubMedBERT (medical domain)                       │    │
│  │     - Prefetch: 20 candidates                                      │    │
│  │     - Metadata: PMID, evidence level, study design                 │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  3. Cross-Encoder Reranking (bge-reranker-v2-m3)                   │    │
│  │     - Joint query-chunk scoring                                    │    │
│  │     - Filter by relevance threshold                                │    │
│  │     - Final top-K: 5-8 chunks                                      │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  4. Image & Table Enrichment                                       │    │
│  │     - Find images/tables on same pages                             │    │
│  │     - Include surgical diagrams, outcome tables                    │    │
│  │     - LLM-generated captions for searchability                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DOCLING DOCUMENT PROCESSOR                            │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  PubMed Papers → PDF Download                                      │    │
│  │    ↓                                                                │    │
│  │  Docling Parser:                                                    │    │
│  │    - Preserve heading hierarchy (Introduction, Methods, Results)   │    │
│  │    - Extract tables (outcomes, demographics, complications)        │    │
│  │    - Extract images (surgical diagrams, before/after photos)       │    │
│  │    - Formula enrichment (statistical formulas, p-values)           │    │
│  │    - Page-aware chunking                                           │    │
│  │    ↓                                                                │    │
│  │  Medical Enhancement:                                               │    │
│  │    - Extract PMID, authors, journal, year                          │    │
│  │    - Study design detection (RCT, cohort, case series)             │    │
│  │    - Evidence level classification (I-V)                            │    │
│  │    - Medical entity extraction (conditions, procedures, outcomes)  │    │
│  │    - Extract n= sample size from text                              │    │
│  │    ↓                                                                │    │
│  │  Structure-Aware Chunking:                                          │    │
│  │    - HybridChunker with heading context                            │    │
│  │    - Study metadata in every chunk                                 │    │
│  │    - Image/table references linked to chunks                       │    │
│  │    - Citation-ready format with PMID                               │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   AGENTIC CHAT WITH MEDICAL GROUNDING                       │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  SSE Streaming Pipeline:                                            │    │
│  │    1. Safety check → Block patient-specific queries                │    │
│  │    2. Tool calling: search_documents()                             │    │
│  │    3. Retrieve hybrid results (KG + vector + rerank)               │    │
│  │    4. Evidence-first generation:                                   │    │
│  │       - Every claim must cite a source[pmid]                       │    │
│  │       - Include evidence level in citations                        │    │
│  │       - Conservative language ("studies suggest", "evidence shows")│    │
│  │    5. Stream tokens with inline citations                          │    │
│  │    6. Display images/tables alongside answer                       │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Multi-Specialist Consultation (Optional):                                  │
│    - Query routed to multiple subspecialty workspaces                      │
│    - Parallel retrieval from each specialist's knowledge base              │
│    - Coordinator synthesizes perspectives                                  │
│    - Highlights consensus vs. conflicting evidence                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

```yaml
Core Framework:
  - Base: NexusRAG codebase (Python, FastAPI, React)
  - Backend: FastAPI + SQLAlchemy 2.0 + asyncio
  - Frontend: React 19 + TypeScript 5.9 + Vite

Document Processing:
  - Docling: PDF/DOCX parsing with structure preservation
  - PubMed E-utilities: Paper acquisition
  - PyMuPDF: Fallback for simple PDFs
  - scispaCy: Medical NER (en_core_sci_md model)

Embeddings & Retrieval:
  - PubMedBERT: Medical domain embeddings (768d)
  - ChromaDB: Vector store (workspace-isolated)
  - LightRAG: Knowledge graph (entity/relationship extraction)
  - bge-reranker-v2-m3: Cross-encoder reranking

LLM:
  - Primary: Claude 3.5 Sonnet (best medical accuracy)
  - Local: Ollama + Llama 3.2 8B (for cost savings)
  - Vision: Gemini 2.0 Flash (for image/table captioning)

Infrastructure:
  - PostgreSQL 15: Metadata, documents, chat history
  - Redis: Session storage, conversation memory
  - Docker Compose: Local development
  - nginx: API gateway (production)

Medical Domain:
  - NCBI Biopython: PubMed API integration
  - MeSH terms: Medical subject headings
  - Evidence grading: Custom classifier
  - Citation normalization: PMID-based
```

### 1.3 Database Schema (Enhanced from NexusRAG)

```sql
-- Workspaces = Subspecialties
CREATE TABLE knowledge_bases (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),  -- "Breast Surgery", "Craniofacial", etc.
    description TEXT,
    system_prompt TEXT,
    subspecialty VARCHAR(100),  -- New: subspecialty identifier
    created_at TIMESTAMP DEFAULT NOW()
);

-- Documents = Research Papers
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    workspace_id INTEGER REFERENCES knowledge_bases(id),
    original_filename VARCHAR(255),
    file_type VARCHAR(50),
    file_size BIGINT,

    -- NexusRAG fields
    status VARCHAR(50),  -- parsing, indexing, indexed, failed
    markdown_content TEXT,
    page_count INTEGER,
    chunk_count INTEGER,
    table_count INTEGER,
    image_count INTEGER,

    -- Medical fields (NEW)
    pmid VARCHAR(50) UNIQUE,  -- PubMed ID
    doi VARCHAR(255),
    title TEXT,
    abstract TEXT,
    authors TEXT[],  -- Array of author names
    journal VARCHAR(255),
    publication_year INTEGER,
    publication_date DATE,
    mesh_terms TEXT[],  -- Medical subject headings
    study_design VARCHAR(100),  -- RCT, cohort, case_series, etc.
    evidence_level VARCHAR(10),  -- I, II, III, IV, V
    sample_size INTEGER,  -- n=

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Images (from NexusRAG)
CREATE TABLE document_images (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    image_id VARCHAR(255) UNIQUE,
    page_no INTEGER,
    file_path TEXT,
    caption TEXT,  -- LLM-generated or extracted
    width INTEGER,
    height INTEGER,
    mime_type VARCHAR(50),

    -- Medical enhancement (NEW)
    image_type VARCHAR(50),  -- diagram, chart, photo, before_after, etc.

    created_at TIMESTAMP DEFAULT NOW()
);

-- Tables (from NexusRAG)
CREATE TABLE document_tables (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    table_id VARCHAR(255) UNIQUE,
    page_no INTEGER,
    content_markdown TEXT,
    caption TEXT,
    num_rows INTEGER,
    num_cols INTEGER,

    -- Medical enhancement (NEW)
    table_type VARCHAR(50),  -- demographics, outcomes, complications, etc.

    created_at TIMESTAMP DEFAULT NOW()
);

-- Chat messages (from NexusRAG, enhanced)
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    workspace_id INTEGER REFERENCES knowledge_bases(id),
    message_id VARCHAR(255) UNIQUE,
    role VARCHAR(50),  -- user, assistant
    content TEXT,
    sources JSONB,  -- Array of source chunks with PMIDs
    image_refs JSONB,
    thinking TEXT,
    agent_steps JSONB,

    -- Medical enhancement (NEW)
    safety_classification VARCHAR(50),  -- literature, patient_specific, emergency
    related_entities TEXT[],  -- Extracted medical entities
    evidence_quality JSONB,  -- Summary of evidence levels used

    created_at TIMESTAMP DEFAULT NOW()
);

-- Medical entity cache (NEW)
CREATE TABLE medical_entities (
    id SERIAL PRIMARY KEY,
    workspace_id INTEGER REFERENCES knowledge_bases(id),
    entity_text VARCHAR(255),
    entity_type VARCHAR(50),  -- condition, procedure, treatment, outcome, etc.
    umls_cui VARCHAR(50),  -- UMLS concept unique identifier (optional)
    frequency INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(workspace_id, entity_text, entity_type)
);
```

### 1.4 Key Implementation Changes

#### 1.4.1 Medical Document Parser (Extends Docling)

```python
# src/services/medical_document_parser.py

from app.services.deep_document_parser import DeepDocumentParser
from Bio import Entrez, Medline
import re

class MedicalDocumentParser(DeepDocumentParser):
    """
    Extends NexusRAG's DeepDocumentParser with medical domain enhancements.
    """

    def parse(self, file_path, document_id, pmid=None):
        """
        Parse medical research paper with enhanced metadata extraction.
        """
        # Use parent Docling parsing
        parsed = super().parse(file_path, document_id, file_path)

        # Enhance with medical metadata
        if pmid:
            medical_meta = self._fetch_pubmed_metadata(pmid)
            parsed.medical_metadata = medical_meta
        else:
            # Extract from text
            parsed.medical_metadata = self._extract_medical_metadata(parsed.markdown)

        # Classify study design
        parsed.study_design = self._classify_study_design(parsed.markdown)

        # Grade evidence level
        parsed.evidence_level = self._grade_evidence(parsed.study_design, parsed.medical_metadata)

        # Extract medical entities
        parsed.medical_entities = self._extract_medical_entities(parsed.markdown)

        # Enhance chunks with medical context
        parsed.chunks = self._enrich_chunks_with_medical_context(
            parsed.chunks,
            parsed.medical_metadata
        )

        return parsed

    def _fetch_pubmed_metadata(self, pmid):
        """Fetch metadata from PubMed API."""
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = Medline.read(handle)
        return {
            'pmid': pmid,
            'title': record.get('TI', ''),
            'abstract': record.get('AB', ''),
            'authors': record.get('AU', []),
            'journal': record.get('JT', ''),
            'year': record.get('DP', '')[:4],
            'doi': record.get('AID', [''])[0],
            'mesh_terms': record.get('MH', []),
        }

    def _classify_study_design(self, text):
        """Classify study design from text."""
        text_lower = text.lower()

        if any(term in text_lower for term in ['randomized controlled trial', 'rct', 'randomized']):
            return 'RCT'
        elif 'meta-analysis' in text_lower or 'systematic review' in text_lower:
            return 'meta_analysis'
        elif 'prospective' in text_lower and 'cohort' in text_lower:
            return 'prospective_cohort'
        elif 'retrospective' in text_lower:
            return 'retrospective_cohort'
        elif 'case-control' in text_lower:
            return 'case_control'
        elif 'case series' in text_lower or 'case report' in text_lower:
            return 'case_series'
        else:
            return 'unknown'

    def _grade_evidence(self, study_design, metadata):
        """Grade evidence level (Oxford Centre for Evidence-Based Medicine)."""
        if study_design == 'meta_analysis':
            return 'I'
        elif study_design == 'RCT':
            return 'I'
        elif study_design == 'prospective_cohort':
            return 'II'
        elif study_design in ['retrospective_cohort', 'case_control']:
            return 'III'
        elif study_design == 'case_series':
            return 'IV'
        else:
            return 'V'

    def _extract_medical_entities(self, text):
        """Extract medical entities using scispaCy."""
        import spacy
        nlp = spacy.load("en_core_sci_md")
        doc = nlp(text[:100000])  # Limit for performance

        entities = []
        for ent in doc.ents:
            if ent.label_ in ['DISEASE', 'CHEMICAL', 'PROCEDURE']:
                entities.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                })
        return entities

    def _enrich_chunks_with_medical_context(self, chunks, medical_metadata):
        """Add medical context header to each chunk."""
        for chunk in chunks:
            # Prepend medical context
            context_header = f"""
Study: {medical_metadata.get('title', 'Unknown')}
PMID: {medical_metadata.get('pmid', 'N/A')}
Study Design: {self.study_design} (Evidence Level: {self.evidence_level})
Sample Size: n={medical_metadata.get('sample_size', 'N/A')}
Authors: {', '.join(medical_metadata.get('authors', [])[:3])}
Journal: {medical_metadata.get('journal', 'Unknown')} ({medical_metadata.get('year', 'N/A')})

Section: {' > '.join(chunk.heading_path) if chunk.heading_path else 'Body'}

Content:
"""
            chunk.content = context_header + chunk.content

        return chunks
```

#### 1.4.2 Medical Safety Classifier

```python
# src/services/medical_safety_classifier.py

import re
from typing import Literal

SafetyClass = Literal['literature', 'patient_specific', 'emergency']

class MedicalSafetyClassifier:
    """
    Classifies queries as literature-based, patient-specific, or emergency.
    Patient-specific queries are downgraded to evidence-only mode.
    """

    PATIENT_SPECIFIC_PATTERNS = [
        r'\bmy patient\b',
        r'\bthis patient\b',
        r'\bshould I (perform|do|give|prescribe)\b',
        r'\bwhat (dose|dosage|medication)\b',
        r'\bhow much\b',
        r'\b\d+[\s-]?year[\s-]?old (male|female|patient|man|woman)\b',
        r'\bBMI\s+\d+\b',
        r'\b(he|she) has\b',
    ]

    EMERGENCY_PATTERNS = [
        r'\bemergency\b',
        r'\bimmediate(ly)?\b',
        r'\burgent(ly)?\b',
        r'\bsevere bleeding\b',
        r'\brespiratory distress\b',
    ]

    def classify(self, query: str) -> SafetyClass:
        """Classify query into safety category."""
        query_lower = query.lower()

        # Check emergency first
        for pattern in self.EMERGENCY_PATTERNS:
            if re.search(pattern, query_lower):
                return 'emergency'

        # Check patient-specific
        for pattern in self.PATIENT_SPECIFIC_PATTERNS:
            if re.search(pattern, query_lower):
                return 'patient_specific'

        # Default to literature
        return 'literature'

    def get_warning_message(self, safety_class: SafetyClass) -> str:
        """Get appropriate warning message."""
        if safety_class == 'emergency':
            return """
⚠️ **EMERGENCY SITUATION DETECTED**

This system is for educational purposes only and should NOT be used for emergency medical decisions.

If this is a medical emergency:
- Call emergency services immediately
- Consult with an attending physician
- Follow your institution's emergency protocols
"""
        elif safety_class == 'patient_specific':
            return """
⚠️ **PATIENT-SPECIFIC QUERY DETECTED**

This query appears to ask about a specific patient case. I can only provide:
- General evidence summaries from research literature
- Published clinical guidelines and protocols
- NOT patient-specific medical advice

Clinical decisions should be made by qualified healthcare professionals considering individual patient circumstances.
"""
        else:
            return ""
```

#### 1.4.3 Multi-Specialist Router (Optional Enhancement)

```python
# src/services/multi_specialist_router.py

from typing import List
import asyncio

class MultiSpecialistRouter:
    """
    Routes queries to multiple subspecialty workspaces and synthesizes results.
    """

    SUBSPECIALTY_KEYWORDS = {
        'craniofacial': ['cleft', 'craniosynostosis', 'facial', 'craniofacial', 'palate'],
        'hand': ['hand', 'finger', 'wrist', 'tendon', 'carpal', 'thumb'],
        'breast': ['breast', 'mammary', 'DIEP', 'TRAM', 'reconstruction', 'mastectomy'],
        'burn': ['burn', 'thermal', 'scar', 'contracture', 'skin graft'],
        'reconstructive': ['flap', 'reconstruction', 'defect', 'microvascular', 'free tissue'],
        'aesthetic': ['rhinoplasty', 'facelift', 'blepharoplasty', 'aesthetic', 'cosmetic'],
    }

    def route_query(self, query: str) -> List[str]:
        """Determine which subspecialties should handle this query."""
        query_lower = query.lower()

        matched = []
        for subspecialty, keywords in self.SUBSPECIALTY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                matched.append(subspecialty)

        # Default to reconstructive if no match
        if not matched:
            matched = ['reconstructive']

        return matched

    async def multi_specialist_query(
        self,
        query: str,
        workspaces: List[int],
        db
    ):
        """
        Query multiple subspecialty workspaces in parallel.
        Returns synthesized result.
        """
        from app.services.nexus_rag_service import NexusRAGService

        # Query each workspace in parallel
        tasks = []
        for workspace_id in workspaces:
            rag_service = NexusRAGService(db, workspace_id)
            task = rag_service.query_deep(query, top_k=5, mode='hybrid')
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Synthesize results
        all_chunks = []
        all_citations = []

        for result in results:
            all_chunks.extend(result.chunks)
            all_citations.extend(result.citations)

        # Deduplicate and rerank across workspaces
        # ... (reranking logic)

        return {
            'chunks': all_chunks,
            'citations': all_citations,
            'subspecialties_consulted': len(workspaces),
        }
```

---

## Part 2: Implementation Plan

### Phase 1: Foundation (Week 1)

**Goal:** Set up environment and adapt NexusRAG for medical domain

#### Day 1-2: Environment Setup
```bash
# Clone and set up NexusRAG
cd ~/GitHub
git clone https://github.com/LeDat98/NexusRAG.git medical-nexus-rag
cd medical-nexus-rag

# Create Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Additional medical dependencies
pip install \
    biopython==1.81 \
    scispacy==0.5.3 \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz

# Start infrastructure
docker-compose -f docker-compose.services.yml up -d
```

#### Day 3-4: Medical Document Parser
- Create `MedicalDocumentParser` extending `DeepDocumentParser`
- Implement PubMed metadata fetching
- Add study design classification
- Add evidence level grading
- Test with 10 sample papers

#### Day 5-7: Database Schema & Models
- Extend `Document` model with medical fields (PMID, authors, evidence_level, etc.)
- Create `MedicalEntity` model
- Enhance `ChatMessage` with safety classification
- Run migrations
- Seed database with test data

**Deliverable:** Working medical document parser that processes PDFs with enhanced metadata

---

### Phase 2: Hybrid Retrieval (Week 2)

**Goal:** Enhance retrieval with medical domain specificity

#### Day 8-9: Medical Embeddings
- Switch from BGE-M3 to PubMedBERT embeddings
- Test embedding quality on medical queries
- Benchmark retrieval accuracy

#### Day 10-11: Medical NER Integration
- Integrate scispaCy for medical entity extraction
- Build entity cache in database
- Enhance Knowledge Graph with medical ontology

#### Day 12-13: Safety Classifier
- Implement `MedicalSafetyClassifier`
- Add to query pipeline
- Test with patient-specific queries
- Add warning UI components

#### Day 14: Integration Testing
- End-to-end test: PubMed paper → Process → Query → Response
- Verify citations include PMID
- Verify evidence levels displayed

**Deliverable:** Working retrieval pipeline with medical safety checks

---

### Phase 3: Agentic Chat & UI (Week 3)

**Goal:** Adapt agentic chat for medical citations and evidence grading

#### Day 15-16: Medical Chat Enhancements
- Modify `chat_agent.py` to include evidence levels in citations
- Add PMID display in citations
- Modify system prompt for medical accuracy
- Test tool calling with medical queries

#### Day 17-18: UI Enhancements
- Add evidence level badges to citation display
- Add PMID links to PubMed
- Display safety warnings
- Show study design metadata

#### Day 19-20: Multi-Specialist Routing (Optional)
- Implement `MultiSpecialistRouter`
- Create subspecialty workspaces
- Test parallel consultation

#### Day 21: Demo Preparation
- Create demo script
- Record demo video
- Prepare presentation

**Deliverable:** Full POC with working UI

---

### Phase 4: Testing & Refinement (Week 4)

**Goal:** Polish and validate medical accuracy

#### Day 22-23: Medical Accuracy Testing
- Create test set of 50 medical queries
- Verify all citations have PMIDs
- Check evidence level correctness
- Test safety classifier recall

#### Day 24-25: Performance Optimization
- Benchmark query latency
- Optimize embedding cache
- Tune reranker thresholds

#### Day 26-27: Documentation
- API documentation
- Deployment guide
- Medical accuracy safeguards document

#### Day 28: Final Demo
- Stakeholder presentation
- Collect feedback
- Plan next iteration

**Deliverable:** Production-ready POC with documentation

---

## Part 3: Data Acquisition Strategy

### 3.1 PubMed Paper Collection

```python
# scripts/collect_pubmed_papers.py

from Bio import Entrez
import os

Entrez.email = "your-email@example.com"
Entrez.api_key = os.getenv("PUBMED_API_KEY")

SUBSPECIALTY_QUERIES = {
    'breast_surgery': """
        (DIEP flap OR TRAM flap OR "breast reconstruction"[MeSH])
        AND (outcomes OR complications OR "surgical technique")
        AND ("plastic surgery"[MeSH] OR "reconstructive surgical procedures"[MeSH])
        AND ("2015"[Date - Publication] : "2026"[Date - Publication])
    """,
    'craniofacial': """
        ("cleft lip"[MeSH] OR "cleft palate"[MeSH] OR craniosynostosis[MeSH])
        AND ("surgical outcomes" OR "treatment outcomes")
        AND ("2015"[Date - Publication] : "2026"[Date - Publication])
    """,
    # ... more subspecialties
}

def fetch_papers_for_subspecialty(subspecialty, query, max_papers=100):
    """Fetch papers for a subspecialty."""
    print(f"\nFetching {max_papers} papers for {subspecialty}...")

    # Search
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_papers,
        sort="relevance",
        usehistory="y"
    )
    record = Entrez.read(handle)
    pmids = record['IdList']

    print(f"Found {len(pmids)} papers")

    # Download PDFs via PMC or other sources
    # (requires institutional access or PMC Open Access Subset)

    return pmids
```

### 3.2 Initial Dataset Recommendations

**For POC (Week 1):**
- **Breast Surgery:** 50 papers on DIEP flap
- **Source:** PMC Open Access Subset (free full-text)
- **Focus:** Recent high-quality papers (2020-2026)

**For Full System (Week 4):**
- **10 subspecialties × 200 papers each = 2,000 papers**
- **Source:** Institutional access or PMC OA
- **Coverage:** Meta-analyses, RCTs, cohort studies (prioritize Level I-II evidence)

---

## Part 4: Evaluation Metrics

### 4.1 Retrieval Quality

```python
# Metrics:
- Precision@5: Are retrieved chunks relevant?
- Recall: Did we find all relevant papers?
- PMID accuracy: 100% of citations must have correct PMID
- Evidence level accuracy: Correct classification rate
```

### 4.2 Medical Safety

```python
# Metrics:
- Patient-specific detection recall: >95%
- Emergency detection recall: >98%
- False positive rate: <5%
```

### 4.3 Citation Accuracy

```python
# Metrics:
- Citation coverage: % of claims with citations
- Citation correctness: Manual verification of citation-claim match
- PMID link validity: All PMIDs resolve to correct papers
```

---

## Part 5: Cost Analysis

### POC (4 weeks)

```
Development Time:
- 1 developer × 4 weeks = 4 person-weeks

Infrastructure (Local):
- MacBook Pro M4: $0 (already owned)
- Docker containers: $0
- PubMedBERT: $0 (local)
- Ollama: $0 (local)

Data:
- PMC Open Access papers: $0
- PubMed API: $0

Optional Cloud:
- Claude API (testing): $20-50
- Total: ~$50
```

### Production (Monthly)

```
Compute:
- GPU server (embedding + LLM): $500-1,000/mo
  OR Cloud API (Claude): $2,000-5,000/mo

Storage:
- PostgreSQL: $50-100/mo
- ChromaDB storage: $50-200/mo

Total: $600-6,000/mo (depending on local vs cloud LLM)
```

---

## Part 6: Risk Mitigation

### Medical Accuracy Risks

**Risk:** Hallucinated medical information
**Mitigation:**
- Evidence-first generation (every claim requires citation)
- Low temperature (0.1) for generation
- Cross-encoder reranking filters low-relevance chunks
- PMID tracing for verification

**Risk:** Outdated medical knowledge
**Mitigation:**
- Filter papers by date (last 5-10 years)
- Display publication year prominently
- Update corpus quarterly

### Safety Risks

**Risk:** System used for patient-specific advice
**Mitigation:**
- Mandatory safety classifier at query time
- Strong UI warnings for patient-specific queries
- Downgrade to evidence-only mode (no recommendations)
- Clear disclaimers: "Educational purposes only"

### Technical Risks

**Risk:** Poor retrieval quality
**Mitigation:**
- Hybrid retrieval (KG + vector + reranking) reduces failure modes
- PubMedBERT provides domain-specific embeddings
- Manual evaluation with test queries

---

## Part 7: Success Criteria

### POC Success (Week 4)

✅ **Functional Requirements:**
- Process 50 PubMed papers with full metadata
- Answer 20 test medical queries with citations
- All citations include PMID and evidence level
- Safety classifier blocks patient-specific queries
- Response time < 5 seconds per query

✅ **Quality Requirements:**
- Citation accuracy: >90% (manual verification)
- Safety classifier recall: >95%
- Evidence level classification: >85% correct

✅ **Demo Requirements:**
- Clean UI showing citations, evidence levels, PMIDs
- Live demo answering questions about breast reconstruction
- Comparison showing improvement over basic RAG

---

## Part 8: Next Steps After POC

### Phase 5: Multi-Specialist System (Weeks 5-8)

1. **Create 10 Subspecialty Workspaces**
   - Craniofacial, Hand, Breast, Burn, etc.
   - 200 papers each

2. **Implement Multi-Agent Coordination**
   - Parallel querying
   - Consensus detection
   - Coordinator synthesis

3. **Add Conversation Memory**
   - Redis-based session management
   - Context-aware follow-ups
   - Paper caching

### Phase 6: Advanced Features (Weeks 9-12)

1. **Structured Data Extraction**
   - Outcome tables parsing
   - Complication rates extraction
   - Demographics extraction

2. **Advanced Citations**
   - Sentence-level provenance
   - Citation quality scoring
   - Conflicting evidence detection

3. **Production Deployment**
   - Kubernetes deployment
   - Monitoring & logging
   - User authentication

---

## Part 9: Comparison: NexusRAG vs Medical-RAG-System Design

| Feature | NexusRAG | Medical-RAG Design | Recommended POC |
|---------|----------|-------------------|-----------------|
| **Document Parsing** | ✅ Docling (best-in-class) | ⚠️ GROBID (complex) | Use Docling |
| **Chunking** | ✅ Structure-aware | ✅ Structure-aware | Use NexusRAG's |
| **Embeddings** | BGE-M3 (general) | PubMedBERT (medical) | Use PubMedBERT |
| **Vector Store** | ✅ ChromaDB | ✅ Qdrant | Either works |
| **Knowledge Graph** | ✅ LightRAG | ❌ Not planned | Use LightRAG |
| **Reranking** | ✅ Cross-encoder | ❌ Not planned | Use reranker |
| **Citations** | ✅ Inline 4-char IDs | ✅ PMID-based | Combine both |
| **Multi-Agent** | ❌ Single workspace | ✅ Multi-subspecialty | Add to NexusRAG |
| **Safety Checks** | ❌ None | ✅ Patient-specific | Add to NexusRAG |
| **Evidence Grading** | ❌ None | ✅ Level I-V | Add to NexusRAG |
| **Agentic Chat** | ✅ SSE streaming | ⚠️ Not specified | Use NexusRAG's |
| **Images/Tables** | ✅ Extracted & captioned | ⚠️ Basic | Use NexusRAG's |

**Conclusion:** NexusRAG provides superior document processing and retrieval architecture. Medical-RAG design provides better domain-specific requirements. **Combining both** gives the best POC.

---

## Part 10: Implementation Checklist

### Week 1: Foundation
- [ ] Clone NexusRAG
- [ ] Install dependencies + medical libraries (scispaCy, biopython)
- [ ] Extend database schema with medical fields
- [ ] Create `MedicalDocumentParser`
- [ ] Test with 10 sample papers
- [ ] Verify PMID extraction

### Week 2: Retrieval
- [ ] Switch to PubMedBERT embeddings
- [ ] Integrate scispaCy NER
- [ ] Create `MedicalSafetyClassifier`
- [ ] Test hybrid retrieval with medical queries
- [ ] Verify evidence level classification

### Week 3: Chat & UI
- [ ] Modify agentic chat for medical citations
- [ ] Add evidence level badges to UI
- [ ] Add PMID links
- [ ] Display safety warnings
- [ ] Test end-to-end flow

### Week 4: Polish & Demo
- [ ] Create test set (50 queries)
- [ ] Measure citation accuracy
- [ ] Optimize performance
- [ ] Record demo video
- [ ] Write documentation

---

## Conclusion

This POC combines NexusRAG's proven architecture with medical domain requirements to create a production-ready medical RAG system in 4 weeks.

**Key Innovations:**
1. **Docling-powered parsing** preserves medical paper structure
2. **Hybrid retrieval** (KG + vector + reranking) improves accuracy
3. **Medical safety checks** prevent misuse
4. **Evidence grading** provides transparency
5. **PMID-based citations** enable verification

**Recommended Path:**
1. Start with NexusRAG codebase (proven architecture)
2. Add medical domain layers (parser, safety, evidence grading)
3. Deploy single-subspecialty POC (Breast Surgery)
4. Expand to multi-subspecialty system if successful

**Timeline:** 4 weeks for full POC, 8 weeks for multi-specialist system

**Cost:** <$100 for POC (local development), $600-6,000/mo for production (depending on LLM strategy)
