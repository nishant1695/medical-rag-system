# POC Implementation Plan - Medical RAG System
## Proof of Concept for Demonstration

**Target:** Working demo in 2-3 weeks using free/low-cost tools
**Platform:** MacBook Pro M4 Pro with 24GB RAM (excellent for local LLMs!)
**Scope:** Single subspecialty agent with core v4.0 features

---

## Table of Contents

1. [POC Scope and Goals](#1-poc-scope-and-goals)
2. [Technology Stack for POC](#2-technology-stack-for-poc)
3. [Local vs API Recommendation](#3-local-vs-api-recommendation)
4. [Architecture for POC](#4-architecture-for-poc)
5. [Week-by-Week Implementation Plan](#5-week-by-week-implementation-plan)
6. [Detailed Implementation Steps](#6-detailed-implementation-steps)
7. [Demo Script](#7-demo-script)
8. [Cost Analysis](#8-cost-analysis)

---

## 1. POC Scope and Goals

### What to Include

✅ **Core v4.0 Features:**
- Single subspecialty (Breast Surgery) - easier to validate
- Shared corpus architecture (even with one agent, prove the concept)
- Evidence-first generation with citations
- Clinical safety classification
- Structure-aware chunking with study context
- Query rewrite with confidence display
- Basic conversation memory

✅ **Demo-Ready Features:**
- Streamlit UI showing query interpretation
- Visual display of retrieved papers with evidence levels
- Sentence-level citation highlighting
- Safety boundary demonstration
- Follow-up question handling

❌ **Defer to Full Implementation:**
- Multiple agents (just show architecture)
- Full PubMed integration (use sample dataset)
- Structured table extraction (show concept with 1-2 tables)
- Production monitoring and logging
- Authentication/authorization

### Success Criteria

**Functional:**
- Can answer 10 test questions about breast surgery
- Shows evidence-first generation (no hallucination)
- Displays query rewriting with confidence
- Handles follow-up questions with context
- Demonstrates safety boundaries

**Technical:**
- Response time < 10 seconds on M4 Pro
- Runs entirely on laptop (portable demo)
- Uses < 20GB RAM
- Can process 50 sample papers

**Presentation:**
- Clean, professional Streamlit UI
- Clear visualization of system components
- Can explain v4.0 architectural decisions
- Side-by-side comparison showing improvements

---

## 2. Technology Stack for POC

### Recommended Stack (Optimized for M4 Pro + Free Tools)

```yaml
Frontend:
  - Streamlit (free, rapid prototyping)
  - Plotly for visualizations

Vector Store:
  - Qdrant (local Docker, free)
  - Alternative: Pinecone free tier (100K vectors, 1 index)
  - Recommendation: Qdrant local for demo portability

Embeddings:
  - Local: sentence-transformers/all-MiniLM-L6-v2 (fast baseline)
  - Medical: dmis-lab/biobert-base-cased-v1.2 (better quality)
  - Recommendation: Start with MiniLM, upgrade to BioBERT if needed

LLM:
  - Local: Llama 3.2 3B via Ollama (fast, fits in 24GB)
  - API fallback: Claude/GPT-4 for comparison
  - Recommendation: Hybrid - local for speed, API for quality comparison

Document Processing:
  - PyMuPDF (PDF extraction)
  - spaCy (NER, sentence splitting)
  - Skip GROBID for POC (complex setup)

Conversation Memory:
  - In-memory dictionary (POC only)
  - Could use SQLite for persistence

Data:
  - 50 manually curated papers on breast reconstruction
  - Or PubMed API for demo (no storage)
```

### Why This Stack?

**M4 Pro Advantages:**
- 24GB unified memory → can run LLM + embeddings simultaneously
- Neural Engine → accelerates local LLMs
- Fast enough for real-time demo

**Cost:**
```
Qdrant local: $0
Streamlit: $0
Ollama: $0
BioBERT embeddings: $0 (local)
Sample data: $0

Optional API calls for comparison:
- Claude API: ~$5 for testing
- Total POC cost: < $10
```

---

## 3. Local vs API Recommendation

### For Embeddings: **LOCAL** ✅

**Recommendation: Use local BioBERT**

```python
# Install
pip install sentence-transformers torch

# Load model (will download once, ~500MB)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')

# Embed (fast on M4 Pro)
embeddings = model.encode(texts, show_progress_bar=True)
# Speed: ~500 chunks/minute on M4 Pro
```

**Why Local:**
- No API costs
- No rate limits
- Portable demo (works offline)
- Fast enough on M4 Pro (~0.1s per chunk)
- Medical domain model available

**When to Use API:**
- If you need OpenAI text-embedding-3 for comparison
- For production with high volume

### For LLM: **HYBRID** ✅

**Recommendation: Ollama Llama 3.2 3B local + Claude API for comparison**

#### Option A: Ollama (Local - Recommended for POC)

```bash
# Install Ollama
brew install ollama

# Pull model (one-time, ~2GB)
ollama pull llama3.2:3b

# Use in Python
import ollama

response = ollama.chat(model='llama3.2:3b', messages=[
    {'role': 'system', 'content': 'You are a medical research assistant.'},
    {'role': 'user', 'content': 'What are DIEP flap complications?'}
])
```

**Performance on M4 Pro:**
- Load time: ~2 seconds
- Generation: ~50 tokens/second
- Memory: ~4GB
- Cost: $0

**Pros:**
- Free, unlimited
- Runs offline
- Fast enough for demo
- Privacy (no data leaves laptop)

**Cons:**
- Lower quality than GPT-4/Claude
- May hallucinate more
- Less medical knowledge

#### Option B: Claude API (API - For Quality Comparison)

```python
from anthropic import Anthropic

client = Anthropic(api_key="...")
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[{"role": "user", "content": "..."}]
)
```

**Cost:**
- Input: $3 / 1M tokens
- Output: $15 / 1M tokens
- POC usage: ~$5 total

**Pros:**
- Best quality
- Best citation following
- Low hallucination

**Cons:**
- Costs money
- Needs internet
- Rate limits

### **RECOMMENDED APPROACH for POC:**

```python
class HybridLLM:
    """
    Use local for speed, API for quality comparison
    """
    def __init__(self, use_api=False):
        self.use_api = use_api

        if use_api:
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            import ollama
            self.client = ollama

    def generate(self, prompt):
        if self.use_api:
            # Use Claude for best quality
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            # Use Ollama for speed/offline
            response = self.client.chat(
                model='llama3.2:3b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
```

**Demo Strategy:**
- Default to Ollama (fast, free, portable)
- Have toggle in UI: "Use Claude API for higher quality"
- Show comparison of both outputs

---

## 4. Architecture for POC

### Simplified Architecture

```
┌─────────────────────────────────────────────────┐
│          STREAMLIT UI                           │
│  ┌──────────────┐  ┌──────────────┐            │
│  │  Query Input │  │  Settings    │            │
│  │              │  │  - Local/API │            │
│  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│      SAFETY CLASSIFIER                          │
│      - Patient-specific detection               │
│      - Display boundary warnings                │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│      CONVERSATION MANAGER                       │
│      - Session state (Streamlit)                │
│      - Query rewriter with confidence           │
│      - Show interpretation                      │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│      QDRANT (Local Docker)                      │
│      - 50 papers × 50 chunks = 2,500 vectors    │
│      - Breast surgery focus                     │
│      - Study metadata in every chunk            │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│      RETRIEVER                                  │
│      - Hybrid search (semantic + keyword)       │
│      - Filter by evidence level (optional)      │
│      - Return top 10 chunks                     │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│      EVIDENCE-FIRST GENERATOR                   │
│      - Extract claims from chunks               │
│      - Map to query                             │
│      - Generate with constraints                │
│      - Attach chunk_ids                         │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│      RESPONSE FORMATTER                         │
│      - Highlight citations                      │
│      - Show evidence quality                    │
│      - Display query interpretation             │
│      - Show retrieved papers                    │
└─────────────────────────────────────────────────┘
```

### Directory Structure

```
medical-rag-poc/
├── app.py                      # Streamlit app
├── requirements.txt
├── .env.example
│
├── config/
│   └── settings.yaml           # Configuration
│
├── data/
│   ├── papers/                 # 50 sample papers (PDFs)
│   └── processed/              # Processed chunks
│
├── src/
│   ├── safety.py               # Safety classifier
│   ├── chunker.py              # Structure-aware chunking
│   ├── embedder.py             # Local BioBERT embedding
│   ├── retriever.py            # Qdrant retrieval
│   ├── generator.py            # Evidence-first generation
│   ├── conversation.py         # Memory management
│   └── utils.py                # Helper functions
│
├── scripts/
│   ├── 01_download_papers.py  # PubMed fetching
│   ├── 02_process_papers.py   # Extract and chunk
│   ├── 03_create_embeddings.py # Generate embeddings
│   └── 04_index_qdrant.py     # Index in Qdrant
│
├── tests/
│   └── test_queries.json       # 10 test questions
│
└── notebooks/
    └── exploration.ipynb       # Development notebook
```

---

## 5. Week-by-Week Implementation Plan

### Week 1: Data & Infrastructure

**Days 1-2: Environment Setup**
```bash
# Install dependencies
pip install streamlit sentence-transformers qdrant-client \
    pymupdf spacy anthropic ollama

# Install Ollama
brew install ollama
ollama pull llama3.2:3b

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Download spaCy model
python -m spacy download en_core_web_sm
```

**Days 3-4: Data Collection**
```python
# Fetch 50 sample papers on breast reconstruction
# Focus areas:
# - DIEP flap (20 papers)
# - TRAM flap (10 papers)
# - Implant-based (10 papers)
# - Complications (10 papers)

# Use PubMed API
python scripts/01_download_papers.py \
    --keywords "DIEP flap,breast reconstruction" \
    --max-papers 50
```

**Days 5-7: Processing Pipeline**
```python
# Process papers
# - Extract text with PyMuPDF
# - Structure-aware chunking
# - Extract study metadata
# - Generate embeddings
# - Index in Qdrant

python scripts/02_process_papers.py
python scripts/03_create_embeddings.py
python scripts/04_index_qdrant.py
```

**Week 1 Goal:** 50 papers indexed in Qdrant with study context

---

### Week 2: Core Features

**Days 8-9: Safety Classifier**
```python
# Implement clinical safety classification
# Test with patient-specific queries
# Demo boundary detection

# File: src/safety.py
class SafetyClassifier:
    def classify(self, query):
        # Rule-based + simple NER
        # Return: literature/patient_specific/emergency
        pass
```

**Days 10-11: Evidence-First Generation**
```python
# Implement claim extraction and mapping
# Constrained generation prompts
# Sentence-level provenance

# File: src/generator.py
class EvidenceFirstGenerator:
    def generate(self, query, chunks):
        # Extract claims
        # Map to query
        # Generate with constraints
        # Validate provenance
        pass
```

**Days 12-13: Conversation Memory**
```python
# Query rewriting with confidence
# Session management in Streamlit
# Context display

# File: src/conversation.py
class ConversationManager:
    def rewrite_query(self, query, history):
        # Rewrite with confidence
        # Display interpretation
        pass
```

**Day 14: Integration Testing**
```python
# Test end-to-end flow
# Verify no hallucination on test set
# Check citation accuracy
```

**Week 2 Goal:** Core v4.0 features working

---

### Week 3: UI & Polish

**Days 15-16: Streamlit UI**
```python
# File: app.py
import streamlit as st

# Pages:
# 1. Main query interface
# 2. Settings (local vs API)
# 3. About/Architecture
# 4. Test queries
```

**Days 17-18: Visualization**
```python
# Add visualizations:
# - Retrieved papers with evidence levels
# - Citation highlighting
# - Query interpretation display
# - Confidence scores
# - Study context cards
```

**Days 19-20: Demo Polish**
```python
# Prepare demo script
# Create comparison examples
# Add example queries
# Test on fresh audience
```

**Day 21: Documentation & Demo Recording**
```
# Create demo video
# Write usage guide
# Prepare presentation slides
```

**Week 3 Goal:** Polished, demo-ready POC

---

## 6. Detailed Implementation Steps

### Step 1: Setup Script

```bash
#!/bin/bash
# setup_poc.sh

echo "Setting up Medical RAG POC..."

# Create directories
mkdir -p medical-rag-poc/{data/papers,data/processed,src,scripts,tests}
cd medical-rag-poc

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
cat > requirements.txt << 'EOF'
streamlit==1.28.0
sentence-transformers==2.2.2
qdrant-client==1.7.0
pymupdf==1.23.0
spacy==3.7.0
anthropic==0.18.0
python-dotenv==1.0.0
pandas==2.1.0
plotly==5.17.0
biopython==1.81
torch==2.1.0
EOF

pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Install Ollama
brew install ollama
ollama pull llama3.2:3b

# Start Qdrant
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

echo "Setup complete! Run: streamlit run app.py"
```

### Step 2: Data Fetching Script

```python
# scripts/01_download_papers.py

from Bio import Entrez, Medline
import os
from dotenv import load_dotenv

load_dotenv()

Entrez.email = os.getenv("PUBMED_EMAIL", "your_email@example.com")
Entrez.api_key = os.getenv("PUBMED_API_KEY")

def fetch_papers(keywords, max_papers=50):
    """Fetch papers from PubMed"""

    # Build query
    query = f'({keywords}) AND ("breast reconstruction"[MeSH] OR "plastic surgery"[MeSH])'

    print(f"Searching PubMed: {query}")

    # Search
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_papers,
        sort="relevance"
    )
    record = Entrez.read(handle)
    pmids = record['IdList']

    print(f"Found {len(pmids)} papers")

    # Fetch details
    papers = []
    for pmid in pmids:
        handle = Entrez.efetch(
            db="pubmed",
            id=pmid,
            rettype="medline",
            retmode="text"
        )
        record = Medline.read(handle)

        paper = {
            'pmid': pmid,
            'title': record.get('TI', ''),
            'abstract': record.get('AB', ''),
            'authors': record.get('AU', []),
            'journal': record.get('JT', ''),
            'year': record.get('DP', '')[:4],
            'mesh_terms': record.get('MH', [])
        }
        papers.append(paper)

        # Save
        with open(f'data/papers/{pmid}.json', 'w') as f:
            json.dump(paper, f, indent=2)

    print(f"Saved {len(papers)} papers to data/papers/")
    return papers

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keywords', default='DIEP flap,breast reconstruction')
    parser.add_argument('--max-papers', type=int, default=50)
    args = parser.parse_args()

    fetch_papers(args.keywords, args.max_papers)
```

### Step 3: Processing Script

```python
# scripts/02_process_papers.py

import json
import os
from pathlib import Path
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_study_metadata(text):
    """Extract study structure from text"""

    # Simple rule-based extraction for POC
    metadata = {
        "design": "unknown",
        "n": None,
        "population": None,
        "limitations": []
    }

    # Detect study design
    text_lower = text.lower()
    if "randomized controlled trial" in text_lower or "rct" in text_lower:
        metadata["design"] = "RCT"
    elif "prospective" in text_lower:
        metadata["design"] = "prospective_cohort"
    elif "retrospective" in text_lower:
        metadata["design"] = "retrospective_cohort"

    # Extract sample size
    import re
    n_match = re.search(r'n\s*=\s*(\d+)', text_lower)
    if n_match:
        metadata["n"] = int(n_match.group(1))

    # Extract limitations (simple)
    if "limitations" in text_lower:
        limitations_section = text_lower.split("limitations")[1][:500]
        if "retrospective" in limitations_section:
            metadata["limitations"].append("Retrospective design")
        if "single-center" in limitations_section or "single center" in limitations_section:
            metadata["limitations"].append("Single-center study")

    return metadata

def chunk_paper(paper, study_metadata):
    """Structure-aware chunking"""

    text = paper['abstract']  # Use abstract for POC
    doc = nlp(text)

    chunks = []
    sentences = list(doc.sents)

    # Chunk every 3-5 sentences
    chunk_size = 4
    for i in range(0, len(sentences), chunk_size):
        chunk_sentences = sentences[i:i+chunk_size]
        chunk_text = ' '.join([s.text for s in chunk_sentences])

        # Add context header
        contextualized = f"""
Paper: {paper['title']}
Study Design: {study_metadata['design']} (n={study_metadata['n']})
Section: Abstract

Content:
{chunk_text}

Limitations: {', '.join(study_metadata['limitations'])}
"""

        chunks.append({
            'content': contextualized,
            'content_raw': chunk_text,
            'paper_pmid': paper['pmid'],
            'metadata': {
                'title': paper['title'],
                'year': paper['year'],
                'journal': paper['journal'],
                'study_metadata': study_metadata
            }
        })

    return chunks

def process_all_papers():
    """Process all papers"""

    papers_dir = Path('data/papers')
    all_chunks = []

    for paper_file in papers_dir.glob('*.json'):
        with open(paper_file) as f:
            paper = json.load(f)

        # Extract study metadata
        full_text = paper['title'] + ' ' + paper.get('abstract', '')
        study_metadata = extract_study_metadata(full_text)

        # Chunk paper
        chunks = chunk_paper(paper, study_metadata)
        all_chunks.extend(chunks)

        print(f"Processed {paper['pmid']}: {len(chunks)} chunks")

    # Save processed chunks
    with open('data/processed/chunks.json', 'w') as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Total chunks: {len(all_chunks)}")

if __name__ == '__main__':
    process_all_papers()
```

### Step 4: Embedding & Indexing Script

```python
# scripts/03_create_embeddings.py

import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

# Load embedding model (local)
print("Loading BioBERT model...")
model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')
print("Model loaded!")

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

# Create collection
collection_name = "breast_surgery_papers"
print(f"Creating collection: {collection_name}")

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# Load chunks
with open('data/processed/chunks.json') as f:
    chunks = json.load(f)

print(f"Embedding {len(chunks)} chunks...")

# Embed and index
points = []
for i, chunk in enumerate(chunks):
    # Generate embedding
    embedding = model.encode(chunk['content'])

    # Create point
    point = PointStruct(
        id=i,
        vector=embedding.tolist(),
        payload={
            'content': chunk['content'],
            'content_raw': chunk['content_raw'],
            'pmid': chunk['paper_pmid'],
            'title': chunk['metadata']['title'],
            'year': chunk['metadata']['year'],
            'journal': chunk['metadata']['journal'],
            'study_design': chunk['metadata']['study_metadata']['design'],
            'sample_size': chunk['metadata']['study_metadata']['n'],
            'limitations': chunk['metadata']['study_metadata']['limitations']
        }
    )
    points.append(point)

    # Batch upload
    if len(points) >= 100:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Indexed {i+1}/{len(chunks)} chunks")
        points = []

# Upload remaining
if points:
    client.upsert(collection_name=collection_name, points=points)

print("Indexing complete!")
```

### Step 5: Streamlit App

```python
# app.py

import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import re

# Page config
st.set_page_config(
    page_title="Medical RAG POC",
    page_icon="🏥",
    layout="wide"
)

# Initialize
@st.cache_resource
def init_system():
    client = QdrantClient(url="http://localhost:6333")
    model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')
    return client, model

client, embedder = init_system()

# Sidebar
st.sidebar.title("Settings")
use_api = st.sidebar.checkbox("Use Claude API (better quality)", value=False)
show_internals = st.sidebar.checkbox("Show system internals", value=True)

# Title
st.title("🏥 Medical RAG System POC")
st.markdown("**Demonstrating v4.0 Production-Ready Architecture**")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Safety Classifier
def classify_safety(query):
    patient_patterns = [
        r'my patient', r'this patient', r'should I',
        r'what dose', r'how much'
    ]

    for pattern in patient_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return "patient_specific"

    return "literature"

# Query Rewriter
def rewrite_query(query, history):
    if not history:
        return query, 1.0, False

    # Simple rewriting for POC
    # In production, use LLM

    # Check if query needs context
    if len(query.split()) < 5 and any(word in query.lower() for word in ['what', 'about', 'how']):
        # Need rewrite
        last_query = history[-1]['query']

        # Extract main topic from last query
        rewritten = f"{query} {last_query}"
        confidence = 0.8
        used_rewrite = True
    else:
        rewritten = query
        confidence = 1.0
        used_rewrite = False

    return rewritten, confidence, used_rewrite

# Retriever
def retrieve(query, top_k=5):
    embedding = embedder.encode(query).tolist()

    results = client.search(
        collection_name="breast_surgery_papers",
        query_vector=embedding,
        limit=top_k
    )

    return results

# Generator
def generate_answer(query, chunks):
    # Build evidence-constrained prompt
    evidence = "\n\n".join([
        f"[{i+1}] {chunk.payload['content_raw']}\n(PMID: {chunk.payload['pmid']}, Study: {chunk.payload['study_design']})"
        for i, chunk in enumerate(chunks)
    ])

    prompt = f"""You are a medical research assistant. Answer using ONLY the evidence below.

CRITICAL RULES:
1. Use ONLY information from the evidence
2. Cite every statement with [N]
3. If evidence doesn't fully answer, say so

EVIDENCE:
{evidence}

QUERY: {query}

ANSWER (with citations):"""

    if use_api:
        # Use Claude (requires API key)
        from anthropic import Anthropic
        client_api = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        response = client_api.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    else:
        # Use Ollama (local)
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']

# Main interface
query = st.text_input("Ask a medical question:", placeholder="What are the complications of DIEP flap surgery?")

if st.button("Submit") and query:
    # Add to history
    st.session_state.messages.append({'query': query, 'type': 'user'})

    with st.spinner("Processing..."):
        # Safety check
        safety = classify_safety(query)

        if safety == "patient_specific":
            st.warning("⚠️ Patient-Specific Query Detected")
            st.info("This query appears to ask about a specific patient. I can only provide general evidence summaries, not patient-specific recommendations.")
            st.stop()

        # Query rewriting
        history = st.session_state.messages
        rewritten_query, confidence, used_rewrite = rewrite_query(query, history)

        if used_rewrite and show_internals:
            st.info(f"**Query Interpretation:**\nOriginal: {query}\nInterpreted as: {rewritten_query}\nConfidence: {confidence:.0%}")

        # Retrieve
        results = retrieve(rewritten_query, top_k=5)

        if show_internals:
            with st.expander("📄 Retrieved Papers"):
                for i, result in enumerate(results):
                    st.markdown(f"**Paper {i+1}** (Score: {result.score:.2f})")
                    st.markdown(f"*{result.payload['title']}*")
                    st.markdown(f"Study: {result.payload['study_design']}, n={result.payload['sample_size']}")
                    st.markdown(f"PMID: {result.payload['pmid']}")
                    st.markdown("---")

        # Generate
        answer = generate_answer(query, results)

        # Display
        st.markdown("### Answer")
        st.markdown(answer)

        # Show evidence quality
        st.markdown("### Evidence Quality")
        designs = [r.payload['study_design'] for r in results]
        st.markdown(f"- Papers: {len(results)}")
        st.markdown(f"- Study types: {', '.join(set(designs))}")

# Show conversation history
if st.session_state.messages:
    st.sidebar.markdown("### Conversation History")
    for msg in st.session_state.messages:
        st.sidebar.markdown(f"- {msg['query'][:50]}...")
```

---

## 7. Demo Script

### Demo Flow (10 minutes)

**1. Introduction (1 min)**
```
"This is a POC of our v4.0 production-ready medical RAG system.
Key improvements over typical RAG:
- Evidence-first generation (no hallucination)
- Clinical safety boundaries
- Query interpretation with confidence
- Structure-aware chunking"
```

**2. Basic Query (2 min)**
```
Query: "What are the outcomes of DIEP flap breast reconstruction?"

Show:
- Retrieved papers with evidence levels
- Generated answer with sentence-level citations
- Evidence quality summary
```

**3. Follow-up Question (2 min)**
```
Query: "What about in obese patients?"

Show:
- Query rewriting: "What are the outcomes of DIEP flap breast reconstruction in obese patients?"
- Confidence score: 89%
- Context from previous turn
- Reused cached papers
```

**4. Safety Boundary (2 min)**
```
Query: "Should I use DIEP flap for my patient with BMI 35?"

Show:
- Safety classification: patient_specific
- Warning message
- Downgrade to evidence-only mode
- Strong disclaimer
```

**5. Comparison: Local vs API (2 min)**
```
Toggle "Use Claude API" checkbox

Show:
- Side-by-side comparison
- Local: Faster, free, but lower quality
- API: Slower, costs $0.02, but better quality
- Both have citations
```

**6. Architecture Explanation (1 min)**
```
Show diagram:
- Shared corpus (not separate stores)
- Evidence-first generation
- Clinical safety layer
- Structure-aware chunks with study context
```

---

## 8. Cost Analysis

### POC Development Costs

```
One-Time Setup:
- Ollama: $0 (free)
- BioBERT model: $0 (free, local)
- Qdrant: $0 (local Docker)
- Sample papers: $0 (PubMed free access)
Total: $0

Optional API Testing:
- Claude API (50 test queries): ~$5
- Total with API: ~$5
```

### POC Running Costs

```
Per Query (Local):
- Embedding: $0 (local)
- LLM: $0 (Ollama local)
- Vector search: $0 (local Qdrant)
Total: $0

Per Query (API):
- Embedding: $0 (local)
- LLM: $0.02 (Claude)
- Vector search: $0
Total: $0.02

Demo (100 queries):
- All local: $0
- With API: $2
```

### Comparison to v3 (Hypothetical)

```
v3 Architecture (Separate Stores):
- 10 agents × 5K papers × embeddings = 5× cost
- Post-validation → more LLM calls
- No caching

v4 Architecture (Shared Corpus):
- 1× embedding cost
- Evidence-first → fewer LLM calls
- Smart caching

Savings: ~60% cost reduction
```

---

## Summary

### Recommended Approach

**Embeddings:** Local BioBERT
- Fast enough on M4 Pro (~0.1s/chunk)
- Medical domain model
- $0 cost

**LLM:** Hybrid
- Default: Ollama Llama 3.2 3B (fast, free, portable)
- Optional: Claude API toggle (better quality, $0.02/query)
- Show comparison in demo

**Vector Store:** Qdrant Local
- Runs in Docker
- Portable demo
- $0 cost

**Total POC Cost:** < $10 (mostly API testing)

### Timeline

- **Week 1:** Data + Infrastructure (50 papers indexed)
- **Week 2:** Core features (safety, evidence-first, conversation)
- **Week 3:** UI polish and demo prep

### Deliverables

1. Working Streamlit demo
2. 50 papers on breast reconstruction
3. Test set of 10 queries
4. Demo video/slides
5. Documentation

**Result:** Production-ready POC demonstrating v4.0 architectural improvements with minimal cost.

