# Comprehensive Multi-Agent Medical RAG System with Conversational Memory
## Complete Design Document for Plastic Surgery Research Assistant

**Version:** 3.0
**Date:** March 12, 2026
**Architecture:** Multi-Agent Specialist System with Conversational Memory

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Data Acquisition Layer](#3-data-acquisition-layer)
4. [Document Processing Layer](#4-document-processing-layer)
5. [Chunking Layer](#5-chunking-layer)
6. [Vector Store Layer](#6-vector-store-layer)
7. [Specialist Agent System](#7-specialist-agent-system)
8. [Query Classification and Routing](#8-query-classification-and-routing)
9. [Retrieval Layer](#9-retrieval-layer)
10. [Generation Layer](#10-generation-layer)
11. [Conversation Memory System](#11-conversation-memory-system)
12. [Multi-Agent Consultation](#12-multi-agent-consultation)
13. [API Layer](#13-api-layer)
14. [Evaluation and Quality Assurance](#14-evaluation-and-quality-assurance)
15. [Implementation Roadmap](#15-implementation-roadmap)
16. [Technology Stack](#16-technology-stack)
17. [Configuration Management](#17-configuration-management)
18. [Monitoring and Analytics](#18-monitoring-and-analytics)
19. [Cost Estimation](#19-cost-estimation)
20. [Appendices](#20-appendices)

---

## 1. Executive Summary

This document presents a comprehensive **multi-agent RAG (Retrieval-Augmented Generation) system** with **conversational memory** designed specifically for plastic surgery research. The system enables medical professionals to query peer-reviewed research papers across multiple subspecialties through natural, contextual conversations.

### 1.1 Core Capabilities

**Multi-Agent Architecture:**
- Each plastic surgery subspecialty has a dedicated specialist agent
- Each agent maintains its own knowledge base of research papers
- Intelligent routing directs queries to appropriate specialists
- Multiple agents collaborate on cross-subspecialty questions
- Coordinator agent synthesizes perspectives from multiple specialists

**Conversational Memory:**
- Natural follow-up questions without repeating context
- Context-aware query rewriting ("What about in children?" → fully contextualized)
- Paper caching for faster follow-up responses
- Session-based conversation tracking
- Progressive topic exploration

**Medical Accuracy Focus:**
- Mandatory citation of all claims with PMID and author
- Evidence quality assessment for all responses
- Conservative generation to minimize hallucination
- Verifiability through direct paper references
- Transparent indication of confidence levels

### 1.2 Key Features

✅ **Specialist Agents**: Craniofacial, Hand Surgery, Aesthetic, Reconstructive, Burn, Breast, Microsurgery, Pediatric, etc.
✅ **User-Defined Keywords**: You provide PubMed search terms for each subspecialty
✅ **Parallel Consultation**: Multiple specialists answer simultaneously
✅ **Consensus Building**: Coordinator identifies agreements and disagreements
✅ **Conversational Flow**: Natural multi-turn dialogues with context preservation
✅ **Paper Caching**: Reuse retrieved papers across conversation turns
✅ **Evidence Grading**: Clear indication of evidence strength and quality

### 1.3 Example User Experience

```
User: Start conversation
System: Session started [ID: abc-123]

Turn 1:
User: "What are the outcomes of DIEP flap breast reconstruction?"
System: [Breast + Reconstructive Agents consulted]
        Retrieved 15 papers, answers with citations
        Processing time: 3.2s

Turn 2:
User: "What about in obese patients?"
System: Detected follow-up question
        Rewritten: "What are the outcomes of DIEP flap breast reconstruction in obese patients?"
        [Same agents consulted]
        Reused 5 papers from cache, retrieved 10 new papers
        Processing time: 1.8s (faster due to cache!)

Turn 3:
User: "Compare that to implant-based reconstruction"
System: Detected comparison mode
        Rewritten: "Compare DIEP flap to implant-based breast reconstruction outcomes in obese patients"
        [Breast + Reconstructive Agents]
        Provides side-by-side comparison
        Processing time: 2.1s

Turn 4:
User: "Which has fewer complications?"
System: Continues comparison context
        Provides complication rate comparison with statistical evidence
        All claims cited with PMIDs
```

---

## 2. System Architecture Overview

### 2.1 Complete System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE LAYER                                │
│  - Web UI / API Client                                                      │
│  - Session Management                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CONVERSATION MANAGER                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Session    │  │   Context    │  │    Query     │  │    Paper     │  │
│  │    Store     │  │   Builder    │  │   Rewriter   │  │    Cache     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUERY CLASSIFICATION & ROUTING                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │   Medical    │→ │ Subspecialty │→ │   Routing    │                     │
│  │     NER      │  │  Classifier  │  │   Decision   │                     │
│  └──────────────┘  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SPECIALIST AGENT LAYER                                │
│                                                                              │
│  ┌──────────────────────────┐  ┌──────────────────────────┐               │
│  │  Craniofacial Agent      │  │  Hand Surgery Agent      │               │
│  │  - Knowledge Base (KB)   │  │  - Knowledge Base (KB)   │               │
│  │  - Vector Store          │  │  - Vector Store          │               │
│  │  - Retriever             │  │  - Retriever             │               │
│  │  - LLM                   │  │  - LLM                   │               │
│  └──────────────────────────┘  └──────────────────────────┘               │
│                                                                              │
│  ┌──────────────────────────┐  ┌──────────────────────────┐               │
│  │  Aesthetic Agent         │  │  Reconstructive Agent    │               │
│  │  - Knowledge Base (KB)   │  │  - Knowledge Base (KB)   │               │
│  │  - Vector Store          │  │  - Vector Store          │               │
│  │  - Retriever             │  │  - Retriever             │               │
│  │  - LLM                   │  │  - LLM                   │               │
│  └──────────────────────────┘  └──────────────────────────┘               │
│                                                                              │
│  ┌──────────────────────────┐  ┌──────────────────────────┐               │
│  │  Burn Agent              │  │  [Additional Agents]     │               │
│  │  - Knowledge Base (KB)   │  │  - Breast Surgery        │               │
│  │  - Vector Store          │  │  - Microsurgery          │               │
│  │  - Retriever             │  │  - Pediatric             │               │
│  │  - LLM                   │  │  - Maxillofacial         │               │
│  └──────────────────────────┘  └──────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       COORDINATOR AGENT                                      │
│  - Response Synthesis                                                        │
│  - Consensus Detection                                                       │
│  - Citation Aggregation                                                      │
│  - Evidence Quality Assessment                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESPONSE LAYER                                      │
│  - Unified Answer with Citations                                            │
│  - Individual Specialist Perspectives                                        │
│  - Evidence Summary                                                          │
│  - Conversation State Update                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Layers

The system operates through distinct layers:

1. **Data Acquisition**: Fetches papers from PubMed using user-provided keywords
2. **Processing**: Extracts text, metadata, and medical entities from papers
3. **Chunking**: Breaks papers into semantic chunks with context preservation
4. **Embedding**: Generates vector embeddings using medical-domain models
5. **Storage**: Stores chunks in subspecialty-specific vector databases
6. **Conversation**: Manages session state and context across turns
7. **Routing**: Determines which specialist agent(s) should answer
8. **Retrieval**: Fetches relevant chunks (with caching for follow-ups)
9. **Generation**: Each agent generates domain-specific answer
10. **Synthesis**: Coordinator combines multiple perspectives
11. **Memory Update**: Stores turn in session for future context

---

## 3. Data Acquisition Layer

### 3.1 PubMed Integration Strategy

**Primary API:** NCBI E-utilities (Entrez Programming Utilities)
- **Rate Limits:** 10 requests/second with API key
- **Coverage:** Complete PubMed database (35M+ citations)
- **Full Text:** PubMed Central Open Access subset (~3M full-text articles)

### 3.2 Agent-Specific Knowledge Base Construction

Each specialist agent gets papers based on **user-provided keywords**:

```python
class AgentKnowledgeBaseBuilder:
    """
    Builds knowledge base for each specialist agent using user-provided keywords
    """
    def __init__(self, pubmed_api_key):
        self.pubmed = PubMedAPI(api_key=pubmed_api_key)

    def build_agent_kb(self, agent_config, max_papers=5000):
        """
        Fetch papers for a specific agent based on their keywords

        Args:
            agent_config: Dict with 'agent_id', 'pubmed_keywords', 'mesh_terms'
            max_papers: Maximum number of papers to fetch
        """
        agent_id = agent_config['agent_id']
        keywords = agent_config['pubmed_keywords']  # User-provided
        mesh_terms = agent_config.get('mesh_terms', [])  # Optional

        # Construct PubMed query
        query = self._build_pubmed_query(keywords, mesh_terms)

        print(f"Building KB for {agent_config['name']}")
        print(f"PubMed Query: {query}")

        # Fetch papers with filters
        papers = self.pubmed.search_and_fetch(
            query=query,
            max_results=max_papers,
            filters={
                'publication_types': [
                    'Journal Article',
                    'Review',
                    'Clinical Trial',
                    'Randomized Controlled Trial',
                    'Meta-Analysis'
                ],
                'languages': ['eng'],
                'date_range': '2000/01/01:2026/12/31'  # Configurable
            }
        )

        print(f"Fetched {len(papers)} papers for {agent_config['name']}")

        # Save papers to agent-specific directory
        self._save_papers(agent_id, papers)

        return papers

    def _build_pubmed_query(self, keywords, mesh_terms):
        """
        Build PubMed query from keywords and MeSH terms
        """
        # Combine keywords with OR
        keyword_query = ' OR '.join([f'"{kw}"' for kw in keywords])

        # Add MeSH terms if provided
        if mesh_terms:
            mesh_query = ' OR '.join(mesh_terms)
            full_query = f"({keyword_query}) OR ({mesh_query})"
        else:
            full_query = keyword_query

        # Add plastic surgery context
        full_query = f"({full_query}) AND (plastic surgery[MeSH] OR reconstructive surgery[MeSH])"

        return full_query

    def _save_papers(self, agent_id, papers):
        """
        Save papers to agent-specific directory
        """
        output_dir = f"data/agents/{agent_id}/raw_papers"
        os.makedirs(output_dir, exist_ok=True)

        for paper in papers:
            # Save PDF if available
            if paper.pdf_url:
                pdf_path = f"{output_dir}/{paper.pmid}.pdf"
                download_pdf(paper.pdf_url, pdf_path)

            # Save metadata
            metadata_path = f"{output_dir}/{paper.pmid}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(paper.to_dict(), f, indent=2)
```

### 3.3 Metadata Extraction

**Critical metadata to capture:**

```python
paper_metadata = {
    "pmid": "12345678",
    "pmcid": "PMC7654321",
    "doi": "10.1097/PRS.0000000000001234",
    "title": "Outcomes of DIEP Flap Breast Reconstruction...",
    "abstract": "Background: Deep inferior epigastric...",
    "authors": ["Smith JA", "Jones BC", "Williams CD"],
    "journal": "Plastic and Reconstructive Surgery",
    "publication_date": "2023-06-15",
    "impact_factor": 4.5,
    "subspecialty": ["breast", "reconstructive"],
    "mesh_terms": ["Mammaplasty", "Perforator Flap", "Breast Neoplasms"],
    "keywords": ["DIEP flap", "breast reconstruction", "outcomes"],
    "citation_count": 45,
    "study_type": "Retrospective Cohort Study",
    "evidence_level": "III",
    "patient_count": 250,
    "full_text_available": true
}
```

### 3.4 Quality Filters

- **Publication Types:** Prioritize systematic reviews, RCTs, clinical trials
- **Exclude:** Editorials, letters, comments (unless high-impact)
- **Time Range:** Configurable (default: 2000-present)
- **Impact Factor:** Optional filtering by journal quality

---

## 4. Document Processing Layer

### 4.1 Text Extraction

**Recommended Tool: GROBID** (Open Source Scientific PDF Parser)
- Specifically designed for scientific papers
- Extracts structured sections (Abstract, Methods, Results, Discussion)
- Handles tables, figures, and citations
- Fallback to PyMuPDF for simple PDFs

```python
class DocumentProcessor:
    """
    Extract and process text from medical papers
    """
    def __init__(self):
        self.grobid = GrobidClient(url="http://localhost:8070")
        self.fallback = PyMuPDFExtractor()

    def process_paper(self, pdf_path, metadata):
        """
        Extract structured content from paper
        """
        try:
            # Primary: GROBID
            structured_data = self.grobid.process_fulltext_document(pdf_path)
        except Exception as e:
            # Fallback: PyMuPDF
            structured_data = self.fallback.extract(pdf_path)

        return {
            "title": structured_data.title,
            "abstract": structured_data.abstract,
            "sections": [
                {"heading": "Introduction", "content": "..."},
                {"heading": "Methods", "content": "..."},
                {"heading": "Results", "content": "..."},
                {"heading": "Discussion", "content": "..."}
            ],
            "tables": self._extract_tables(structured_data),
            "figures": self._extract_figure_captions(structured_data),
            "references": self._extract_citations(structured_data)
        }
```

### 4.2 Medical Text Preprocessing

**Critical for Medical Domain:**

#### 4.2.1 Abbreviation Expansion

Medical texts are full of abbreviations that need context:

```python
PLASTIC_SURGERY_ABBREVIATIONS = {
    "DIEP": "deep inferior epigastric perforator",
    "TRAM": "transverse rectus abdominis myocutaneous",
    "ALT": "anterolateral thigh",
    "SIEA": "superficial inferior epigastric artery",
    "TMJ": "temporomandibular joint",
    "LeFort": "LeFort osteotomy classification",
    # ... hundreds more
}

def expand_abbreviations(text, abbreviation_dict):
    """
    Expand medical abbreviations while preserving original
    """
    expanded_text = text
    for abbrev, expansion in abbreviation_dict.items():
        # Keep both: "DIEP (deep inferior epigastric perforator) flap"
        pattern = rf'\b{abbrev}\b'
        replacement = f"{abbrev} ({expansion})"
        expanded_text = re.sub(pattern, replacement, expanded_text)

    return expanded_text
```

#### 4.2.2 Medical Entity Recognition

Extract and tag medical entities:

```python
import scispacy
import spacy

nlp = spacy.load("en_core_sci_md")  # ScispaCy medical model

def extract_medical_entities(text):
    """
    Extract medical entities from text
    """
    doc = nlp(text)

    entities = {
        "procedures": [],
        "anatomical": [],
        "materials": [],
        "outcomes": [],
        "conditions": []
    }

    for ent in doc.ents:
        if ent.label_ == "PROCEDURE":
            entities["procedures"].append(ent.text)
        elif ent.label_ == "ANATOMY":
            entities["anatomical"].append(ent.text)
        # ... more entity types

    return entities
```

#### 4.2.3 Section Identification

Different sections have different information value:

```python
def identify_sections(structured_paper):
    """
    Tag sections with their type and importance
    """
    section_types = {
        "abstract": {"importance": "high", "use_for": ["overview", "outcomes"]},
        "introduction": {"importance": "medium", "use_for": ["background", "rationale"]},
        "methods": {"importance": "high", "use_for": ["technique", "procedure"]},
        "results": {"importance": "high", "use_for": ["outcomes", "statistics"]},
        "discussion": {"importance": "high", "use_for": ["interpretation", "clinical_implications"]},
        "conclusion": {"importance": "medium", "use_for": ["summary", "recommendations"]}
    }

    for section in structured_paper['sections']:
        section_name = section['heading'].lower()
        if section_name in section_types:
            section['metadata'] = section_types[section_name]

    return structured_paper
```

---

## 5. Chunking Layer

### 5.1 Hierarchical Semantic Chunking Strategy

**Why Hierarchical?**
- Medical papers have natural hierarchy: Paper → Section → Paragraph
- Context flows across levels
- Need both specific facts and broader context

```python
class MedicalPaperChunker:
    """
    Hierarchical semantic chunking for medical papers
    """
    def __init__(self):
        self.min_chunk_size = 200  # tokens
        self.max_chunk_size = 800  # tokens
        self.overlap = 100  # tokens
        self.embedder = SentenceTransformer('pubmedbert')

    def chunk_paper(self, paper):
        """
        Create hierarchical chunks from paper
        """
        chunks = []

        # Level 1: Abstract (always separate)
        if paper.abstract:
            chunks.append(self.create_chunk(
                content=paper.abstract,
                level="abstract",
                metadata=paper.metadata,
                section_name="Abstract"
            ))

        # Level 2: Section-based chunks
        for section in paper.sections:
            section_chunks = self.chunk_section(
                section,
                paper_title=paper.title,
                paper_metadata=paper.metadata
            )
            chunks.extend(section_chunks)

        # Level 3: Tables (special handling)
        for table in paper.tables:
            chunks.append(self.chunk_table(table, paper.metadata))

        return chunks

    def chunk_section(self, section, paper_title, paper_metadata):
        """
        Semantic chunking within a section
        """
        sentences = sent_tokenize(section.content)

        # Calculate embeddings for semantic similarity
        embeddings = self.embedder.encode(sentences)

        # Find semantic break points (where similarity drops)
        break_points = self.find_semantic_breaks(embeddings)

        chunks = []
        for start, end in break_points:
            chunk_text = " ".join(sentences[start:end])

            # Add contextual headers
            contextualized_text = self.add_context_header(
                text=chunk_text,
                paper_title=paper_title,
                section_heading=section.heading
            )

            chunks.append(self.create_chunk(
                content=contextualized_text,
                content_raw=chunk_text,
                level="section",
                section_name=section.heading,
                metadata=paper_metadata
            ))

        return chunks

    def add_context_header(self, text, paper_title, section_heading):
        """
        Prepend contextual information to chunk
        """
        header = f"""
Paper: {paper_title}
Section: {section_heading}

Content:
"""
        return header + text

    def find_semantic_breaks(self, embeddings):
        """
        Find natural break points based on semantic similarity
        """
        break_points = []
        current_start = 0

        for i in range(1, len(embeddings)):
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]

            # If similarity drops below threshold, it's a break point
            if similarity < 0.7:  # Threshold
                break_points.append((current_start, i))
                current_start = i

        # Add final chunk
        if current_start < len(embeddings):
            break_points.append((current_start, len(embeddings)))

        return break_points
```

### 5.2 Chunk Metadata Schema

```python
chunk_metadata = {
    "chunk_id": "uuid-1234",
    "paper_pmid": "12345678",
    "chunk_type": "section",  # abstract|section|table|figure
    "section_name": "Results",
    "hierarchical_position": "3.1",

    # Content
    "content": "Paper: Title\nSection: Results\n\nContent: ...",  # With headers
    "content_raw": "...",  # Without headers
    "token_count": 450,

    # Paper metadata (inherited)
    "paper_metadata": {
        "title": "...",
        "authors": ["..."],
        "journal": "...",
        "publication_date": "2023-06-15",
        "pmid": "12345678",
        "doi": "...",
        "subspecialty": ["breast", "reconstructive"],
        "evidence_level": "III",
        "citation_count": 45
    },

    # Chunk-specific enrichment
    "extracted_entities": {
        "procedures": ["DIEP flap", "microsurgical anastomosis"],
        "anatomical_terms": ["deep inferior epigastric artery"],
        "outcomes": ["flap survival", "donor site morbidity"]
    },
    "contains_statistics": true,
    "statistical_claims": ["95% survival rate (p<0.001)"],

    # For retrieval
    "embedding": [0.123, 0.456, ...]  # 768-dim vector
}
```

### 5.3 Special Handling for Tables

Tables need special treatment:

```python
def chunk_table(self, table, paper_metadata):
    """
    Convert table to retrievable format
    """
    # Create natural language description
    nl_description = f"""
Table: {table.caption}
Columns: {', '.join(table.columns)}

Key findings:
{self.generate_table_summary(table)}

Full table data:
{self.format_table_as_text(table)}
"""

    return {
        "content": nl_description,
        "content_raw": nl_description,
        "chunk_type": "table",
        "structured_data": {
            "headers": table.columns,
            "rows": table.rows,
            "caption": table.caption
        },
        "paper_metadata": paper_metadata
    }
```

---

## 6. Vector Store Layer

### 6.1 Embedding Model Selection

**Recommended: PubMedBERT** (Microsoft)
- Model: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Embedding Dimension: 768
- Training Data: PubMed abstracts + full-text papers
- **Best for medical terminology and clinical context**

**Alternative: Dual Embedding Approach**
```python
class DualEmbeddingSystem:
    def __init__(self):
        # Primary: Medical domain model
        self.medical_embedder = PubMedBERTEmbedder()

        # Secondary: High-performance general model (optional)
        self.general_embedder = OpenAIEmbedder(model="text-embedding-3-large")

    def embed_chunk(self, chunk):
        return {
            "medical_embedding": self.medical_embedder.embed(chunk.content),
            "general_embedding": self.general_embedder.embed(chunk.content)  # Optional
        }
```

### 6.2 Vector Database: Qdrant (Recommended)

**Why Qdrant?**
- Excellent metadata filtering (critical for medical domain)
- Supports multiple vectors per point (dual embeddings)
- Production-ready, mature
- Easy Docker deployment
- Fast similarity search

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class AgentVectorStoreManager:
    """
    Manages separate vector stores for each specialist agent
    """
    def __init__(self, qdrant_url="localhost:6333"):
        self.client = QdrantClient(url=qdrant_url)
        self.embedder = PubMedBERTEmbedder()

    def create_agent_collection(self, agent_id):
        """
        Create dedicated collection for an agent
        """
        collection_name = f"{agent_id}_papers"

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "medical": VectorParams(
                    size=768,  # PubMedBERT
                    distance=Distance.COSINE
                )
            }
        )

        # Create payload indexes for filtering
        self._create_indexes(collection_name)

        print(f"Created collection: {collection_name}")

    def index_agent_papers(self, agent_id, chunks):
        """
        Index all chunks for an agent's knowledge base
        """
        collection_name = f"{agent_id}_papers"

        points = []
        for chunk in chunks:
            # Generate embedding
            embedding = self.embedder.embed(chunk.content)

            point = PointStruct(
                id=chunk.chunk_id,
                vector={"medical": embedding},
                payload={
                    "agent_id": agent_id,
                    "pmid": chunk.pmid,
                    "content": chunk.content,
                    "content_raw": chunk.content_raw,
                    "section_name": chunk.section_name,
                    "chunk_type": chunk.chunk_type,
                    "paper_title": chunk.paper_metadata.title,
                    "authors": chunk.paper_metadata.authors,
                    "journal": chunk.paper_metadata.journal,
                    "publication_date": chunk.paper_metadata.publication_date,
                    "evidence_level": chunk.paper_metadata.evidence_level,
                    "citation_count": chunk.paper_metadata.citation_count,
                    "extracted_entities": chunk.extracted_entities,
                    "citation_text": f"{chunk.paper_metadata.authors[0]} et al. ({chunk.paper_metadata.publication_date[:4]}). {chunk.paper_metadata.title}. PMID: {chunk.pmid}"
                }
            )
            points.append(point)

            # Batch upload every 100 points
            if len(points) >= 100:
                self.client.upsert(collection_name=collection_name, points=points)
                points = []

        # Upload remaining
        if points:
            self.client.upsert(collection_name=collection_name, points=points)

        print(f"Indexed {len(chunks)} chunks for agent {agent_id}")

    def _create_indexes(self, collection_name):
        """
        Create payload indexes for fast filtering
        """
        indexes = [
            "evidence_level",
            "publication_date",
            "section_name",
            "chunk_type",
            "pmid"
        ]

        for field in indexes:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema="keyword"
            )
```

---

## 7. Specialist Agent System

### 7.1 Agent Architecture

Each subspecialty has a dedicated specialist agent:

```python
class SpecialistAgent:
    """
    Each agent operates independently with its own vector store
    """
    def __init__(self, config):
        self.agent_id = config['agent_id']
        self.name = config['name']
        self.pubmed_keywords = config['pubmed_keywords']
        self.mesh_terms = config.get('mesh_terms', [])

        # Each agent has its own vector database collection
        self.vector_store = QdrantClient()
        self.collection_name = f"{self.agent_id}_papers"

        # Agent-specific embedder
        self.embedder = PubMedBERTEmbedder()

        # Agent-specific retriever
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            collection_name=self.collection_name
        )

        # Agent-specific LLM with specialized system prompt
        self.llm = self._initialize_llm()

        # Knowledge base metadata
        self.paper_count = 0
        self.last_updated = None

    def _initialize_llm(self):
        """
        Initialize LLM with specialist system prompt
        """
        system_prompt = f"""
You are a medical research specialist in {self.name}.

Your knowledge base consists of peer-reviewed research papers specifically
about {', '.join(self.pubmed_keywords[:5])}.

CRITICAL RULES:
1. Only answer questions within your specialty domain
2. If a question is outside your expertise, clearly state this
3. Cite every claim with PMID and author
4. Use conservative, precise medical language
5. Indicate evidence quality and certainty level
6. For multi-specialty questions, focus only on your domain

Your responses should reflect deep expertise in your subspecialty.
"""
        return AnthropicLLM(
            model="claude-3-5-sonnet-20241022",
            system_prompt=system_prompt
        )

    def retrieve(self, query, top_k=10):
        """
        Retrieve relevant papers from this agent's knowledge base
        """
        # Enhance query with specialty-specific terms
        enhanced_query = self._enhance_query(query)

        # Retrieve from this agent's vector store
        results = self.retriever.retrieve(
            query=enhanced_query,
            top_k=top_k
        )

        return results

    def answer(self, query, retrieved_docs):
        """
        Generate answer based on retrieved documents
        """
        # Assess if query is within this agent's domain
        relevance = self._assess_domain_relevance(query)

        if relevance < 0.3:
            return {
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "answer": f"This question appears to be outside the scope of {self.name}. I recommend consulting a different specialist.",
                "confidence": 0.0,
                "citations": []
            }

        # Generate answer
        response = self._generate_response(query, retrieved_docs)

        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "answer": response['answer'],
            "confidence": response['confidence'],
            "citations": response['citations'],
            "evidence_quality": response['evidence_quality'],
            "retrieved_papers_count": len(retrieved_docs)
        }
```

### 7.2 Agent Registry Configuration

Users configure agents with their keywords:

```yaml
# config/agents.yaml

agents:
  - agent_id: "craniofacial"
    name: "Craniofacial Surgery Specialist"
    pubmed_keywords:
      - "cleft lip"
      - "cleft palate"
      - "craniosynostosis"
      - "craniofacial surgery"
      - "orthognathic surgery"
    mesh_terms:
      - "Cleft Lip[MeSH]"
      - "Cleft Palate[MeSH]"
    max_papers: 5000

  - agent_id: "hand_surgery"
    name: "Hand Surgery Specialist"
    pubmed_keywords:
      - "hand surgery"
      - "hand reconstruction"
      - "carpal tunnel syndrome"
      - "tendon repair"
    max_papers: 5000

  - agent_id: "breast"
    name: "Breast Surgery Specialist"
    pubmed_keywords:
      - "breast reconstruction"
      - "DIEP flap"
      - "TRAM flap"
      - "mastectomy reconstruction"
    max_papers: 5000

  # ... more agents
```

---

## 8. Query Classification and Routing

### 8.1 Multi-Label Subspecialty Classification

```python
class QueryClassifier:
    """
    Determines which specialist agents should be consulted
    """
    def __init__(self, agent_registry):
        self.agents = agent_registry

    def classify(self, query):
        """
        Returns list of relevant agent_ids with confidence scores
        """
        # Method 1: Medical NER + keyword matching
        entities = extract_medical_entities(query)
        keyword_matches = self._match_to_specialists(entities)

        # Method 2: Embedding similarity
        query_embedding = embed_query(query)
        embedding_matches = self._embedding_similarity(query_embedding)

        # Method 3: LLM-based classification (most accurate)
        llm_matches = self._llm_classify(query)

        # Combine all methods
        combined_scores = self._combine_classifications(
            keyword_matches,
            embedding_matches,
            llm_matches
        )

        # Return agents with score > threshold
        relevant_agents = [
            agent_id for agent_id, score in combined_scores.items()
            if score > 0.3
        ]

        return relevant_agents, combined_scores

    def _llm_classify(self, query):
        """
        Use LLM to classify query to subspecialties
        """
        classification_prompt = f"""
You are a medical query router for plastic surgery subspecialties.

Available subspecialties:
{self._format_agent_descriptions()}

Query: "{query}"

Determine which subspecialty specialists should be consulted.
A query may require multiple specialists.

Respond with JSON:
{{
    "agent_id": confidence_score (0.0 to 1.0)
}}
"""
        response = self.llm.generate(classification_prompt)
        scores = json.loads(response)

        return scores
```

### 8.2 Routing Decision Logic

```python
class QueryRouter:
    """
    Routes queries to appropriate specialist agents
    """
    def __init__(self, agents, classifier):
        self.agents = agents
        self.classifier = classifier

    def route(self, query):
        """
        Determine routing strategy
        """
        relevant_agents, confidence_scores = self.classifier.classify(query)

        if len(relevant_agents) == 0:
            return {
                "routing_strategy": "no_match",
                "agents": [],
                "message": "Unable to match query to any subspecialty."
            }

        elif len(relevant_agents) == 1:
            return {
                "routing_strategy": "single_agent",
                "agents": relevant_agents,
                "primary_agent": relevant_agents[0],
                "confidence_scores": confidence_scores
            }

        else:
            # Multiple agents needed
            ranked_agents = sorted(
                relevant_agents,
                key=lambda x: confidence_scores[x],
                reverse=True
            )

            return {
                "routing_strategy": "multi_agent",
                "agents": ranked_agents,
                "primary_agent": ranked_agents[0],
                "secondary_agents": ranked_agents[1:],
                "confidence_scores": confidence_scores
            }
```

---

## 9. Retrieval Layer

### 9.1 Hybrid Search Strategy

**Combines dense (semantic) and sparse (keyword) search:**

```python
class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse search
    """
    def __init__(self, vector_store, collection_name):
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.dense_weight = 0.7
        self.sparse_weight = 0.3

    def retrieve(self, query, filters=None, top_k=20):
        """
        Hybrid retrieval with reranking
        """
        # 1. Dense vector search (semantic)
        dense_results = self.dense_search(
            query=query,
            filters=filters,
            top_k=top_k * 2
        )

        # 2. Sparse search (BM25-like keyword matching)
        sparse_results = self.sparse_search(
            query=query,
            filters=filters,
            top_k=top_k * 2
        )

        # 3. Reciprocal Rank Fusion
        combined = self.reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=60
        )

        # 4. Rerank with cross-encoder
        reranked = self.rerank(query, combined[:top_k*2])

        return reranked[:top_k]

    def reciprocal_rank_fusion(self, list1, list2, k=60):
        """
        RRF: score(d) = sum(1 / (k + rank_i(d)))
        """
        scores = defaultdict(float)

        for rank, item in enumerate(list1):
            scores[item.id] += 1 / (k + rank + 1)

        for rank, item in enumerate(list2):
            scores[item.id] += 1 / (k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 9.2 Medical Metadata Filtering

Apply filters to ensure relevance:

```python
def apply_medical_filters(query_type, subspecialty=None):
    """
    Apply appropriate filters based on query
    """
    filters = {
        "must": [],
        "should": []
    }

    # 1. Subspecialty filter
    if subspecialty:
        filters["must"].append({
            "key": "subspecialty",
            "match": {"value": subspecialty}
        })

    # 2. Evidence level filter (prefer higher quality)
    if query_type in ["procedure", "indication"]:
        filters["should"].extend([
            {"key": "evidence_level", "match": {"value": "I"}},
            {"key": "evidence_level", "match": {"value": "II"}},
        ])

    # 3. Recency boost (last 5 years)
    filters["should"].append({
        "key": "publication_date",
        "range": {"gte": "2021-01-01"}
    })

    # 4. Citation count boost
    filters["should"].append({
        "key": "citation_count",
        "range": {"gte": 10}
    })

    return filters
```

### 9.3 Reranking with Cross-Encoder

```python
from sentence_transformers import CrossEncoder

class MedicalReranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

    def rerank(self, query, candidates, top_k=10):
        """
        Rerank candidates using cross-encoder
        """
        pairs = [[query, cand.content] for cand in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [cand for cand, score in ranked[:top_k]]
```

---

## 10. Generation Layer

### 10.1 LLM Selection

**Recommended: Claude 3.5 Sonnet**
- Excellent reasoning and low hallucination
- Strong citation capabilities
- Good at following complex instructions

**Alternative: GPT-4**
- Strong general capability
- Good medical knowledge
- Requires strict prompting for citations

### 10.2 Prompt Engineering for Medical Accuracy

```python
MEDICAL_RAG_PROMPT = """
You are a medical research assistant specialized in {specialty}.

Your role is to synthesize information from published research papers.

CRITICAL RULES:
1. **Only use information from the provided research papers** below
2. **Cite every claim** with PMID and author in format: (Author et al., Year, PMID: xxxxxx)
3. **If papers don't contain the answer**, say so explicitly
4. **Use conservative language** - avoid absolute statements
5. **Distinguish evidence strength**:
   - Strong: Multiple high-quality studies agree
   - Moderate: Single study or mixed results
   - Limited: Case reports, small series
6. **Note**: "This information is for educational purposes. Clinical decisions should be made by qualified healthcare professionals."

RETRIEVED RESEARCH PAPERS:
{context}

QUESTION: {query}

ANSWER:
"""
```

### 10.3 Citation Enforcement

```python
def validate_citations(response, retrieved_chunks):
    """
    Ensure all claims are properly cited
    """
    claims = extract_claims(response)

    uncited_claims = []
    for claim in claims:
        if not has_citation(claim):
            # Try to find supporting chunk
            supporting_chunk = find_supporting_evidence(claim, retrieved_chunks)
            if supporting_chunk:
                claim_with_citation = f"{claim} ({supporting_chunk.citation_text})"
            else:
                uncited_claims.append(claim)

    if uncited_claims:
        response += "\n\n[Note: Some claims could not be verified in provided sources]"

    return response
```

### 10.4 Hallucination Prevention

```python
def verify_facts(response, retrieved_chunks):
    """
    Check if facts in response are supported by chunks
    """
    facts = extract_factual_claims(response)

    unsupported_facts = []
    for fact in facts:
        if not any(fact_matches(fact, chunk.content) for chunk in retrieved_chunks):
            unsupported_facts.append(fact)

    if unsupported_facts:
        logger.warning(f"Unsupported facts: {unsupported_facts}")
        return False

    return True
```

---

## 11. Conversation Memory System

### 11.1 Session Storage

```python
@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    turn_id: str
    turn_number: int
    timestamp: datetime
    user_query: str
    rewritten_query: Optional[str] = None
    routing_decision: Dict = field(default_factory=dict)
    consulted_agents: List[str] = field(default_factory=list)
    retrieved_papers: Dict[str, List] = field(default_factory=dict)
    response: Dict = field(default_factory=dict)
    processing_time: float = 0.0

@dataclass
class ConversationSession:
    """Complete conversation session with memory"""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    turns: List[ConversationTurn] = field(default_factory=list)

    # Cumulative context
    active_topics: List[str] = field(default_factory=list)
    active_subspecialties: List[str] = field(default_factory=list)
    paper_cache: Dict[str, Dict] = field(default_factory=dict)
    mentioned_entities: Dict[str, List] = field(default_factory=dict)
    current_focus: Optional[str] = None
    comparison_mode: bool = False

    def get_recent_turns(self, n: int = 3) -> List[ConversationTurn]:
        """Get last N turns"""
        return self.turns[-n:] if len(self.turns) >= n else self.turns
```

### 11.2 Context-Aware Query Rewriting

```python
class ContextualQueryRewriter:
    """
    Rewrites follow-up queries using conversation context
    """
    def __init__(self, llm):
        self.llm = llm

    def rewrite_query(self, current_query: str, session: ConversationSession) -> str:
        """
        Rewrite query with context from conversation history
        """
        # Check if query needs context
        if not self._needs_context(current_query):
            return current_query

        # Get relevant context
        context = self._build_context(session)

        # Rewrite query
        rewritten = self._perform_rewrite(current_query, context)

        return rewritten

    def _needs_context(self, query: str) -> bool:
        """
        Determine if query needs context from previous turns
        """
        context_indicators = [
            r'\b(it|that|this|these|those|them)\b',  # Pronouns
            r'\b(compare|versus|vs|instead)\b',  # Comparisons
            r'\b(what about|how about|also)\b',  # Follow-up phrases
            r'\b(more|less|better|worse)\b',  # Relative terms
        ]

        import re
        for pattern in context_indicators:
            if re.search(pattern, query.lower()):
                return True

        return len(query.split()) < 5  # Short queries likely need context

    def _build_context(self, session: ConversationSession) -> str:
        """
        Build context summary from conversation history
        """
        recent_turns = session.get_recent_turns(n=3)

        context_parts = ["RECENT CONVERSATION:"]
        for turn in recent_turns:
            context_parts.append(f"Q{turn.turn_number}: {turn.user_query}")

        if session.active_topics:
            context_parts.append(f"ACTIVE TOPICS: {', '.join(session.active_topics)}")

        if session.current_focus:
            context_parts.append(f"CURRENT FOCUS: {session.current_focus}")

        return '\n'.join(context_parts)

    def _perform_rewrite(self, query: str, context: str) -> str:
        """
        Use LLM to rewrite query with context
        """
        prompt = f"""
{context}

CURRENT FOLLOW-UP QUESTION: "{query}"

Rewrite this as a standalone, self-contained question that includes necessary context.

EXAMPLES:
Context: Q1: "What are DIEP flap complications?"
Follow-up: "What about in obese patients?"
Rewritten: "What are DIEP flap complications in obese patients?"

Context: Q1: "What is the Furlow technique?"
Follow-up: "Compare that to straight-line repair"
Rewritten: "Compare the Furlow technique to straight-line repair for cleft palate"

REWRITTEN QUESTION:"""

        rewritten = self.llm.generate(prompt, temperature=0.1)
        return rewritten.strip().strip('"')
```

### 11.3 Paper Caching

```python
class PaperCache:
    """
    Cache retrieved papers to avoid redundant retrieval
    """
    def __init__(self):
        self.cache = {}  # session_id -> {agent_id -> {pmid -> paper}}

    def add_papers(self, session_id: str, agent_id: str, papers: List[Dict]):
        """Add papers to cache"""
        if session_id not in self.cache:
            self.cache[session_id] = {}

        if agent_id not in self.cache[session_id]:
            self.cache[session_id][agent_id] = {}

        for paper in papers:
            self.cache[session_id][agent_id][paper['pmid']] = paper

    def get_relevant_cached_papers(self, session_id: str, agent_id: str,
                                   query: str, top_k: int = 5) -> List[Dict]:
        """Get cached papers relevant to current query"""
        if session_id not in self.cache or agent_id not in self.cache[session_id]:
            return []

        cached_papers = list(self.cache[session_id][agent_id].values())

        # Re-rank by relevance to query
        relevance_scores = [
            (paper, self._calculate_relevance(query, paper))
            for paper in cached_papers
        ]
        relevance_scores.sort(key=lambda x: x[1], reverse=True)

        return [paper for paper, score in relevance_scores[:top_k]]
```

### 11.4 Conversation Manager

```python
class ConversationManager:
    """
    Main orchestrator for conversational RAG
    """
    def __init__(self, multi_agent_system, session_store, query_rewriter,
                 router, paper_cache):
        self.multi_agent = multi_agent_system
        self.session_store = session_store
        self.query_rewriter = query_rewriter
        self.router = router
        self.paper_cache = paper_cache

    def start_conversation(self, user_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        session = self.session_store.create_session(user_id=user_id)
        return session.session_id

    def process_query(self, session_id: str, query: str) -> Dict:
        """Process a query within a conversation session"""
        # Load session
        session = self.session_store.load_session(session_id)

        # Create turn
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            turn_number=len(session.turns) + 1,
            timestamp=datetime.now(),
            user_query=query
        )

        # Step 1: Rewrite query with context
        if len(session.turns) > 0:
            rewritten_query = self.query_rewriter.rewrite_query(query, session)
            turn.rewritten_query = rewritten_query
        else:
            rewritten_query = query

        print(f"Original: {query}")
        print(f"Rewritten: {rewritten_query}")

        # Step 2: Context-aware routing
        routing_decision = self.router.route_with_context(
            query, rewritten_query, session
        )
        turn.routing_decision = routing_decision
        turn.consulted_agents = routing_decision['agents']

        # Step 3: Retrieve with cache
        retrieval_results = {}
        cache_stats = {}

        for agent_id in routing_decision['agents']:
            agent = self.multi_agent.agents[agent_id]

            # Use cached papers
            cached_papers = self.paper_cache.get_relevant_cached_papers(
                session_id, agent_id, rewritten_query, top_k=5
            )

            # Retrieve new papers
            num_new_needed = max(0, 10 - len(cached_papers))
            if num_new_needed > 0:
                new_papers = agent.retrieve(rewritten_query, top_k=num_new_needed)
                self.paper_cache.add_papers(session_id, agent_id, new_papers)
            else:
                new_papers = []

            all_papers = cached_papers + new_papers
            retrieval_results[agent_id] = all_papers

            cache_stats[agent_id] = {
                "cached": len(cached_papers),
                "new": len(new_papers),
                "total": len(all_papers)
            }

        turn.retrieved_papers = retrieval_results

        # Step 4: Generate answers
        agent_responses = self.multi_agent._parallel_answer(
            rewritten_query,
            retrieval_results
        )

        # Step 5: Synthesize with conversation context
        response = self._synthesize_with_context(
            query, rewritten_query, agent_responses, routing_decision, session
        )

        turn.response = response

        # Step 6: Update session state
        self._update_session_state(session, turn, response)

        # Step 7: Save session
        session.add_turn(turn)
        self.session_store.save_session(session)

        # Add metadata
        response['session_id'] = session_id
        response['turn_number'] = turn.turn_number
        response['cache_stats'] = cache_stats

        return response

    def _update_session_state(self, session, turn, response):
        """Update session state based on current turn"""
        # Extract topics
        topics = self._extract_topics(turn, response)
        for topic in topics:
            if topic not in session.active_topics:
                session.active_topics.append(topic)

        session.active_topics = session.active_topics[-5:]  # Keep last 5

        # Add subspecialties
        for agent_id in turn.consulted_agents:
            if agent_id not in session.active_subspecialties:
                session.active_subspecialties.append(agent_id)

        # Update current focus
        session.current_focus = topics[0] if topics else None

        # Detect comparison mode
        session.comparison_mode = 'compare' in turn.user_query.lower()
```

---

## 12. Multi-Agent Consultation

### 12.1 Parallel Agent Execution

```python
class MultiAgentConsultation:
    """
    Manages consultation with multiple specialist agents
    """
    def __init__(self, agents, router):
        self.agents = agents
        self.router = router
        self.coordinator = CoordinatorAgent()

    def consult(self, query):
        """Main consultation flow"""
        # Step 1: Route query
        routing_decision = self.router.route(query)

        if routing_decision['routing_strategy'] == 'no_match':
            return routing_decision

        # Step 2: Retrieve from agents in parallel
        agent_ids = routing_decision['agents']
        retrieval_results = self._parallel_retrieve(query, agent_ids)

        # Step 3: Generate answers in parallel
        agent_responses = self._parallel_answer(query, retrieval_results)

        # Step 4: Coordinate responses
        if routing_decision['routing_strategy'] == 'single_agent':
            final_response = agent_responses[agent_ids[0]]
        else:
            final_response = self.coordinator.synthesize(
                query, agent_responses, routing_decision
            )

        return final_response

    def _parallel_retrieve(self, query, agent_ids):
        """Retrieve from multiple agents in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        retrieval_results = {}

        def retrieve_for_agent(agent_id):
            agent = self.agents[agent_id]
            docs = agent.retrieve(query, top_k=10)
            return agent_id, docs

        with ThreadPoolExecutor(max_workers=len(agent_ids)) as executor:
            futures = [
                executor.submit(retrieve_for_agent, agent_id)
                for agent_id in agent_ids
            ]

            for future in as_completed(futures):
                agent_id, docs = future.result()
                retrieval_results[agent_id] = docs

        return retrieval_results

    def _parallel_answer(self, query, retrieval_results):
        """Generate answers from multiple agents in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        agent_responses = {}

        def answer_for_agent(agent_id, docs):
            agent = self.agents[agent_id]
            response = agent.answer(query, docs)
            return agent_id, response

        with ThreadPoolExecutor(max_workers=len(retrieval_results)) as executor:
            futures = [
                executor.submit(answer_for_agent, agent_id, docs)
                for agent_id, docs in retrieval_results.items()
            ]

            for future in as_completed(futures):
                agent_id, response = future.result()
                agent_responses[agent_id] = response

        return agent_responses
```

### 12.2 Coordinator Agent

```python
class CoordinatorAgent:
    """
    Coordinates and synthesizes responses from multiple specialists
    """
    def __init__(self):
        self.llm = AnthropicLLM(model="claude-3-5-sonnet-20241022")

    def synthesize(self, query, agent_responses, routing_decision):
        """Synthesize multiple specialist perspectives"""
        synthesis_prompt = self._build_synthesis_prompt(
            query, agent_responses, routing_decision
        )

        synthesized = self.llm.generate(synthesis_prompt)

        return {
            "query": query,
            "routing_strategy": routing_decision['routing_strategy'],
            "consulted_specialists": list(agent_responses.keys()),
            "synthesized_answer": synthesized,
            "individual_specialist_responses": self._format_individual_responses(agent_responses),
            "all_citations": self._collect_all_citations(agent_responses),
            "consensus_level": self._assess_consensus(agent_responses),
            "evidence_summary": self._summarize_evidence(agent_responses)
        }

    def _build_synthesis_prompt(self, query, agent_responses, routing_decision):
        """Build prompt for response synthesis"""
        prompt = f"""
You are a medical coordinator synthesizing input from multiple plastic surgery specialists.

ORIGINAL QUERY: "{query}"

SPECIALIST CONSULTATIONS:
"""
        for agent_id, response in agent_responses.items():
            prompt += f"""
─────────────────────────────────────────
SPECIALIST: {response['agent_name']}
CONFIDENCE: {response['confidence']:.2f}

ANSWER:
{response['answer']}

CITATIONS: {len(response['citations'])} papers
─────────────────────────────────────────
"""

        prompt += """

YOUR TASK:
1. Synthesize specialist perspectives into unified answer
2. When specialists agree, present consensus clearly
3. When specialists disagree, present all perspectives
4. Maintain all citations
5. Provide evidence quality assessment

FORMAT:
## Unified Answer
[Synthesized response]

## Specialist Perspectives
[If multiple viewpoints]

## Consensus Assessment
[Agreement level]

## Evidence Summary
[Overall quality]

## References
[All citations]
"""
        return prompt

    def _assess_consensus(self, agent_responses):
        """Assess agreement level between agents"""
        if len(agent_responses) < 2:
            return "single_specialist"

        # Extract key claims
        all_claims = []
        for response in agent_responses.values():
            claims = extract_key_claims(response['answer'])
            all_claims.append(set(claims))

        # Calculate overlap
        common_claims = set.intersection(*all_claims)
        total_claims = set.union(*all_claims)

        consensus_ratio = len(common_claims) / len(total_claims) if total_claims else 0

        if consensus_ratio > 0.7:
            return "high_consensus"
        elif consensus_ratio > 0.4:
            return "moderate_consensus"
        else:
            return "diverse_perspectives"

    def _collect_all_citations(self, agent_responses):
        """Collect and deduplicate citations"""
        all_citations = {}

        for agent_id, response in agent_responses.items():
            for citation in response['citations']:
                pmid = citation['pmid']
                if pmid not in all_citations:
                    all_citations[pmid] = citation
                    all_citations[pmid]['cited_by_agents'] = []

                all_citations[pmid]['cited_by_agents'].append(response['agent_name'])

        return list(all_citations.values())
```

---

## 13. API Layer

### 13.1 REST API Endpoints

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Conversational Multi-Agent Medical RAG")

class ConversationStart(BaseModel):
    user_id: Optional[str] = None

class ConversationQuery(BaseModel):
    query: str

@app.post("/conversation/start")
async def start_conversation(request: ConversationStart):
    """Start a new conversation session"""
    session_id = conversation_manager.start_conversation(user_id=request.user_id)

    return {
        "session_id": session_id,
        "message": "Conversation started. Use this session_id for follow-up queries."
    }

@app.post("/conversation/{session_id}/query")
async def conversation_query(session_id: str, request: ConversationQuery):
    """
    Send query within conversation
    Automatically handles context from previous turns
    """
    try:
        response = conversation_manager.process_query(
            session_id=session_id,
            query=request.query
        )
        return response

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/conversation/{session_id}/summary")
async def get_conversation_summary(session_id: str):
    """Get summary of conversation session"""
    summary = conversation_manager.get_conversation_summary(session_id)

    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])

    return summary

@app.get("/conversation/{session_id}/history")
async def get_conversation_history(session_id: str, last_n: int = 10):
    """Get conversation history"""
    session = session_store.load_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    recent_turns = session.get_recent_turns(n=last_n)

    return {
        "session_id": session_id,
        "total_turns": len(session.turns),
        "history": [
            {
                "turn_number": turn.turn_number,
                "query": turn.user_query,
                "rewritten_query": turn.rewritten_query,
                "consulted_agents": turn.consulted_agents,
                "timestamp": turn.timestamp.isoformat()
            }
            for turn in recent_turns
        ]
    }

@app.delete("/conversation/{session_id}")
async def end_conversation(session_id: str):
    """End a conversation session"""
    summary = conversation_manager.end_conversation(session_id)

    return {
        "message": "Conversation ended",
        "summary": summary
    }

@app.get("/agents")
async def list_agents():
    """List all available specialist agents"""
    return {
        "agents": [
            {
                "agent_id": agent_id,
                "name": agent.name,
                "keywords": agent.pubmed_keywords,
                "paper_count": agent.paper_count,
                "last_updated": agent.last_updated
            }
            for agent_id, agent in agents.items()
        ]
    }

@app.get("/agents/{agent_id}/stats")
async def get_agent_stats(agent_id: str):
    """Get statistics for specific agent"""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = agents[agent_id]
    stats = kb_stats.get_agent_stats(agent_id)

    return {
        "agent": agent.name,
        "knowledge_base": stats
    }
```

### 13.2 Client Usage Example

```python
import requests

# Start conversation
response = requests.post("http://localhost:8000/conversation/start")
session_id = response.json()["session_id"]

# First query
response = requests.post(
    f"http://localhost:8000/conversation/{session_id}/query",
    json={"query": "What are the complications of DIEP flap surgery?"}
)
print(response.json()['synthesized_answer'])

# Follow-up (context-aware)
response = requests.post(
    f"http://localhost:8000/conversation/{session_id}/query",
    json={"query": "What about in obese patients?"}
)
result = response.json()
print(f"Rewritten: {result['rewritten_query']}")
print(f"Cache stats: {result['cache_stats']}")
print(result['synthesized_answer'])

# Get conversation summary
response = requests.get(f"http://localhost:8000/conversation/{session_id}/summary")
print(response.json())
```

---

## 14. Evaluation and Quality Assurance

### 14.1 Evaluation Metrics

**Retrieval Metrics:**
- Recall@K: Are relevant papers retrieved?
- Precision@K: Are retrieved papers relevant?
- MRR: Mean Reciprocal Rank

**Generation Metrics:**
- Citation Rate: % of claims with citations
- Hallucination Rate: % of facts not in source
- Answer Relevance: Does answer address query?
- Faithfulness: Is answer faithful to sources?

**Medical-Specific Metrics:**
- Medical Accuracy: Verified by expert
- Evidence Quality Score: Quality of cited evidence
- Clinical Safety: No harmful advice

### 14.2 Evaluation Dataset

```python
EVALUATION_QUERIES = [
    {
        "query": "What are the indications for DIEP flap?",
        "subspecialty": "breast",
        "expected_agents": ["breast", "reconstructive"],
        "gold_standard_pmids": ["12345678", "23456789"]
    },
    {
        "query": "How do you treat facial burns in children?",
        "subspecialty": "burn",
        "expected_agents": ["burn", "pediatric", "reconstructive"],
        "expected_routing": "multi_agent"
    }
    # ... 100+ queries
]
```

### 14.3 Continuous Monitoring

```python
class RAGMonitor:
    def log_interaction(self, query, retrieved_chunks, response):
        """Log every interaction for analysis"""
        self.metrics.append({
            "timestamp": datetime.now(),
            "query": query,
            "num_chunks_retrieved": len(retrieved_chunks),
            "pmids_cited": self.extract_pmids(response),
            "avg_chunk_score": np.mean([c.score for c in retrieved_chunks])
        })

    def detect_anomalies(self):
        """Detect potential issues"""
        recent = self.metrics[-100:]
        avg_score = np.mean([m["avg_chunk_score"] for m in recent])

        if avg_score < 0.5:
            alert("Low retrieval scores detected")

        no_citation_rate = sum(1 for m in recent if not m["pmids_cited"]) / len(recent)
        if no_citation_rate > 0.1:
            alert("High rate of responses without citations")
```

---

## 15. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

**Week 1: Infrastructure Setup**
```bash
# Setup project
mkdir medical-rag-agents && cd medical-rag-agents

# Install dependencies
pip install -r requirements.txt

# Start infrastructure
docker-compose up -d  # Qdrant, Redis, PostgreSQL

# Configure agents
cp config/agents.yaml.template config/agents.yaml
# User fills in PubMed keywords
```

**Week 2: First Agent**
```bash
# Initialize single agent (test)
python scripts/initialize_agent.py --agent-id craniofacial

# Test retrieval
python scripts/test_retrieval.py --agent-id craniofacial

# Test generation
python scripts/test_single_agent.py --agent-id craniofacial
```

**Week 3: Core RAG Pipeline**
- Document processing (GROBID)
- Chunking implementation
- Vector store setup
- Basic retrieval + generation

### Phase 2: Multi-Agent System (Weeks 4-6)

**Week 4: Multiple Agents**
```bash
# Initialize all agents
python scripts/initialize_all_agents.py

# Monitor progress
python scripts/monitor_initialization.py
```

**Week 5: Routing & Coordination**
- Query classification
- Multi-agent routing
- Coordinator agent
- Parallel execution

**Week 6: Integration Testing**
- Test single-agent queries
- Test multi-agent queries
- Test consensus building

### Phase 3: Conversational Memory (Weeks 7-8)

**Week 7: Memory System**
- Session storage (Redis)
- Query rewriter
- Paper caching
- Context builder

**Week 8: Integration**
- Integrate memory with multi-agent
- Test conversational flows
- Performance optimization

### Phase 4: Production (Weeks 9-12)

**Week 9: API Development**
- FastAPI implementation
- Authentication
- Rate limiting
- Documentation

**Week 10: Evaluation**
- Build evaluation dataset
- Run metrics
- Expert review
- Iterate on quality

**Week 11: Optimization**
- Caching improvements
- Query optimization
- Cost reduction
- Performance tuning

**Week 12: Deployment**
- Production deployment
- Monitoring setup
- User documentation
- Training materials

---

## 16. Technology Stack

### Core Stack

```yaml
Data Acquisition:
  - PubMed E-utilities API (NCBI)
  - Biopython
  - Requests

Document Processing:
  - GROBID (PDF extraction)
  - PyMuPDF (fallback)
  - scispaCy (medical NER)
  - spaCy

Embeddings:
  - Primary: PubMedBERT (HuggingFace)
  - Alternative: OpenAI text-embedding-3-large
  - SentenceTransformers

Vector Store:
  - Qdrant (primary)
  - Alternatives: Weaviate, Pinecone

Retrieval:
  - Qdrant client
  - CrossEncoder (reranking)

Generation:
  - Claude 3.5 Sonnet (Anthropic)
  - Alternative: GPT-4 (OpenAI)

Conversation Memory:
  - Redis (session storage)
  - Alternative: PostgreSQL

Orchestration:
  - Custom pipeline (full control)
  - Optional: LangChain patterns

API:
  - FastAPI
  - Pydantic
  - Uvicorn

Monitoring:
  - Prometheus + Grafana
  - Custom logging

Infrastructure:
  - Docker
  - Docker Compose
```

### Python Requirements

```
# requirements.txt
anthropic>=0.18.0
openai>=1.0.0
qdrant-client>=1.7.0
sentence-transformers>=2.3.0
transformers>=4.35.0
torch>=2.1.0
spacy>=3.7.0
scispacy>=0.5.3
biopython>=1.81
pymupdf>=1.23.0
redis>=5.0.0
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
pandas>=2.1.0
numpy>=1.24.0
sqlalchemy>=2.0.0
python-dotenv>=1.0.0
```

---

## 17. Configuration Management

### 17.1 Main Configuration File

```yaml
# config/system.yaml

system:
  name: "Medical RAG System"
  version: "1.0.0"

  # APIs
  pubmed_api_key: "${PUBMED_API_KEY}"
  anthropic_api_key: "${ANTHROPIC_API_KEY}"

  # Infrastructure
  vector_store:
    type: "qdrant"
    url: "localhost:6333"

  session_store:
    type: "redis"
    url: "redis://localhost:6379"
    ttl_days: 7

  # Models
  embedding_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
  llm_provider: "anthropic"
  llm_model: "claude-3-5-sonnet-20241022"

# Retrieval settings
retrieval:
  top_k_per_agent: 10
  hybrid_search:
    dense_weight: 0.7
    sparse_weight: 0.3
  reranking: true

# Generation settings
generation:
  temperature: 0.1
  max_tokens: 2000
  require_citations: true

# Conversation settings
conversation:
  enable_query_rewriting: true
  enable_paper_cache: true
  max_turns_in_context: 3
  auto_cleanup_days: 7
```

### 17.2 Agent Configuration

```yaml
# config/agents.yaml

agents:
  - agent_id: "craniofacial"
    name: "Craniofacial Surgery Specialist"
    description: "Expert in cleft lip/palate, craniosynostosis, facial deformities"

    # USER PROVIDES THESE KEYWORDS
    pubmed_keywords:
      - "cleft lip"
      - "cleft palate"
      - "craniosynostosis"
      - "craniofacial surgery"

    # Optional MeSH terms
    mesh_terms:
      - "Cleft Lip[MeSH]"
      - "Cleft Palate[MeSH]"

    max_papers: 5000
    update_frequency: "monthly"

  - agent_id: "breast"
    name: "Breast Surgery Specialist"
    description: "Expert in breast reconstruction"

    pubmed_keywords:
      - "breast reconstruction"
      - "DIEP flap"
      - "TRAM flap"
      - "mastectomy reconstruction"

    max_papers: 5000

  # ... more agents
```

---

## 18. Monitoring and Analytics

### 18.1 System Dashboard

```python
class SystemDashboard:
    """Real-time monitoring"""

    def get_system_stats(self):
        return {
            "total_agents": len(self.agents),
            "total_papers": sum(a.paper_count for a in self.agents.values()),
            "total_queries": self.get_total_queries(),
            "active_sessions": self.get_active_session_count(),
            "avg_response_time": self.get_avg_response_time()
        }

    def get_agent_performance(self):
        return {
            agent_id: {
                "queries": self.get_query_count(agent_id),
                "avg_confidence": self.get_avg_confidence(agent_id),
                "cache_hit_rate": self.get_cache_hit_rate(agent_id)
            }
            for agent_id in self.agents.keys()
        }
```

### 18.2 Conversation Analytics

```python
class ConversationAnalytics:
    """Analyze conversation patterns"""

    def get_conversation_stats(self):
        return {
            "avg_turns_per_conversation": self.calculate_avg_turns(),
            "most_common_topics": self.get_top_topics(n=10),
            "most_consulted_agents": self.get_agent_usage_stats(),
            "avg_session_duration": self.calculate_avg_duration(),
            "cache_efficiency": self.calculate_cache_efficiency()
        }
```

---

## 19. Cost Estimation

### Monthly Costs (10,000 queries/month)

**Storage:**
- Vector DB (Qdrant Cloud): $50-200/month (or $0 self-hosted)
- Redis (session store): $20-50/month
- Document storage (S3): $10-50/month

**Compute:**
- Embedding generation: $20-50/month (one-time for initial ingestion)
- LLM API (Claude 3.5 Sonnet):
  - Avg 10K input tokens/query: $0.15/query
  - Avg 1K output tokens/query: $0.75/query
  - **Total: ~$9,000/month for 10K queries**

**With Caching Benefits:**
- First query: Full cost (~$0.90)
- Follow-up queries: ~50% cost savings due to paper caching
- Effective cost: ~$6,000-7,000/month with conversational usage

**Alternative (Self-Hosted LLM):**
- GPU server: $500-2,000/month
- Llama 3 70B or BioGPT
- **Total: $600-2,400/month** (significant savings at scale)

**Recommendation:**
- Start with API-based (Claude) for quality
- Monitor query volume and costs
- Consider self-hosted if volume exceeds 20K queries/month

---

## 20. Appendices

### Appendix A: Directory Structure

```
medical-rag-agents/
├── config/
│   ├── system.yaml
│   ├── agents.yaml              # User configures
│   └── agents.yaml.template
├── data/
│   ├── agents/
│   │   ├── craniofacial/
│   │   │   ├── raw_papers/
│   │   │   └── processed/
│   │   ├── hand_surgery/
│   │   └── [other agents]/
│   ├── evaluation/
│   └── kb_statistics.json
├── src/
│   ├── agents/
│   │   ├── specialist_agent.py
│   │   ├── coordinator_agent.py
│   │   └── agent_registry.py
│   ├── acquisition/
│   │   ├── pubmed_fetcher.py
│   │   └── kb_builder.py
│   ├── processing/
│   │   ├── pdf_extractor.py
│   │   ├── chunker.py
│   │   └── entity_extractor.py
│   ├── vectorstore/
│   │   ├── qdrant_manager.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── llm_client.py
│   │   └── prompt_templates.py
│   ├── conversation/
│   │   ├── conversation_manager.py
│   │   ├── session_store.py
│   │   ├── query_rewriter.py
│   │   └── paper_cache.py
│   ├── routing/
│   │   ├── query_classifier.py
│   │   └── router.py
│   └── api/
│       ├── main.py
│       └── models.py
├── scripts/
│   ├── initialize_agent.py
│   ├── initialize_all_agents.py
│   ├── test_conversation.py
│   └── update_agents.py
├── tests/
│   ├── test_agents.py
│   ├── test_routing.py
│   ├── test_conversation.py
│   └── test_generation.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── README.md
└── docs/
    ├── setup_guide.md
    ├── api_documentation.md
    └── user_guide.md
```

### Appendix B: Example Conversations

**Example 1: Single Specialty Deep Dive**
```
User: Start conversation
System: Session abc-123 started

Turn 1:
User: "What are the surgical techniques for cleft lip repair?"
System: [Craniofacial Agent]
        10 papers retrieved
        Response with multiple techniques

Turn 2:
User: "What about the Millard technique specifically?"
System: Original: "What about the Millard technique specifically?"
        Rewritten: "What are the details of the Millard rotation-advancement technique for cleft lip repair?"
        [Craniofacial Agent]
        Cache: 3 papers reused, 7 new papers
        Detailed response about Millard technique

Turn 3:
User: "What are the outcomes?"
System: Rewritten: "What are the outcomes of the Millard rotation-advancement technique for cleft lip repair?"
        Cache: 5 papers reused
        Response with outcome data and statistics
```

**Example 2: Multi-Specialty Consultation**
```
Turn 1:
User: "How do you treat facial burns in children?"
System: [Burn + Pediatric + Reconstructive Agents]
        Multi-specialist response

Turn 2:
User: "What about scar management?"
System: Rewritten: "What are scar management strategies for facial burns in children?"
        [Burn + Pediatric Agents]
        Cache: 6 papers reused
        Focused response on scar management

Turn 3:
User: "When do you do reconstruction?"
System: Rewritten: "When is the optimal timing for facial reconstruction after burns in children?"
        [All 3 agents re-engaged]
        Response on reconstruction timing
```

### Appendix C: Quick Start Guide

```bash
# 1. Clone repository
git clone <repo-url>
cd medical-rag-agents

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment variables
cp .env.example .env
# Edit .env with your API keys:
# - PUBMED_API_KEY
# - ANTHROPIC_API_KEY

# 4. Configure agents
cp config/agents.yaml.template config/agents.yaml
# Edit agents.yaml and add your PubMed keywords for each specialty

# 5. Start infrastructure
docker-compose up -d

# 6. Initialize first agent (test)
python scripts/initialize_agent.py --agent-id craniofacial

# 7. Test the system
python scripts/test_conversation.py

# 8. Initialize all agents
python scripts/initialize_all_agents.py

# 9. Start API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 10. Access API documentation
# Open browser: http://localhost:8000/docs
```

### Appendix D: Troubleshooting

**Common Issues:**

1. **GROBID not working**
   ```bash
   # Start GROBID server
   docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.3
   ```

2. **Out of memory during embedding**
   ```python
   # Process in smaller batches
   batch_size = 100  # Reduce from 1000
   ```

3. **Slow retrieval**
   ```python
   # Check Qdrant indexes
   client.get_collection(collection_name)
   # Ensure indexes exist on frequently filtered fields
   ```

4. **High API costs**
   ```python
   # Increase cache usage
   cache_threshold = 0.7  # Lower threshold for cache reuse
   ```

---

## Document Control

**Version:** 3.0
**Date:** March 12, 2026
**Status:** Complete Comprehensive Design

**Key Features:**
- ✅ Multi-agent specialist system
- ✅ User-configured PubMed keywords
- ✅ Conversational memory with context
- ✅ Paper caching for efficiency
- ✅ Medical accuracy focus with citations
- ✅ Parallel agent consultation
- ✅ Complete implementation roadmap

**Next Steps:**
1. User provides PubMed keywords for each subspecialty in `config/agents.yaml`
2. Begin Phase 1 implementation
3. Deploy first pilot agent
4. Scale to full multi-agent system

---

**END OF COMPREHENSIVE DESIGN DOCUMENT**
