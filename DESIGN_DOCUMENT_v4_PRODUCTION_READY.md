# Production-Ready Medical RAG System - Design Document v4.0

**Version:** 4.0 (Production Architecture)
**Date:** March 12, 2026
**Status:** Planning Phase - Production-Ready Design

---

## Document Purpose

This v4.0 revision addresses **critical architectural flaws and production concerns** identified in expert review of v3.0. This is a planning document - no implementation has begun.

### What Changed from v3.0

| Area | v3.0 Issue | v4.0 Solution |
|------|-----------|---------------|
| **Knowledge Store** | Separate vector DB per agent → redundancy, inconsistency | Single canonical corpus with agent filtering |
| **Generation Safety** | Post-hoc validation → hallucination risk | Evidence-first generation |
| **Clinical Safety** | Disclaimer only | Active safety classification layer |
| **Chunking** | Semantic similarity only → breaks context | Section-aware + structure preservation |
| **Query Rewrite** | Silent rewriting → semantic drift | Confidence scoring + user visibility |
| **Tables** | Text summaries → precision loss | Structured extraction |
| **Cost Control** | Monitoring only | Hard budgets + cost-aware routing |
| **Failure Handling** | Assumed success | Explicit degraded modes |

---

## Table of Contents

1. [Non-Goals and System Boundaries](#1-non-goals-and-system-boundaries)
2. [Critical Risks and Mitigations](#2-critical-risks-and-mitigations)
3. [Core Architecture Changes](#3-core-architecture-changes)
4. [Shared Canonical Corpus](#4-shared-canonical-corpus)
5. [Clinical Safety Layer](#5-clinical-safety-layer)
6. [Evidence-First Generation](#6-evidence-first-generation)
7. [Structure-Aware Chunking](#7-structure-aware-chunking)
8. [Query Rewriting with Confidence](#8-query-rewriting-with-confidence)
9. [Structured Table Handling](#9-structured-table-handling)
10. [Cost and Latency Budgets](#10-cost-and-latency-budgets)
11. [Deduplication Strategy](#11-deduplication-strategy)
12. [Operational Safeguards](#12-operational-safeguards)
13. [Security and Compliance](#13-security-and-compliance)
14. [Evidence Grading System](#14-evidence-grading-system)
15. [Implementation Priorities](#15-implementation-priorities)

---

## 1. Non-Goals and System Boundaries

### What This System Is NOT

❌ **Not a clinical decision support tool**
Does not provide patient-specific treatment recommendations

❌ **Not a replacement for clinician judgment**
Summarizes evidence; does not make clinical decisions

❌ **Not comprehensive literature review**
Best-effort retrieval from configured sources, not exhaustive

❌ **Not regulatory-grade medical device**
For research and education, not diagnostic/treatment purposes

❌ **Not real-time guideline tracker**
Ingestion is periodic, not continuous

❌ **Not PHI-handling system**
Does not accept, store, or process patient health information

### Explicit Boundaries

**Input Boundaries:**
- ✅ General medical questions about procedures, techniques, outcomes
- ⚠️ Patient-specific queries → downgrade to evidence-only mode
- ❌ Emergency queries, dose calculations, diagnostic interpretations

**Output Boundaries:**
- ✅ Evidence summaries with citations and quality grading
- ✅ Comparative analysis of published techniques
- ❌ Specific treatment recommendations
- ❌ Patient management decisions

---

## 2. Critical Risks and Mitigations

### Risk 1: Hallucination (Unsupported Medical Claims)
**Severity:** 🔴 Critical (Patient Safety)

**Problem:**
- LLM generates plausible medical claims not in retrieved papers
- Citations fabricated or misattributed
- Statistics invented

**v3.0 Approach (Flawed):**
```
Generate answer → Validate citations → Flag problems
Problem: Unsupported content already generated
```

**v4.0 Solution: Evidence-First Generation**
```
1. Extract atomic claims from retrieved chunks
2. Map query to relevant claims
3. Generate ONLY from mapped claims
4. Attach chunk_id to each sentence
5. Reject sentences without chunk_id
```

**Implementation:**
```python
# Before generating, create evidence map
evidence_map = {
    "claim": "DIEP flap survival 95-98%",
    "chunk_id": "chunk-123",
    "pmid": "12345678",
    "sentence_location": "Results, para 2"
}

# Generate with constraint
prompt = f"""
Using ONLY these evidence claims:
{format_evidence_claims(evidence_map)}

Answer: {query}

Rules:
- Use only provided claims
- Cite chunk_id for each statement
- If claim not in evidence, say "not found in sources"
"""
```

---

### Risk 2: Query Rewrite Drift (Semantic Distortion)
**Severity:** 🟡 High (Clinical Accuracy)

**Problem:**
```
Turn 1: "DIEP flap outcomes in diabetics"
Turn 2: "What about complications?"
Wrong Rewrite: "DIEP flap outcomes complications in diabetics"
Should Be: "DIEP flap complications in diabetics"
```

**v3.0 Approach (Flawed):**
- Silent rewriting, user unaware

**v4.0 Solution:**

1. **Confidence Scoring**
```python
def rewrite_with_confidence(query, context):
    rewritten = llm.rewrite(query, context)

    # Measure semantic drift
    orig_emb = embed(query)
    rewrite_emb = embed(rewritten)
    confidence = cosine_similarity(orig_emb, rewrite_emb)

    # Require high confidence
    if confidence < 0.7:
        return {
            "use_original": True,
            "rewritten": query,
            "confidence": confidence,
            "reason": "Low confidence in rewrite"
        }

    return {
        "use_original": False,
        "rewritten": rewritten,
        "confidence": confidence
    }
```

2. **User Visibility**
```json
Response includes:
{
  "original_query": "What about complications?",
  "interpreted_as": "What are DIEP flap complications in diabetic patients?",
  "rewrite_confidence": 0.89,
  "answer": "..."
}
```

User sees:
```
Your query: "What about complications?"
Interpreted as: "What are DIEP flap complications in diabetic patients?"
[Confidence: 89%]

Answer: ...
```

---

### Risk 3: Clinical Safety Boundary Violation
**Severity:** 🔴 Critical (Legal/Ethical)

**Problem:**
- System provides patient-specific treatment advice
- Answers "Should I do X for this patient?"
- Used by patients for self-diagnosis

**v4.0 Solution: Active Safety Classification**

```python
class ClinicalSafetyClassifier:
    """
    Detects queries that cross safety boundaries
    """
    PATIENT_SPECIFIC_PATTERNS = [
        r"my patient",
        r"this patient",
        r"should I (do|use|perform)",
        r"when should I",
        r"what dose",
        r"how much should I give"
    ]

    EMERGENCY_PATTERNS = [
        r"emergency",
        r"urgent",
        r"immediately",
        r"right now"
    ]

    def classify(self, query):
        """
        Classify query safety level
        """
        # Level 1: Patient-specific decision
        if self._is_patient_specific(query):
            return {
                "safety_level": "patient_specific",
                "response_mode": "evidence_only",
                "disclaimer": "strong"
            }

        # Level 2: Emergency
        if self._is_emergency(query):
            return {
                "safety_level": "emergency",
                "response_mode": "refuse",
                "message": "This system cannot provide emergency guidance. Seek immediate medical attention."
            }

        # Level 3: General literature query
        return {
            "safety_level": "literature_query",
            "response_mode": "full",
            "disclaimer": "standard"
        }
```

**Response Modification:**
```python
if safety_classification == "patient_specific":
    response = f"""
    ⚠️ This appears to be a patient-specific decision query.

    I can summarize published evidence on [topic], but cannot
    make patient-specific recommendations.

    Evidence Summary:
    {evidence_summary}

    Clinical decisions should be made by qualified healthcare
    professionals considering:
    - Individual patient factors
    - Complete medical history
    - Current clinical guidelines
    - Local practice standards
    """
```

---

### Risk 4: Chunking Breaks Evidence Context
**Severity:** 🟡 High (Evidence Integrity)

**Problem:**
```
Chunk 1: "DIEP flap survival was 97.3% (p<0.001)"
[separated by semantic chunking]
Chunk 2: "Significant limitations: retrospective, single-center, selection bias"

If only Chunk 1 retrieved → misleading
```

**v4.0 Solution: Section-Aware Chunking**

```python
class StructureAwareChunker:
    """
    Chunk within section boundaries, preserve study context
    """
    def chunk_paper(self, paper):
        chunks = []

        # Extract study metadata ONCE
        study_metadata = extract_study_structure(paper)
        # {design, population, intervention, n, limitations, ...}

        for section in paper.sections:
            # Chunk within section boundaries
            section_chunks = self._chunk_section(section)

            for chunk in section_chunks:
                # Inject study metadata into every chunk
                chunk.metadata['study_structure'] = study_metadata

                # Add section context
                chunk.content = f"""
Study: {paper.title}
Design: {study_metadata['design']} (n={study_metadata['n']})
Population: {study_metadata['population']}
Section: {section.name}

Content:
{chunk.raw_content}

Study Limitations: {', '.join(study_metadata['limitations'])}
"""
                chunks.append(chunk)

        return chunks
```

**Key Principle:** Every chunk carries full study context

---

### Risk 5: Table Information Loss
**Severity:** 🟡 High (Evidence Precision)

**Problem:**
```
Table 3: DIEP Flap Complications by BMI
| BMI Group | n  | Complication Rate | 95% CI      | p-value |
|-----------|----|--------------------|-------------|---------|
| <25       | 50 | 8.0%              | 2.1-13.9   | ref     |
| 25-30     | 85 | 12.9%             | 6.5-19.3   | 0.21    |
| >30       | 45 | 22.2%             | 10.1-34.3  | 0.03    |

Text summary: "Complication rates were higher in obese patients"
Lost: Exact rates, CIs, sample sizes, statistical significance
```

**v4.0 Solution: Structured Table Extraction**

```python
class StructuredTableExtractor:
    """
    Extract tables as structured, queryable objects
    """
    def extract_table(self, table_element):
        """
        Parse table into structured format
        """
        structured_table = {
            "table_id": generate_id(),
            "caption": table_element.caption,
            "headers": table_element.headers,
            "rows": table_element.rows,
            "cells": []
        }

        # Create searchable cell objects
        for row_idx, row in enumerate(table_element.rows):
            for col_idx, cell in enumerate(row):
                cell_obj = {
                    "cell_id": f"table_{table_id}_r{row_idx}_c{col_idx}",
                    "row": row_idx,
                    "column": col_idx,
                    "column_name": table_element.headers[col_idx],
                    "value": cell.value,
                    "value_type": infer_type(cell.value),  # number, percentage, text
                    "row_label": row[0],  # First column usually label

                    # Make cell searchable
                    "searchable_text": f"""
                    Table: {table_element.caption}
                    Row: {row[0]}
                    Column: {table_element.headers[col_idx]}
                    Value: {cell.value}
                    """
                }
                structured_table['cells'].append(cell_obj)

        return structured_table
```

**Storage:**
```python
# Store table cells as retrievable chunks
for cell in table['cells']:
    chunk = {
        "chunk_id": cell['cell_id'],
        "chunk_type": "table_cell",
        "content": cell['searchable_text'],
        "structured_data": cell,
        "table_id": table['table_id'],
        # ... standard metadata
    }
    vector_store.add(chunk)
```

**Retrieval:**
```python
# Query can retrieve specific table cells
query = "What is the complication rate in obese patients?"

# Retrieves chunk with:
# Row: BMI >30
# Column: Complication Rate
# Value: 22.2%
# Context: Full table reference
```

---

## 3. Core Architecture Changes

### 3.1 Architecture Comparison

**v3.0 Architecture (Problematic):**
```
User Query
    ↓
Conversation Manager
    ↓
Query Router
    ↓
╔════════════════════════════════════════╗
║  10 Separate Agent Knowledge Bases     ║
║  ┌─────────┐  ┌─────────┐             ║
║  │ Breast  │  │  Hand   │  ... (10x)  ║
║  │ VectorDB│  │ VectorDB│             ║
║  │ 5K papers│ │ 5K papers│            ║
║  └─────────┘  └─────────┘             ║
║  Problem: Massive redundancy           ║
╚════════════════════════════════════════╝
    ↓
Generate → Validate → Response
Problem: Post-hoc validation
```

**v4.0 Architecture (Production-Ready):**
```
User Query
    ↓
╔═══════════════════════════════════════════════╗
║  CLINICAL SAFETY LAYER (NEW)                  ║
║  Classifies: Literature | Patient | Emergency ║
╚═══════════════════════════════════════════════╝
    ↓
Conversation Manager (with confidence scoring)
    ↓
Query Router
    ↓
╔═══════════════════════════════════════════════╗
║  SINGLE CANONICAL CORPUS (NEW)                ║
║  ┌────────────────────────────────────────┐   ║
║  │ One Vector DB: 10K unique papers      │   ║
║  │ Multi-labeled with subspecialty tags  │   ║
║  │ Agents filter by tag, not separate DB │   ║
║  └────────────────────────────────────────┘   ║
║  Benefit: No redundancy, consistent answers   ║
╚═══════════════════════════════════════════════╝
    ↓
╔═══════════════════════════════════════════════╗
║  EVIDENCE-FIRST GENERATION (NEW)              ║
║  1. Extract claims from chunks                ║
║  2. Map claims to query                       ║
║  3. Generate from mapped claims only          ║
║  4. Attach chunk_id to every sentence         ║
╚═══════════════════════════════════════════════╝
    ↓
Response with provenance
```

### 3.2 Key Changes Summary

| Component | Change | Benefit |
|-----------|--------|---------|
| Knowledge Store | 10 separate DBs → 1 shared corpus | 80% storage reduction, consistency |
| Generation | Post-validation → Evidence-first | Eliminates hallucination path |
| Safety | Passive → Active classification | Prevents unsafe usage |
| Chunking | Semantic → Structure-aware | Preserves study context |
| Rewriting | Silent → Confidence-scored | User sees interpretation |
| Tables | Text summary → Structured | Maintains numerical precision |

---

## 4. Shared Canonical Corpus

### 4.1 Design Principle

**One paper, one representation, multiple views**

Instead of each agent maintaining separate copies, store papers once with multi-label tagging.

### 4.2 Storage Structure

```python
# Single Qdrant collection
COLLECTION_NAME = "plastic_surgery_canonical_corpus"

# Each chunk has subspecialty tags
chunk_metadata = {
    "chunk_id": "uuid-123",
    "canonical_paper_id": "pmid-12345678",  # Deduplication key

    # Multi-label subspecialty tagging
    "subspecialty_tags": ["breast", "reconstructive", "microsurgery"],
    "primary_subspecialty": "breast",

    # Study structure (in every chunk)
    "study_structure": {
        "design": "retrospective_cohort",
        "n": 250,
        "population": "Women post-mastectomy",
        "intervention": "DIEP flap",
        "limitations": ["Single-center", "Retrospective"]
    },

    # Evidence quality
    "evidence_grade": "Level III",

    # Content
    "section_name": "Results",
    "content": "...",  # With study context injected

    # For structured tables
    "chunk_type": "table_cell",  # or "text"
    "table_id": "table-456",
    "structured_data": {...}  # If table cell
}
```

### 4.3 Agent Filtering

```python
class BreastSurgeryAgent:
    """
    Agent is a FILTER VIEW over shared corpus
    """
    def __init__(self, shared_corpus):
        self.corpus = shared_corpus
        self.filter = {
            "must": [
                {"key": "subspecialty_tags", "match": {"any": ["breast"]}}
            ]
        }

    def retrieve(self, query, top_k=10):
        """
        Query shared corpus with filter
        """
        return self.corpus.search(
            query=query,
            filter=self.filter,
            limit=top_k
        )
```

### 4.4 Benefits

**Storage:**
```
Before: 10 agents × 5,000 papers × 50 chunks = 2.5M chunks
After:  1 corpus × 10,000 papers × 50 chunks = 500K chunks
Savings: 80% reduction
```

**Consistency:**
- All agents see same version of paper
- Updates propagate automatically
- No divergence

**Cost:**
```
Before: 10× embedding computation
After:  1× embedding computation
Savings: 90% embedding cost
```

---

## 5. Clinical Safety Layer

### 5.1 Query Classification

Every query is classified before processing:

```python
class SafetyClassifier:
    def classify_query(self, query):
        """
        Three-tier classification
        """
        # Tier 1: Literature Query (Safe)
        if self.is_general_literature_query(query):
            return {
                "tier": "literature",
                "mode": "full_system",
                "disclaimer": "standard"
            }

        # Tier 2: Patient-Specific (Restricted)
        if self.is_patient_specific(query):
            return {
                "tier": "patient_specific",
                "mode": "evidence_only",
                "disclaimer": "strong",
                "warning": "Cannot provide patient-specific recommendations"
            }

        # Tier 3: Emergency/High-Risk (Refuse)
        if self.is_high_risk(query):
            return {
                "tier": "emergency",
                "mode": "refuse",
                "message": "This system cannot provide emergency or high-risk clinical guidance."
            }

    def is_patient_specific(self, query):
        """
        Detect patient-specific context
        """
        patterns = [
            r"my patient",
            r"this patient",
            r"should I (do|use|perform|give)",
            r"what dose",
            r"how much (medication|drug)",
            r"patient.*age.*weight"  # Patient demographics
        ]

        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        # Also check with NER
        entities = extract_entities(query)
        if "patient_mention" in entities:
            return True

        return False
```

### 5.2 Response Modification

```python
def generate_safe_response(query, classification, evidence):
    """
    Modify response based on safety classification
    """
    if classification['tier'] == 'patient_specific':
        return f"""
⚠️ Patient-Specific Query Detected

Your query appears to ask about a specific patient or treatment decision.

I can summarize published evidence on this topic, but cannot make
patient-specific recommendations.

Published Evidence Summary:
{summarize_evidence(evidence)}

Important: Clinical decisions must consider:
• Complete patient history
• Individual risk factors
• Current clinical guidelines
• Multidisciplinary consultation

Please consult with qualified healthcare professionals.
"""

    elif classification['tier'] == 'emergency':
        return """
⛔ Emergency Query Detected

This system cannot provide emergency medical guidance.

If this is a medical emergency:
• Call emergency services immediately
• Seek immediate medical attention

For urgent clinical questions, consult with:
• On-call attending physician
• Emergency department
• Relevant specialist on call
"""

    else:
        # Normal evidence summary
        return generate_evidence_summary(query, evidence)
```

---

## 6. Evidence-First Generation

### 6.1 Process Flow

**Traditional (v3.0 - Risky):**
```
1. Retrieve chunks
2. Generate answer (can hallucinate)
3. Validate citations (too late)
4. Flag problems (damage done)
```

**Evidence-First (v4.0 - Safe):**
```
1. Retrieve chunks
2. Extract atomic claims from chunks
3. Map claims to query relevance
4. Generate using ONLY mapped claims
5. Attach chunk_id to each sentence
6. Reject any sentence without chunk_id
```

### 6.2 Implementation

```python
class EvidenceFirstGenerator:
    """
    Generate only from explicitly mapped evidence
    """
    def generate(self, query, retrieved_chunks):
        """
        Evidence-first generation pipeline
        """
        # Step 1: Extract claims from chunks
        evidence_claims = []
        for chunk in retrieved_chunks:
            claims = self.extract_claims(chunk)
            for claim in claims:
                evidence_claims.append({
                    "claim_text": claim,
                    "chunk_id": chunk.id,
                    "pmid": chunk.pmid,
                    "section": chunk.section,
                    "citation": chunk.citation_text
                })

        # Step 2: Map claims to query
        relevant_claims = self.map_claims_to_query(query, evidence_claims)

        # Step 3: Create constrained prompt
        prompt = self.build_evidence_constrained_prompt(query, relevant_claims)

        # Step 4: Generate with strict instructions
        response = self.llm.generate(prompt, temperature=0.1)

        # Step 5: Validate every sentence has chunk_id
        validated = self.validate_provenance(response, relevant_claims)

        return validated

    def extract_claims(self, chunk):
        """
        Extract atomic factual claims from chunk
        """
        # Use LLM to extract claims
        extraction_prompt = f"""
Extract atomic factual claims from this medical text.
Each claim should be a single, verifiable statement.

Text: {chunk.content}

Format each claim as:
- [Claim text]

Claims:
"""
        claims = self.llm.generate(extraction_prompt)
        return parse_claims(claims)

    def map_claims_to_query(self, query, evidence_claims):
        """
        Map which claims are relevant to query
        """
        query_embedding = self.embed(query)

        relevant = []
        for claim in evidence_claims:
            claim_embedding = self.embed(claim['claim_text'])
            similarity = cosine_similarity(query_embedding, claim_embedding)

            if similarity > 0.6:  # Threshold
                claim['relevance_score'] = similarity
                relevant.append(claim)

        return sorted(relevant, key=lambda x: x['relevance_score'], reverse=True)

    def build_evidence_constrained_prompt(self, query, relevant_claims):
        """
        Build prompt that constrains generation to evidence
        """
        evidence_list = "\n".join([
            f"[{i+1}] {claim['claim_text']} (Source: {claim['pmid']}, {claim['section']})"
            for i, claim in enumerate(relevant_claims)
        ])

        prompt = f"""
You are a medical research assistant. Answer the query using ONLY the evidence provided below.

CRITICAL RULES:
1. Use ONLY information from the evidence list below
2. For each statement, cite the evidence number [N]
3. Do NOT add information from your training data
4. If the evidence doesn't fully answer the query, say so explicitly
5. Every sentence must have a citation [N]

EVIDENCE:
{evidence_list}

QUERY: {query}

ANSWER (with citations after each statement):
"""
        return prompt

    def validate_provenance(self, response, relevant_claims):
        """
        Ensure every sentence has valid citation
        """
        sentences = sent_tokenize(response)

        for sentence in sentences:
            # Check if sentence has citation
            citation_match = re.search(r'\[(\d+)\]', sentence)

            if not citation_match:
                # Sentence lacks citation
                logger.warning(f"Sentence without citation: {sentence}")
                # Remove or flag sentence
                response = response.replace(sentence, "")

        return response
```

### 6.3 Example

**Input Query:** "What is the DIEP flap survival rate?"

**Step 1: Extract Claims**
```
From Chunk 1 (PMID 12345):
- Claim: "DIEP flap survival rate was 97.3% in 250 consecutive cases"
- Claim: "No cases of complete flap loss"

From Chunk 2 (PMID 67890):
- Claim: "Five-year flap survival 95.6% (95% CI: 92.1-98.1)"
```

**Step 2: Map to Query**
```
Relevance scores:
- Claim 1: 0.95 ✓
- Claim 2: 0.93 ✓
- Claim 3: 0.91 ✓
```

**Step 3: Generate**
```
Prompt:
Use ONLY these claims:
[1] DIEP flap survival rate was 97.3% in 250 consecutive cases (PMID 12345)
[2] Five-year flap survival 95.6% (95% CI: 92.1-98.1) (PMID 67890)

Answer: What is the DIEP flap survival rate?

Generated Response:
"DIEP flap survival rates are consistently high. Short-term survival
was 97.3% in a series of 250 cases [1]. Long-term follow-up shows
five-year survival of 95.6% (95% CI: 92.1-98.1) [2]."

Every sentence has citation ✓
```

---

## 7. Structure-Aware Chunking

### 7.1 Problem with Semantic Chunking

**Semantic chunking** splits text based on embedding similarity drops. Problem:

```
Original paper structure:
┌─────────────────────────────────────┐
│ Methods                             │
│ - Retrospective review              │
│ - n=250 patients                    │
│ - Inclusion criteria                │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ Results                             │
│ - Complication rate: 12.4%          │
│ - Statistical analysis              │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ Limitations                         │
│ - Single-center                     │
│ - Selection bias                    │
│ - Retrospective design              │
└─────────────────────────────────────┘

Semantic chunking might create:
Chunk A: [Methods + part of Results]  ← crosses boundary
Chunk B: [Rest of Results]
Chunk C: [Limitations]  ← separated from results!

If Chunk B retrieved without Chunk C → misleading
```

### 7.2 Structure-Aware Solution

```python
class StructureAwareChunker:
    """
    Preserve study structure in every chunk
    """
    def chunk_paper(self, paper):
        """
        Chunk within section boundaries, inject study context
        """
        # 1. Extract study metadata ONCE
        study_metadata = {
            "design": "retrospective_cohort",
            "n": 250,
            "population": "Women undergoing breast reconstruction",
            "intervention": "DIEP flap",
            "comparator": "Implant-based",
            "follow_up": "24 months",
            "limitations": [
                "Single-center study",
                "Retrospective design",
                "Selection bias"
            ],
            "evidence_grade": "Level III"
        }

        chunks = []

        # 2. Chunk WITHIN section boundaries
        for section in paper.sections:
            section_chunks = self.chunk_section(section)

            for chunk in section_chunks:
                # 3. Inject study context into EVERY chunk
                chunk.content = self.inject_context(
                    chunk_text=chunk.raw_content,
                    study=study_metadata,
                    section=section.name,
                    paper_title=paper.title
                )

                # 4. Store metadata
                chunk.metadata['study_structure'] = study_metadata
                chunk.metadata['section'] = section.name

                chunks.append(chunk)

        return chunks

    def inject_context(self, chunk_text, study, section, paper_title):
        """
        Every chunk gets full context header
        """
        context = f"""
Paper: {paper_title}
Study Design: {study['design']} (n={study['n']})
Population: {study['population']}
Intervention: {study['intervention']}
Evidence Level: {study['evidence_grade']}
Section: {section}

Content:
{chunk_text}

Note: Limitations include: {', '.join(study['limitations'])}
"""
        return context

    def chunk_section(self, section):
        """
        Chunk within section, respecting paragraph boundaries
        """
        paragraphs = section.text.split('\n\n')

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)

            if current_tokens + para_tokens > self.max_chunk_size:
                # Save current chunk
                chunks.append({
                    "raw_content": '\n\n'.join(current_chunk),
                    "token_count": current_tokens
                })
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add final chunk
        if current_chunk:
            chunks.append({
                "raw_content": '\n\n'.join(current_chunk),
                "token_count": current_tokens
            })

        return chunks
```

### 7.3 Result

**Every chunk now contains:**
- Full study context (design, n, population, intervention)
- Limitations mentioned upfront
- Section identity
- Evidence level

**Example chunk:**
```
Paper: Outcomes of DIEP Flap Breast Reconstruction in Obese Patients
Study Design: retrospective_cohort (n=250)
Population: Women undergoing breast reconstruction
Intervention: DIEP flap
Evidence Level: Level III
Section: Results

Content:
Complication rates varied by BMI category. In patients with BMI <25,
the complication rate was 8.0%. For BMI 25-30, the rate was 12.9%.
In obese patients (BMI >30), the complication rate was significantly
higher at 22.2% (p=0.03).

Note: Limitations include: Single-center study, Retrospective design, Selection bias
```

Now if this chunk is retrieved alone, user sees:
- It's from a retrospective study (not RCT)
- Sample size is 250
- There are important limitations
- Evidence grade is Level III

---

## 8. Query Rewriting with Confidence

### 8.1 The Problem

```
Turn 1: "What are DIEP flap complications?"
Turn 2: "What about in obese patients?"

Silent rewrite could produce:
- "What are DIEP flap complications in obese patients?" ✓ Correct
- "What are DIEP flap obese patients?" ✗ Wrong
- "What about DIEP flap outcomes in obese patients?" ✗ Semantic drift
```

User never sees interpretation, can't correct errors.

### 8.2 Solution: Confidence Scoring + Display

```python
class ConfidentQueryRewriter:
    """
    Rewrite with confidence measurement
    """
    def rewrite(self, query, conversation_context):
        """
        Rewrite and measure confidence
        """
        # 1. Generate rewrite
        rewrite_prompt = self.build_rewrite_prompt(query, conversation_context)
        rewritten = self.llm.generate(rewrite_prompt, temperature=0.1)

        # 2. Measure confidence
        confidence = self.calculate_confidence(query, rewritten, conversation_context)

        # 3. Decide whether to use rewrite
        if confidence < 0.7:
            # Low confidence - use original
            return {
                "final_query": query,
                "rewritten_query": rewritten,
                "used_rewrite": False,
                "confidence": confidence,
                "reason": "Low confidence in context interpretation"
            }
        else:
            # High confidence - use rewrite
            return {
                "final_query": rewritten,
                "rewritten_query": rewritten,
                "used_rewrite": True,
                "confidence": confidence
            }

    def calculate_confidence(self, original, rewritten, context):
        """
        Measure how confident we are in the rewrite
        """
        # Method 1: Semantic drift
        orig_embedding = self.embed(original)
        rewrite_embedding = self.embed(rewritten)
        semantic_similarity = cosine_similarity(orig_embedding, rewrite_embedding)

        # Method 2: Entity preservation
        orig_entities = extract_entities(original)
        rewrite_entities = extract_entities(rewritten)
        entity_overlap = len(orig_entities & rewrite_entities) / len(orig_entities)

        # Method 3: Context integration check
        # Does rewrite actually use context terms?
        context_terms = extract_key_terms(context)
        rewrite_terms = extract_key_terms(rewritten)
        context_usage = len(context_terms & rewrite_terms) / len(context_terms)

        # Combined confidence score
        confidence = (
            0.4 * semantic_similarity +  # Not too different
            0.3 * entity_overlap +       # Preserves entities
            0.3 * context_usage          # Actually uses context
        )

        return confidence
```

### 8.3 User Interface

```json
Response includes rewrite info:
{
  "query_handling": {
    "original": "What about in obese patients?",
    "interpreted_as": "What are DIEP flap complications in obese patients?",
    "rewrite_used": true,
    "confidence": 0.89,
    "context_from": "Turn 1: DIEP flap complications"
  },
  "answer": "..."
}
```

User sees:
```
┌──────────────────────────────────────────────────────────┐
│ Your query: "What about in obese patients?"             │
│                                                           │
│ Interpreted as:                                          │
│ "What are DIEP flap complications in obese patients?"   │
│ Confidence: 89% ✓                                        │
│ Context from: Turn 1                                     │
│                                                           │
│ [✓ Looks correct] [✗ Let me rephrase]                   │
└──────────────────────────────────────────────────────────┘
```

User can correct if interpretation is wrong.

---

## 9. Structured Table Handling

### 9.1 Why Tables Need Special Treatment

Medical papers often have critical data in tables:

```
Table 2: Complications by Patient BMI

| BMI Category | n   | Flap Loss | Partial Necrosis | Revision Rate | p-value |
|--------------|-----|-----------|------------------|---------------|---------|
| <25          | 50  | 0 (0%)    | 2 (4%)           | 3 (6%)        | ref     |
| 25-30        | 85  | 1 (1.2%)  | 6 (7.1%)         | 8 (9.4%)      | 0.18    |
| >30          | 45  | 2 (4.4%)  | 10 (22.2%)       | 12 (26.7%)    | 0.001   |

Text summary loses precision:
"Complication rates were higher in obese patients"

Lost:
- Exact percentages (22.2% vs 4%)
- Statistical significance (p=0.001)
- Sample sizes (n=45 vs n=50)
- Specific complication types
```

### 9.2 Structured Extraction

```python
class StructuredTableHandler:
    """
    Extract and store tables as queryable objects
    """
    def process_table(self, table_element, paper_metadata):
        """
        Convert table into retrievable structured format
        """
        table_obj = {
            "table_id": generate_uuid(),
            "paper_pmid": paper_metadata.pmid,
            "caption": table_element.caption,
            "headers": table_element.get_headers(),
            "rows": [],
            "cells": []
        }

        # Process each row
        for row_idx, row in enumerate(table_element.rows):
            row_obj = {
                "row_id": row_idx,
                "row_label": row[0],  # Usually first column
                "values": row
            }
            table_obj['rows'].append(row_obj)

            # Process each cell
            for col_idx, cell_value in enumerate(row):
                cell = self.create_cell_chunk(
                    table_id=table_obj['table_id'],
                    row_idx=row_idx,
                    col_idx=col_idx,
                    row_label=row[0],
                    column_name=table_obj['headers'][col_idx],
                    value=cell_value,
                    caption=table_obj['caption'],
                    paper_metadata=paper_metadata
                )
                table_obj['cells'].append(cell)

        return table_obj

    def create_cell_chunk(self, table_id, row_idx, col_idx, row_label,
                         column_name, value, caption, paper_metadata):
        """
        Create a retrievable chunk for each table cell
        """
        # Make cell searchable with natural language
        searchable_text = f"""
Table from: {paper_metadata.title}
Table caption: {caption}
Row: {row_label}
Column: {column_name}
Value: {value}

Full context: In the study on {paper_metadata.title},
for the {row_label} group, the {column_name} was {value}.
"""

        return {
            "chunk_id": f"table_{table_id}_r{row_idx}_c{col_idx}",
            "chunk_type": "table_cell",
            "content": searchable_text,  # For embedding

            # Structured data preserved
            "structured_data": {
                "table_id": table_id,
                "row": row_idx,
                "column": col_idx,
                "row_label": row_label,
                "column_name": column_name,
                "raw_value": value,
                "value_type": self.infer_type(value),  # number, percentage, text
                "caption": caption
            },

            # Standard metadata
            "paper_pmid": paper_metadata.pmid,
            "section": "Tables",
            # ... other metadata
        }

    def infer_type(self, value):
        """
        Infer data type of cell value
        """
        # Try to parse as number
        try:
            if '%' in str(value):
                return "percentage"
            float(str(value).replace(',', ''))
            return "number"
        except:
            pass

        # Check for statistical notation
        if 'p' in str(value).lower() and '=' in str(value):
            return "p_value"

        if 'CI' in str(value) or 'confidence interval' in str(value).lower():
            return "confidence_interval"

        return "text"
```

### 9.3 Retrieval and Usage

```python
# Query: "What is the revision rate in obese patients?"

# Retrieves table cell chunk:
{
  "content": "...(natural language as above)...",
  "structured_data": {
    "row_label": "BMI >30",
    "column_name": "Revision Rate",
    "raw_value": "12 (26.7%)",
    "value_type": "percentage"
  }
}

# Generation can now say:
"In obese patients (BMI >30), the revision rate was 26.7% (12/45 patients),
which was significantly higher than normal BMI patients (p=0.001) [Table 2, PMID 12345]."

# Maintains exact numbers from table ✓
```

---

## 10. Cost and Latency Budgets

### 10.1 Why Budgets Matter

Multi-agent system with reranking can be expensive and slow:

```
Without budgets:
1. Query rewrite: 0.3s, $0.01
2. Route to 3 agents
3. Each agent:
   - Retrieve 50 candidates: 0.5s
   - Rerank: 0.8s, $0.05
   - Generate answer: 2.0s, $0.30
4. Coordinator synthesis: 2.0s, $0.40

Total: ~10s, $1.50 per query

At 10K queries/month: $15,000/month ❌
```

### 10.2 Budget Enforcement

```python
class CostAwareOrchestrator:
    """
    Enforce cost and latency budgets
    """
    BUDGETS = {
        "simple_query": {
            "max_latency": 3.0,  # seconds
            "max_cost": 0.50,    # dollars
            "max_agents": 1,
            "enable_rerank": False
        },
        "standard_query": {
            "max_latency": 5.0,
            "max_cost": 0.90,
            "max_agents": 2,
            "enable_rerank": True
        },
        "complex_query": {
            "max_latency": 8.0,
            "max_cost": 1.50,
            "max_agents": 3,
            "enable_rerank": True
        }
    }

    def process_query(self, query, user_tier="standard"):
        """
        Process query within budget
        """
        budget = self.BUDGETS[user_tier]

        start_time = time.time()
        cost_tracker = CostTracker()

        # 1. Classification (required)
        safety_class = self.safety_classifier.classify(query)
        cost_tracker.add("classification", 0.01)

        # 2. Routing
        routing = self.router.route(query)

        # Enforce max agents
        if len(routing['agents']) > budget['max_agents']:
            routing['agents'] = routing['agents'][:budget['max_agents']]

        # 3. Retrieval (parallel)
        retrieval_results = self.retrieve_parallel(query, routing['agents'])
        cost_tracker.add("retrieval", 0.10 * len(routing['agents']))

        # 4. Reranking (conditional)
        if budget['enable_rerank'] and cost_tracker.total < budget['max_cost'] * 0.7:
            retrieval_results = self.rerank(query, retrieval_results)
            cost_tracker.add("rerank", 0.05 * len(routing['agents']))

        # 5. Generation
        # Check remaining budget
        remaining_cost = budget['max_cost'] - cost_tracker.total
        if remaining_cost < 0.20:
            # Skip synthesis, return single agent
            response = self.generate_single_agent(query, retrieval_results)
            cost_tracker.add("generation", 0.15)
        else:
            # Full synthesis
            response = self.generate_with_synthesis(query, retrieval_results)
            cost_tracker.add("generation", 0.40)

        # 6. Check latency
        elapsed = time.time() - start_time
        if elapsed > budget['max_latency']:
            logger.warning(f"Query exceeded latency budget: {elapsed}s > {budget['max_latency']}s")

        # 7. Check cost
        if cost_tracker.total > budget['max_cost']:
            logger.warning(f"Query exceeded cost budget: ${cost_tracker.total} > ${budget['max_cost']}")

        response['cost'] = cost_tracker.total
        response['latency'] = elapsed
        response['budget_used'] = {
            "cost_pct": cost_tracker.total / budget['max_cost'] * 100,
            "latency_pct": elapsed / budget['max_latency'] * 100
        }

        return response
```

### 10.3 Cost-Aware Routing

```python
class CostAwareRouter:
    """
    Route based on query complexity and budget
    """
    def route(self, query, budget):
        """
        Smart routing based on query complexity
        """
        # Estimate query complexity
        complexity = self.estimate_complexity(query)

        if complexity == "simple":
            # Single most relevant agent
            agent = self.get_primary_agent(query)
            return {
                "strategy": "single_agent",
                "agents": [agent],
                "estimated_cost": 0.40
            }

        elif complexity == "moderate":
            # 1-2 agents
            agents = self.get_top_agents(query, n=2)
            return {
                "strategy": "multi_agent",
                "agents": agents,
                "estimated_cost": 0.80
            }

        else:  # complex
            # Full multi-agent if budget allows
            if budget['max_cost'] >= 1.20:
                agents = self.get_top_agents(query, n=3)
                return {
                    "strategy": "full_synthesis",
                    "agents": agents,
                    "estimated_cost": 1.40
                }
            else:
                # Budget constrained - limit agents
                agents = self.get_top_agents(query, n=2)
                return {
                    "strategy": "multi_agent",
                    "agents": agents,
                    "estimated_cost": 0.80
                }
```

### 10.4 Latency Targets

```python
LATENCY_TARGETS = {
    "safety_classification": 0.3,   # 300ms
    "query_rewrite": 0.3,            # 300ms
    "routing": 0.2,                  # 200ms
    "retrieval": 0.7,                # 700ms per agent
    "reranking": 0.5,                # 500ms
    "generation": 2.5,               # 2.5s
    "synthesis": 2.0,                # 2.0s

    "total_p50": 3.0,                # 50th percentile
    "total_p95": 5.0,                # 95th percentile
    "total_p99": 8.0                 # 99th percentile
}
```

---

## 11. Deduplication Strategy

### 11.1 The Problem

Same paper can appear through:
- Multiple keyword searches
- Different subspecialty agents
- PubMed updates
- Preprint → final publication
- Multiple identifiers (PMID, PMCID, DOI)

### 11.2 Canonical ID Assignment

```python
class PaperDeduplicator:
    """
    Assign canonical IDs to papers
    """
    def get_canonical_id(self, paper):
        """
        Determine unique identifier for paper
        """
        # Priority 1: PMID (most stable)
        if paper.pmid:
            return f"pmid-{paper.pmid}"

        # Priority 2: DOI
        if paper.doi:
            normalized_doi = self.normalize_doi(paper.doi)
            return f"doi-{normalized_doi}"

        # Priority 3: PMCID
        if paper.pmcid:
            return f"pmcid-{paper.pmcid}"

        # Priority 4: Content hash
        content_hash = self.compute_content_hash(
            paper.title,
            paper.authors,
            paper.year
        )
        return f"hash-{content_hash}"

    def is_duplicate(self, paper, existing_corpus):
        """
        Check if paper already exists
        """
        canonical_id = self.get_canonical_id(paper)

        # Check if ID exists
        if canonical_id in existing_corpus:
            return True

        # Fuzzy matching for edge cases
        fuzzy_match = self.fuzzy_match(paper, existing_corpus)
        if fuzzy_match:
            return True

        return False

    def fuzzy_match(self, paper, existing_corpus):
        """
        Fuzzy matching based on title + authors + year
        """
        title_normalized = self.normalize_title(paper.title)

        for existing_paper in existing_corpus:
            existing_title = self.normalize_title(existing_paper.title)

            # Title similarity
            title_sim = self.string_similarity(title_normalized, existing_title)
            if title_sim < 0.9:
                continue

            # Author overlap
            author_overlap = len(set(paper.authors) & set(existing_paper.authors))
            if author_overlap < len(paper.authors) * 0.7:
                continue

            # Year match
            if abs(paper.year - existing_paper.year) > 1:
                continue

            # Likely duplicate
            return existing_paper.canonical_id

        return None
```

### 11.3 Merge Strategy

```python
def merge_paper_metadata(existing, new):
    """
    When duplicate detected, merge metadata
    """
    # Keep most complete identifiers
    if not existing.pmid and new.pmid:
        existing.pmid = new.pmid

    if not existing.pmcid and new.pmcid:
        existing.pmcid = new.pmcid

    # Merge subspecialty tags
    existing.subspecialty_tags = list(set(
        existing.subspecialty_tags + new.subspecialty_tags
    ))

    # Update citation count (keep higher)
    existing.citation_count = max(
        existing.citation_count,
        new.citation_count
    )

    # Update ingestion date
    existing.last_updated = datetime.now()

    return existing
```

---

## 12. Operational Safeguards

### 12.1 Failure Modes

```python
class OperationalFailureHandler:
    """
    Handle component failures gracefully
    """
    def process_query_with_fallbacks(self, query):
        """
        Execute with fallbacks for each component
        """
        try:
            # Primary: Full system
            return self.full_pipeline(query)

        except RetrievalFailure as e:
            # Fallback 1: Use cached results
            logger.error(f"Retrieval failed: {e}")
            cached_results = self.cache.get_similar_queries(query)
            if cached_results:
                return self.generate_from_cache(query, cached_results)

            # Fallback 2: Graceful degradation
            return {
                "status": "degraded",
                "message": "Retrieval system temporarily unavailable. Please try again shortly.",
                "error_code": "RETRIEVAL_FAILURE"
            }

        except LLMFailure as e:
            # Fallback: Return raw evidence without synthesis
            logger.error(f"LLM failed: {e}")
            return {
                "status": "degraded",
                "message": "Synthesis unavailable. Showing raw evidence.",
                "evidence": self.format_raw_evidence(retrieved_chunks)
            }

        except VectorStoreFailure as e:
            # Critical failure
            logger.critical(f"Vector store failed: {e}")
            return {
                "status": "error",
                "message": "System temporarily unavailable. Please contact support.",
                "error_code": "SYSTEM_FAILURE"
            }
```

### 12.2 Rate Limiting

```python
class RateLimiter:
    """
    Protect system from overload
    """
    LIMITS = {
        "free_tier": {
            "queries_per_minute": 2,
            "queries_per_day": 50
        },
        "professional": {
            "queries_per_minute": 10,
            "queries_per_day": 500
        },
        "institution": {
            "queries_per_minute": 50,
            "queries_per_day": 5000
        }
    }

    def check_rate_limit(self, user_id, tier):
        """
        Check if user is within rate limits
        """
        limit = self.LIMITS[tier]

        # Check per-minute limit
        recent_queries = self.get_queries_last_minute(user_id)
        if recent_queries >= limit['queries_per_minute']:
            raise RateLimitExceeded(
                "Too many queries per minute. Please wait before trying again."
            )

        # Check daily limit
        today_queries = self.get_queries_today(user_id)
        if today_queries >= limit['queries_per_day']:
            raise RateLimitExceeded(
                "Daily query limit reached. Limit resets at midnight UTC."
            )
```

### 12.3 Monitoring

```python
class SystemMonitor:
    """
    Monitor system health and performance
    """
    def __init__(self):
        self.metrics = PrometheusMetrics()

    def record_query(self, query_metadata):
        """
        Record query metrics
        """
        self.metrics.queries_total.inc()
        self.metrics.query_latency.observe(query_metadata['latency'])
        self.metrics.query_cost.observe(query_metadata['cost'])

        # Agent usage
        for agent in query_metadata['agents_used']:
            self.metrics.agent_queries.labels(agent=agent).inc()

        # Cache hit rate
        if query_metadata['cache_hit']:
            self.metrics.cache_hits.inc()
        else:
            self.metrics.cache_misses.inc()

        # Safety classifications
        self.metrics.safety_classifications.labels(
            tier=query_metadata['safety_tier']
        ).inc()

    def get_health_status(self):
        """
        Check system health
        """
        health = {
            "vector_store": self.check_vector_store(),
            "llm_api": self.check_llm_api(),
            "redis": self.check_redis(),
            "overall": "healthy"
        }

        # Determine overall status
        if any(v == "down" for v in health.values()):
            health["overall"] = "down"
        elif any(v == "degraded" for v in health.values()):
            health["overall"] = "degraded"

        return health
```

---

## 13. Security and Compliance

### 13.1 Authentication and Authorization

```python
class AuthenticationLayer:
    """
    User authentication and authorization
    """
    def authenticate_user(self, token):
        """
        Verify user token
        """
        # Verify JWT token
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            user_id = payload['user_id']
            tier = payload['tier']
            credentials = payload['credentials']

            return {
                "user_id": user_id,
                "tier": tier,
                "credentials": credentials,
                "authenticated": True
            }
        except jwt.InvalidTokenError:
            return {"authenticated": False}

    def authorize_query(self, user, query_type):
        """
        Check if user authorized for query type
        """
        # Verify professional credentials for clinical queries
        if query_type == "clinical" and not user['credentials']:
            raise Unauthorized(
                "Clinical queries require verified professional credentials"
            )

        # Check tier limits
        if user['tier'] == 'free' and query_type == 'multi_agent':
            raise Unauthorized(
                "Multi-agent queries require professional tier or higher"
            )
```

### 13.2 PHI Detection and Scrubbing

```python
class PHIDetector:
    """
    Detect and scrub Protected Health Information
    """
    PHI_PATTERNS = {
        "names": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
        "dates": r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        "mrn": r'\bMRN:?\s*\d+\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b'
    }

    def detect_phi(self, query):
        """
        Check if query contains PHI
        """
        for phi_type, pattern in self.PHI_PATTERNS.items():
            if re.search(pattern, query):
                return {
                    "contains_phi": True,
                    "phi_type": phi_type,
                    "action": "scrub"
                }

        # Use NER for additional detection
        entities = extract_entities(query)
        if "PERSON" in entities or "DATE" in entities:
            return {
                "contains_phi": True,
                "phi_type": "entity",
                "action": "scrub"
            }

        return {"contains_phi": False}

    def scrub_phi(self, query):
        """
        Remove PHI from query
        """
        scrubbed = query

        for phi_type, pattern in self.PHI_PATTERNS.items():
            scrubbed = re.sub(pattern, f"[{phi_type.upper()}_REDACTED]", scrubbed)

        return scrubbed
```

### 13.3 Audit Logging

```python
class AuditLogger:
    """
    Comprehensive audit trail
    """
    def log_query(self, user, query, response, metadata):
        """
        Log every query for audit
        """
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user['user_id'],
            "user_tier": user['tier'],
            "user_credentials": user['credentials'],

            # Query
            "query": query,
            "query_hash": hashlib.sha256(query.encode()).hexdigest(),
            "safety_classification": metadata['safety_tier'],

            # Response
            "response_hash": hashlib.sha256(response.encode()).hexdigest(),
            "papers_cited": metadata['cited_pmids'],
            "agents_consulted": metadata['agents'],

            # Metrics
            "latency": metadata['latency'],
            "cost": metadata['cost'],

            # Provenance
            "session_id": metadata['session_id'],
            "turn_number": metadata['turn_number']
        }

        # Store in secure audit log
        self.audit_db.insert(audit_entry)

        # Also log to immutable append-only log for compliance
        self.immutable_log.append(audit_entry)
```

---

## 14. Evidence Grading System

### 14.1 Systematic Evidence Grading

```python
class EvidenceGrader:
    """
    Systematic evidence quality assessment
    """
    def grade_paper(self, paper):
        """
        Assign evidence level using CEBM or similar hierarchy
        """
        # Extract study characteristics
        design = paper.study_structure['design']
        sample_size = paper.study_structure['n']
        is_rct = paper.study_structure.get('is_rct', False)
        is_meta_analysis = 'meta-analysis' in paper.publication_types
        is_systematic_review = 'systematic review' in paper.publication_types

        # Grade using Oxford CEBM Levels of Evidence

        # Level Ia: Systematic review of RCTs
        if is_systematic_review and is_rct:
            return "Level Ia"

        # Level Ib: Individual RCT with narrow CI
        if is_rct and sample_size > 100:
            return "Level Ib"

        # Level IIa: Systematic review of cohort studies
        if is_systematic_review and design == 'cohort':
            return "Level IIa"

        # Level IIb: Individual cohort study
        if design == 'prospective_cohort':
            return "Level IIb"

        # Level III: Case-control or retrospective cohort
        if design in ['retrospective_cohort', 'case_control']:
            return "Level III"

        # Level IV: Case series
        if design == 'case_series':
            return "Level IV"

        # Level V: Expert opinion
        if 'editorial' in paper.publication_types:
            return "Level V"

        # Default
        return "Level IV"

    def assess_study_quality(self, paper):
        """
        Assess methodological quality
        """
        quality_score = 0
        flags = []

        # Sample size
        if paper.study_structure['n'] > 100:
            quality_score += 2
        elif paper.study_structure['n'] > 50:
            quality_score += 1
        else:
            flags.append("Small sample size")

        # Study design
        if paper.study_structure['design'] == 'RCT':
            quality_score += 3
        elif paper.study_structure['design'] == 'prospective_cohort':
            quality_score += 2
        elif paper.study_structure['design'] == 'retrospective_cohort':
            quality_score += 1
            flags.append("Retrospective design")

        # Limitations
        limitations = paper.study_structure.get('limitations', [])
        if 'single-center' in ' '.join(limitations).lower():
            flags.append("Single-center study")
        if 'selection bias' in ' '.join(limitations).lower():
            flags.append("Selection bias")

        # Journal quality
        if paper.journal_impact_factor > 5.0:
            quality_score += 1

        return {
            "quality_score": quality_score,
            "max_score": 10,
            "flags": flags
        }
```

### 14.2 Evidence Aggregation

```python
def assess_evidence_strength(papers_cited):
    """
    Assess overall evidence strength across multiple papers
    """
    # Count by evidence level
    level_counts = Counter([p.evidence_level for p in papers_cited])

    # Determine overall strength
    if level_counts.get('Level Ia', 0) > 0 or level_counts.get('Level Ib', 0) >= 2:
        strength = "Strong"
        description = "Multiple high-quality RCTs or systematic reviews"

    elif level_counts.get('Level II', 0) >= 3:
        strength = "Moderate"
        description = "Multiple cohort studies with consistent findings"

    elif level_counts.get('Level III', 0) >= 2:
        strength = "Limited"
        description = "Retrospective studies or case-control studies"

    else:
        strength = "Weak"
        description = "Case series or expert opinion only"

    return {
        "strength": strength,
        "description": description,
        "evidence_distribution": dict(level_counts),
        "total_papers": len(papers_cited)
    }
```

---

## 15. Implementation Priorities

### 15.1 Phase 1: Critical Foundation (Weeks 1-4)

**Priority: Get core architecture right before building features**

```
Week 1-2: Shared Canonical Corpus
├── Design canonical ID system
├── Implement deduplication
├── Build single Qdrant collection
├── Implement multi-label tagging
└── Test with 1,000 sample papers

Week 3: Evidence-First Generation
├── Implement claim extraction
├── Build claim-to-query mapping
├── Constrained generation prompts
├── Sentence-level provenance
└── Test hallucination prevention

Week 4: Clinical Safety Layer
├── Query classification
├── Safety boundary detection
├── Response mode switching
├── Disclaimer management
└── Test with safety-critical queries
```

### 15.2 Phase 2: Production Features (Weeks 5-8)

```
Week 5: Structure-Aware Chunking
├── Study metadata extraction
├── Section-aware chunking
├── Context injection
└── Test context preservation

Week 6: Structured Tables
├── Table extraction
├── Cell-level indexing
├── Structured retrieval
└── Test numerical precision

Week 7: Query Rewriting with Confidence
├── Confidence scoring
├── User-visible interpretation
├── Correction mechanism
└── Test semantic drift prevention

Week 8: Cost & Latency Budgets
├── Budget enforcement
├── Cost-aware routing
├── Graceful degradation
└── Test performance targets
```

### 15.3 Phase 3: Operational Readiness (Weeks 9-12)

```
Week 9: Security & Compliance
├── Authentication/authorization
├── PHI detection
├── Audit logging
└── Test compliance

Week 10: Failure Handling
├── Graceful degradation
├── Fallback strategies
├── Rate limiting
└── Test fault tolerance

Week 11: Evidence Grading
├── Systematic grading
├── Quality assessment
├── Strength aggregation
└── Test accuracy

Week 12: Integration & Testing
├── End-to-end testing
├── Load testing
├── Expert review
└── Production deployment
```

### 15.4 MVP Scope

**Minimum viable product includes:**

✅ Shared canonical corpus (no separate agent stores)
✅ Evidence-first generation (no post-hoc validation)
✅ Clinical safety layer (active classification)
✅ Structure-aware chunking (with context)
✅ Query rewrite confidence (user-visible)
✅ Basic cost/latency budgets
✅ Authentication
✅ Audit logging

**Can be deferred to v2:**
- Structured table extraction (use text summaries initially)
- Multi-tier sources beyond PubMed
- Advanced failure recovery
- Fine-tuned medical models

### 15.5 Success Criteria

**Technical:**
- Hallucination rate < 5% (measured on test set)
- Query rewrite accuracy > 90%
- P95 latency < 5s for single-agent queries
- Cost per query < $1.00 average
- 99.9% uptime

**Medical:**
- Expert review: 95% factual accuracy
- Clinical safety: 0% unsafe recommendations
- Evidence grading: 90% agreement with expert assessment
- Citation accuracy: 98% verifiable

**Operational:**
- Handle 1,000 queries/day initially
- Scale to 10,000 queries/day by month 3
- Cost per query decreasing over time (through caching)

---

## Conclusion

This v4.0 design addresses critical production concerns identified in expert review:

### Key Changes Summary

1. **Shared Canonical Corpus** → 80% storage reduction, consistency
2. **Evidence-First Generation** → Eliminates primary hallucination path
3. **Clinical Safety Layer** → Prevents unsafe usage
4. **Structure-Aware Chunking** → Preserves study context integrity
5. **Query Rewrite Confidence** → Prevents silent semantic drift
6. **Structured Tables** → Maintains numerical precision
7. **Cost/Latency Budgets** → Production-viable economics
8. **Comprehensive Security** → Compliance-ready

### Implementation Philosophy

**Build foundation first:**
- Get architecture right before adding features
- Prioritize safety over speed
- Measure everything
- Fail gracefully

**Production-ready means:**
- Handles failures without breaking
- Operates within cost budgets
- Maintains audit trails
- Prevents unsafe usage
- Degrades gracefully

This is still a **planning document** - no implementation has begun. Ready to proceed to implementation phase once design is validated.

---

**Document Version:** 4.0
**Status:** Production Architecture Specification
**Next Step:** Expert review and validation before implementation

