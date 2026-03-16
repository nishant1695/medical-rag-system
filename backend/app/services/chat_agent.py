"""
Medical Agentic Chat — Multi-Specialist Architecture with Query Decomposition + HyDE

Flow:
  1. Classify which subspecialties the query touches (keyword-based, no LLM call)
  2. Detect complexity — clinical vignettes / multi-aspect queries trigger decomposition
  3. [Complex only] Decompose into 2-4 sub-questions; generate HyDE paragraph per sub-question
  4. Run parallel subspecialty-filtered retrievals (one per sub-query × matched specialty)
  5. Deduplicate retrieved chunks by chunk_id (keep highest-score copy)
  6. Single synthesizer LLM produces a structured answer
  7. Stream SSE events throughout

For short / single-specialty queries the flow degenerates to simple RAG (one retrieval,
one LLM call) with no extra overhead.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import string
import uuid
from typing import AsyncGenerator, List, Optional

from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import KnowledgeBase, ChatMessage
from app.schemas import ChatRequest
from app.services.llm import get_llm_provider
from app.services.llm.base import LLMMessage
from app.services.medical_safety_classifier import safety_classifier
from app.services.knowledge_graph import get_knowledge_graph
from app.services.rag_service import get_rag_service
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_ITERATIONS   = 3
SSE_HEARTBEAT_INTERVAL = 15  # seconds
_CID_CHARS = string.ascii_lowercase + string.digits

# Chunks retrieved per specialist (total context = N_specialists × SPECIALIST_TOP_K)
SPECIALIST_TOP_K = 6

# Conversation history management
VERBATIM_TURNS  = 8   # most-recent turns sent verbatim to the LLM
SUMMARY_MAX_AGE = 40  # older turns beyond this are dropped from summary

# ─── Conversation history management ─────────────────────────────────────────

def _build_history_messages(
    history: list[dict],
) -> tuple[list[LLMMessage], str]:
    """
    Convert raw history into LLM messages + an optional summary prefix.

    Returns
    -------
    messages : list[LLMMessage]
        The verbatim recent turns to include in the messages array.
    summary : str
        A plain-text summary of older turns (empty string if history is short).
        Caller should append this to the system prompt so the LLM has full
        context without the messages array getting unwieldy.

    Strategy
    --------
    - If history ≤ VERBATIM_TURNS: send everything verbatim, no summary.
    - If history > VERBATIM_TURNS: keep last VERBATIM_TURNS turns verbatim;
      condense older turns into a one-paragraph summary (no extra LLM call).
    """
    if len(history) <= VERBATIM_TURNS:
        messages = [
            LLMMessage(
                role="user" if m["role"] == "user" else "assistant",
                content=m["content"],
            )
            for m in history
        ]
        return messages, ""

    verbatim = history[-VERBATIM_TURNS:]
    older = history[:-VERBATIM_TURNS][-SUMMARY_MAX_AGE:]  # cap at SUMMARY_MAX_AGE

    # Build a compact summary — first ~120 chars of each turn
    lines = []
    for m in older:
        role_label = "User" if m["role"] == "user" else "Assistant"
        snippet = m["content"].strip()[:120].replace("\n", " ")
        if len(m["content"]) > 120:
            snippet += "…"
        lines.append(f"  {role_label}: {snippet}")

    summary = (
        f"Earlier in this conversation ({len(older)} messages):\n"
        + "\n".join(lines)
    )

    messages = [
        LLMMessage(
            role="user" if m["role"] == "user" else "assistant",
            content=m["content"],
        )
        for m in verbatim
    ]
    return messages, summary


# ─── Subspecialty definitions ─────────────────────────────────────────────────

SUBSPECIALTIES: list[str] = [
    "aesthetic",
    "pediatric_plastic",
    "microsurgery",
    "craniofacial",
    "supermicrosurgery",
    "burn_surgery",
    "wound_care",
    "hand",
]

# Keywords that map a query to one or more subspecialties.
# Deliberate overlap: e.g. "replantation" is both hand and microsurgery.
SUBSPECIALTY_KEYWORDS: dict[str, list[str]] = {
    "aesthetic": [
        "aesthetic", "cosmetic", "rhinoplasty", "facelift", "rhytidectomy",
        "liposuction", "blepharoplasty", "botox", "botulinum", "filler",
        "hyaluronic acid", "abdominoplasty", "tummy tuck", "brow lift", "otoplasty",
        "breast augmentation", "augmentation mammoplasty", "gynecomastia",
        "body contouring", "fat grafting", "fat transfer", "labiaplasty",
    ],
    "pediatric_plastic": [
        "pediatric", "paediatric", "child", "infant", "neonatal", "congenital",
        "cleft lip", "cleft palate", "cleft", "hemangioma", "haemangioma",
        "vascular malformation", "birthmark", "syndactyly", "polydactyly",
        "auricular reconstruction", "microtia", "ear reconstruction",
        "pediatric burn", "paediatric burn", "craniosynostosis",
    ],
    "microsurgery": [
        "free flap", "perforator flap", "perforator", "anastomosis", "microsurg",
        "alt flap", "anterolateral thigh", "gracilis", "latissimus dorsi",
        "diep", "tram flap", "lat flap", "rectus abdominis",
        "flap failure", "flap necrosis", "flap survival", "free tissue transfer",
        "replantation", "revascularisation", "revascularization",
        "pedicle flap", "myocutaneous", "fasciocutaneous",
    ],
    "craniofacial": [
        "craniofacial", "skull base", "orbit", "orbital", "mandible", "maxilla",
        "facial bone", "jaw reconstruction", "zygoma", "zygomatic",
        "forehead reconstruction", "midface", "le fort", "distraction osteogenesis",
        "orthognathic", "facial trauma", "orbital fracture", "panfacial",
        "frontal sinus", "calvarium", "calvarial",
    ],
    "supermicrosurgery": [
        "supermicrosurgery", "lymphedema", "lymphoedema", "lymphatic",
        "lymphovenous anastomosis", "lva", "vascularized lymph node",
        "vascularised lymph node", "vlnt", "super-thin flap",
        "perforator-to-perforator", "superficial lymphatic", "lymphatic mapping",
        "indocyanine green lymphography", "icg lymphography",
    ],
    "burn_surgery": [
        "burn", "thermal injury", "eschar", "escharotomy", "tbsa",
        "total body surface area", "inhalation injury", "smoke inhalation",
        "carbon monoxide poisoning", "burn wound", "burn patient",
        "fluid resuscitation", "parkland formula", "acute burn",
        "burn debridement", "burn reconstruction", "hypertrophic scar",
    ],
    "wound_care": [
        "wound healing", "chronic wound", "wound care", "wound bed preparation",
        "pressure ulcer", "pressure injury", "venous ulcer", "diabetic foot",
        "diabetic ulcer", "dehiscence", "wound dehiscence",
        "negative pressure wound therapy", "npwt", "vac therapy",
        "wound vac", "skin substitute", "biofilm", "granulation tissue",
        "wound closure", "secondary intention", "ulceration",
    ],
    "hand": [
        "hand surgery", "hand", "finger", "thumb", "wrist", "tendon repair",
        "tendon transfer", "carpal tunnel", "dupuytren", "trigger finger",
        "digital replantation", "metacarpal", "phalanx", "phalangeal",
        "nerve repair", "nerve graft", "extensor tendon", "flexor tendon",
        "hand trauma", "hand reconstruction", "mallet finger", "boutonniere",
    ],
}

# Display names for the UI
SUBSPECIALTY_LABELS: dict[str, str] = {
    "aesthetic": "Aesthetic Surgery",
    "pediatric_plastic": "Pediatric Plastic Surgery",
    "microsurgery": "Microsurgery",
    "craniofacial": "Craniofacial Surgery",
    "supermicrosurgery": "Supermicrosurgery",
    "burn_surgery": "Burn Surgery",
    "wound_care": "Wound Care",
    "hand": "Hand Surgery",
}

# ─── System prompts ───────────────────────────────────────────────────────────

SINGLE_SPECIALIST_PROMPT = """\
You are a clinical decision support assistant helping qualified healthcare professionals
understand evidence from peer-reviewed medical literature.

## Citation Format
After every factual claim add an inline citation: claim text[cid]
Example: "DIEP flap has a 3-5% flap failure rate[a3x9]."

## Evidence Grading
Mention the evidence level where known:
"A 2022 RCT (Level I) found..." or "A retrospective cohort (Level III) reported..."

## Language
Use "studies suggest", "evidence indicates", "the literature supports" rather than
overly hedged or overly definitive language.

## Footer
End every response with:
"*This is a synthesis of published evidence. Clinical decisions must integrate
individual patient factors and professional clinical judgment.*"
"""

MULTI_SPECIALIST_PROMPT = """\
You are a senior clinical consultant synthesising evidence from multiple subspecialty
literature pools to answer a complex medical question.

## Your task
You will receive search results from {n_specs} subspecialty literature pools.
Structure your answer as follows:

### Evidence by Subspecialty
For each subspecialty that has relevant findings, summarise what that literature says.
Cite every claim with [cid] inline.

### Areas of Agreement
List the key points where subspecialties converge.

### Areas of Disagreement / Nuance
Note where subspecialties differ, use different techniques, or have conflicting evidence.
Explain the likely reasons (patient population, indication, technical approach).

### Final Verdict
A concise evidence-graded summary answering the original question.
Include the strongest evidence level available (e.g. "supported by Level I–II evidence").

## Rules
- Cite every factual claim with [cid].
- Do NOT invent findings — if a specialty pool has no relevant results, say so.
- Conservative language: "evidence suggests", "studies indicate".
- End with: "*This is a research summary — not a substitute for clinical judgment.*"
"""

CLINICAL_DECISION_PROMPT = """\
You are a clinical decision support assistant. Your role is to help qualified healthcare
professionals by synthesising the latest evidence from the medical literature.

## Answer Format
Structure every response with these sections:

### Clinical Summary
A 2-3 sentence direct answer to the clinical question.

### Evidence-Based Options
Ranked by evidence strength. For each option:
- **Option**: [name / intervention]
- **Evidence**: [Level I–V] — cite all supporting sources with [cid]
- **Key Data**: specific outcomes, success rates, or timelines from the retrieved literature

### Key Considerations
- Patient selection criteria from the evidence
- Contraindications or risk factors noted in the literature
- Factors that modify the recommendation

### Evidence Quality
Brief summary: how strong is the evidence base, are there guidelines, what are the gaps.

---
*This is a synthesis of published evidence for qualified healthcare professionals.
Clinical decisions must integrate individual patient factors and professional judgment.*

## Citation Rules
- Every factual claim must have an inline citation: claim text[cid]
- State evidence level: "A 2022 RCT (Level I)[abc1] found..."
- If retrieved sources do not address a sub-question, state that explicitly
- Do NOT generate information not present in the provided sources
"""

HARD_SYSTEM_SUFFIX = """
## Non-negotiable Rules
- Base every claim on the retrieved sources. Do not add unsourced information.
- Cite every factual claim with [cid].
- For life-threatening emergencies, direct to emergency services immediately.
"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _new_cid(seen: set[str]) -> str:
    while True:
        cid = "".join(random.choices(_CID_CHARS, k=4))
        if any(c.isalpha() for c in cid) and cid not in seen:
            return cid


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


async def _sse_heartbeat(
    source: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def _pump():
        try:
            async for item in source:
                await queue.put(item)
        finally:
            await queue.put(None)

    task = asyncio.create_task(_pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_INTERVAL)
                if item is None:
                    break
                yield item
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ─── Conversation-aware query rewriting ──────────────────────────────────────

# Patterns that signal a follow-up / context-dependent question
_FOLLOWUP_RE = re.compile(
    r"^(what about|how about|and what|tell me more|expand|elaborate|"
    r"can you|could you|why (is|are|does|do|was|were)|"
    r"(it|they|those|these|that|this)\b)",
    re.IGNORECASE,
)


def _contextualize_query(message: str, history: list[dict]) -> str:
    """
    Rewrite a follow-up question as a self-contained retrieval query.

    Only kicks in when the question is short or uses context-dependent language.
    Prepends the most recent topic from history so the vector search finds the
    right chunks without needing to know what "it" or "that" refers to.

    Examples
    --------
    history: "What is a DIEP flap?"  →  assistant answer about DIEP flap
    message: "What are the complications?"
    → rewritten: "What are the complications of DIEP flap breast reconstruction?"
    """
    if not history or len(history) < 2:
        return message

    word_count = len(message.split())
    is_followup = word_count <= 8 or bool(_FOLLOWUP_RE.match(message.strip()))
    if not is_followup:
        return message

    # Extract the most recent user question as context anchor
    recent_user_qs = [
        m["content"] for m in history[-6:] if m.get("role") == "user"
    ]
    if not recent_user_qs:
        return message

    # Trim to a short phrase — just enough to anchor the topic
    prior = recent_user_qs[-1][:120].rstrip("?.,")
    return f"{message} (in the context of: {prior})"


# ─── Complexity detection ─────────────────────────────────────────────────────

_CLINICAL_VIGNETTE_RE = re.compile(
    r"\b\d+[- ]?year[- ]?old\b.{0,200}\b(presents?|history of|complains?|underwent|diagnosed)\b",
    re.IGNORECASE | re.DOTALL,
)


def _is_complex_query(message: str) -> bool:
    """
    Returns True for queries that benefit from decomposition + HyDE.

    Heuristics (any one sufficient):
    - Clinical vignette: age + clinical presentation in same sentence
    - Comparative question with enough detail (>30 words)
    - Multi-question: two or more explicit question marks
    - Long query: >60 words (likely multi-aspect)
    """
    word_count = len(message.split())
    if word_count < 15:
        return False  # Very short queries don't need decomposition

    if _CLINICAL_VIGNETTE_RE.search(message):
        return True

    if re.search(
        r"\b(compare|vs\.?|versus|difference between|better than|superior to|"
        r"advantages? of|disadvantages? of)\b",
        message,
        re.IGNORECASE,
    ) and word_count > 25:
        return True

    if message.count("?") >= 2:
        return True

    return word_count > 60


# ─── LLM quick-complete helper ────────────────────────────────────────────────

async def _llm_quick_complete(
    provider,
    messages: list[LLMMessage],
    system_prompt: str = "",
    max_tokens: int = 300,
) -> str:
    """Collect an entire (short) LLM streaming response into a string."""
    result = ""
    async for chunk in provider.astream(
        messages,
        system_prompt=system_prompt,
        temperature=0.0,
        max_tokens=max_tokens,
    ):
        if chunk.type == "text":
            result += chunk.text
    return result.strip()


# ─── Query decomposition ──────────────────────────────────────────────────────

async def _decompose_query(message: str, provider) -> list[str]:
    """
    Break a complex clinical query into 2-4 focused sub-questions.

    Each sub-question is short and self-contained so it hits the embedding limit
    and returns tightly scoped results.  Falls back to [message] on any error.
    """
    prompt = (
        "Decompose the following medical question into 2-4 specific, concise "
        "sub-questions suitable for literature retrieval. Each sub-question must "
        "be self-contained (no pronouns referring to other sub-questions).\n"
        "Output ONLY a JSON array of strings, no other text.\n"
        'Example: ["What are the complication rates of X?", '
        '"How does X compare to Y in terms of outcomes?"]\n\n'
        f"Question: {message}"
    )
    try:
        raw = await _llm_quick_complete(
            provider,
            [LLMMessage(role="user", content=prompt)],
            system_prompt=(
                "You decompose medical questions for literature retrieval. "
                "Output only valid JSON arrays of strings."
            ),
            max_tokens=400,
        )
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            parts = json.loads(match.group())
            if isinstance(parts, list) and all(isinstance(s, str) for s in parts):
                cleaned = [p.strip() for p in parts if p.strip()]
                if cleaned:
                    return cleaned[:4]
    except Exception as exc:
        logger.debug(f"Query decomposition failed: {exc}")
    return [message]


# ─── HyDE (Hypothetical Document Embeddings) ──────────────────────────────────

async def _generate_hyde_query(query: str, provider) -> str:
    """
    Generate a short hypothetical abstract paragraph for better vector retrieval.

    Embedding a plausible answer paragraph instead of the raw query pulls it
    closer to the document embedding space, improving recall for complex queries.
    Falls back to the original query on error.
    """
    prompt = (
        "Write a concise 2-3 sentence medical literature abstract excerpt that would "
        "directly answer the following question. Focus on key findings, outcomes, and "
        "evidence. Do not add citations, references, or section headers.\n\n"
        f"Question: {query}"
    )
    try:
        abstract = await _llm_quick_complete(
            provider,
            [LLMMessage(role="user", content=prompt)],
            system_prompt=(
                "You write short, factual medical literature abstract paragraphs "
                "that synthesise findings relevant to a clinical question."
            ),
            max_tokens=200,
        )
        if abstract and len(abstract) > 30:
            return abstract
    except Exception as exc:
        logger.debug(f"HyDE generation failed: {exc}")
    return query


# ─── Subspecialty classifier ──────────────────────────────────────────────────

def classify_subspecialties(message: str) -> list[str]:
    """
    Keyword-based classifier — no LLM call required.

    Returns the subset of SUBSPECIALTIES relevant to the query.
    Falls back to an empty list (= broad search, no subspecialty filter) when
    the query doesn't clearly match any specialty.
    """
    msg_lower = message.lower()
    matched = [
        spec
        for spec, keywords in SUBSPECIALTY_KEYWORDS.items()
        if any(kw in msg_lower for kw in keywords)
    ]
    return matched  # empty = broad / cross-cutting query


# ─── Retrieval helpers ────────────────────────────────────────────────────────

async def _retrieve_for_specialty(
    workspace_id: int,
    query: str,
    subspecialty: Optional[str],   # None = no filter
    db: AsyncSession,
    seen_cids: set[str],
    top_k: int = SPECIALIST_TOP_K,
) -> tuple[str, list[dict], dict[str, int]]:
    """
    Run hybrid retrieval for one subspecialty (or broad if subspecialty=None).

    Returns:
        context_str  – formatted text for the LLM prompt
        sources      – list of source dicts for the SSE sources event
        evidence_summary – {level: count}
    """
    rag = get_rag_service(db, workspace_id)
    result = await rag.query(
        question=query,
        top_k=min(top_k, 10),
        subspecialty=subspecialty,
    )

    sources: list[dict] = []
    context_parts: list[str] = []
    evidence_summary: dict[str, int] = {}

    for chunk in result.chunks:
        cid = _new_cid(seen_cids)
        seen_cids.add(cid)

        meta_parts: list[str] = []
        if chunk.pmid:
            meta_parts.append(f"PMID:{chunk.pmid}")
        if chunk.evidence_level:
            meta_parts.append(f"Level {chunk.evidence_level}")
            evidence_summary[chunk.evidence_level] = (
                evidence_summary.get(chunk.evidence_level, 0) + 1
            )
        if chunk.page_no:
            meta_parts.append(f"p.{chunk.page_no}")
        meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""

        context_parts.append(f"Source [{cid}]{meta_str}:\n{chunk.content}")
        sources.append({
            "index": cid,
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "document_id": chunk.document_id,
            "page_no": chunk.page_no,
            "heading_path": chunk.heading_path,
            "score": chunk.score,
            "pmid": chunk.pmid,
            "evidence_level": chunk.evidence_level,
            "paper_url": chunk.paper_url,
            "subspecialty": chunk.subspecialty or subspecialty,
        })

    context = "\n\n---\n\n".join(context_parts)
    return context, sources, evidence_summary


async def _retrieve_raw_chunks(
    workspace_id: int,
    query: str,
    subspecialty: Optional[str],
    db: AsyncSession,
    top_k: int = SPECIALIST_TOP_K,
) -> tuple[list[RetrievedChunk], Optional[str]]:
    """
    Run hybrid retrieval for one (query, subspecialty) pair.
    Returns raw chunks without cid assignment so callers can deduplicate first.
    """
    rag = get_rag_service(db, workspace_id)
    result = await rag.query(
        question=query,
        top_k=min(top_k, 10),
        subspecialty=subspecialty,
    )
    return result.chunks, subspecialty


def _format_deduplicated_chunks(
    chunks_with_spec: list[tuple[RetrievedChunk, Optional[str]]],
    seen_cids: set[str],
) -> tuple[str, list[dict], dict[str, int]]:
    """
    Assign cids and format a deduplicated chunk list into context/sources/evidence.
    chunks_with_spec: list of (chunk, subspecialty) tuples already deduplicated.
    """
    context_parts: list[str] = []
    sources: list[dict] = []
    evidence_summary: dict[str, int] = {}

    for chunk, subspecialty in chunks_with_spec:
        cid = _new_cid(seen_cids)
        seen_cids.add(cid)

        meta_parts: list[str] = []
        if chunk.pmid:
            meta_parts.append(f"PMID:{chunk.pmid}")
        if chunk.evidence_level:
            meta_parts.append(f"Level {chunk.evidence_level}")
            evidence_summary[chunk.evidence_level] = (
                evidence_summary.get(chunk.evidence_level, 0) + 1
            )
        if chunk.page_no:
            meta_parts.append(f"p.{chunk.page_no}")
        meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""

        context_parts.append(f"Source [{cid}]{meta_str}:\n{chunk.content}")
        sources.append({
            "index": cid,
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "document_id": chunk.document_id,
            "page_no": chunk.page_no,
            "heading_path": chunk.heading_path,
            "score": chunk.score,
            "pmid": chunk.pmid,
            "evidence_level": chunk.evidence_level,
            "paper_url": chunk.paper_url,
            "subspecialty": chunk.subspecialty or subspecialty,
        })

    context = "\n\n---\n\n".join(context_parts)
    return context, sources, evidence_summary


# ─── Agent stream ─────────────────────────────────────────────────────────────

async def _agent_stream(
    workspace_id: int,
    message: str,
    history: List[dict],
    db: AsyncSession,
    system_prompt: str,
    enable_thinking: bool,
    safety_class: str,
) -> AsyncGenerator[dict, None]:
    """
    Core agent loop. Yields event dicts: {event, data}.

    Multi-specialist path (default):
      1. Classify subspecialties from query
      2. Parallel retrieval for each matched subspecialty
      3. Single synthesizer LLM call over combined contexts

    Single-specialty / broad path:
      Same as above but with one retrieval (no subspecialty filter or single match).
    """
    provider = get_llm_provider()
    seen_cids: set[str] = set()
    all_sources: list[dict] = []
    all_evidence: dict[str, int] = {}

    safety_suffix = safety_classifier.get_system_prompt_modification(safety_class)

    # ── Build conversation context ─────────────────────────────────────────────
    messages, history_summary = _build_history_messages(history)

    # ── Step 0: Rewrite follow-up questions for retrieval ─────────────────────
    retrieval_query = _contextualize_query(message, history)
    if retrieval_query != message:
        logger.debug(f"Query rewritten for retrieval: {retrieval_query!r}")

    # ── Step 1: Classify subspecialties ───────────────────────────────────────
    subspecialties = classify_subspecialties(retrieval_query)

    if subspecialties:
        spec_labels = [SUBSPECIALTY_LABELS.get(s, s) for s in subspecialties]
        yield {
            "event": "status",
            "data": {
                "step": "classifying",
                "detail": f"Identified subspecialties: {', '.join(spec_labels)}",
                "subspecialties": subspecialties,
            },
        }
    else:
        yield {
            "event": "status",
            "data": {
                "step": "classifying",
                "detail": "Broad cross-specialty query — searching all literature",
                "subspecialties": [],
            },
        }

    # ── Step 2: Complexity check → decomposition + HyDE ───────────────────────
    is_complex = _is_complex_query(message)

    if is_complex:
        yield {
            "event": "status",
            "data": {"step": "decomposing", "detail": "Decomposing complex query…"},
        }
        sub_questions = await _decompose_query(message, provider)
        logger.debug(f"Decomposed into {len(sub_questions)} sub-questions: {sub_questions}")

        yield {
            "event": "status",
            "data": {
                "step": "hyde",
                "detail": f"Generating retrieval queries for {len(sub_questions)} sub-questions…",
            },
        }
        hyde_tasks = [_generate_hyde_query(q, provider) for q in sub_questions]
        retrieval_queries = list(await asyncio.gather(*hyde_tasks))
        logger.debug(f"HyDE queries generated: {len(retrieval_queries)}")
    else:
        sub_questions = [retrieval_query]
        retrieval_queries = [retrieval_query]

    # ── Step 2b: Knowledge graph query expansion ───────────────────────────────
    try:
        kg = get_knowledge_graph(workspace_id)
        expanded_queries = []
        for rq in retrieval_queries:
            related = await kg.expand_query(rq, db)
            if related:
                expanded_queries.append(
                    f"{rq}\nRelated concepts: {', '.join(related)}"
                )
                logger.debug(f"KG expanded query with: {related}")
            else:
                expanded_queries.append(rq)
        retrieval_queries = expanded_queries
    except Exception as kg_exc:
        logger.debug(f"KG query expansion skipped: {kg_exc}")

    # ── Step 3: Parallel retrievals per (query × subspecialty) ────────────────
    is_multi = len(subspecialties) > 1

    # Build the full job list
    if not subspecialties:
        spec_list: list[Optional[str]] = [None]
        per_query_k = SPECIALIST_TOP_K * 2
    else:
        spec_list = list(subspecialties)  # type: ignore[assignment]
        per_query_k = SPECIALIST_TOP_K

    # Announce retrieval steps
    if subspecialties:
        for spec in spec_list:
            label = SUBSPECIALTY_LABELS.get(spec, spec) if spec else "All literature"
            yield {
                "event": "status",
                "data": {"step": "retrieving", "detail": f"Searching {label}…"},
            }
    else:
        yield {
            "event": "status",
            "data": {"step": "retrieving", "detail": "Searching literature…"},
        }

    raw_tasks = [
        _retrieve_raw_chunks(workspace_id, rq, spec, db, per_query_k)
        for rq in retrieval_queries
        for spec in spec_list
    ]
    raw_results: list[tuple[list[RetrievedChunk], Optional[str]]] = list(
        await asyncio.gather(*raw_tasks)
    )

    # ── Step 4: Deduplicate by chunk_id (keep highest-score copy) ─────────────
    best_by_cid: dict[str, tuple[RetrievedChunk, Optional[str]]] = {}
    for chunks, spec in raw_results:
        for chunk in chunks:
            existing = best_by_cid.get(chunk.chunk_id)
            if existing is None or chunk.score > existing[0].score:
                best_by_cid[chunk.chunk_id] = (chunk, spec)

    # Sort by score desc, then group by subspecialty
    all_deduped = sorted(best_by_cid.values(), key=lambda x: x[0].score, reverse=True)

    # Group by subspecialty for specialist context blocks
    spec_to_chunks: dict[str, list[tuple[RetrievedChunk, Optional[str]]]] = {}
    for chunk, spec in all_deduped:
        key = spec or "general"
        if key not in spec_to_chunks:
            spec_to_chunks[key] = []
        spec_to_chunks[key].append((chunk, spec))

    # Format each specialty group independently (so cids are globally unique)
    specialist_contexts: list[dict] = []
    for spec_key, chunk_pairs in spec_to_chunks.items():
        context, sources, evidence = _format_deduplicated_chunks(chunk_pairs, seen_cids)
        all_sources.extend(sources)
        for k, v in evidence.items():
            all_evidence[k] = all_evidence.get(k, 0) + v

        display_spec = chunk_pairs[0][1] if chunk_pairs[0][1] else None
        label = SUBSPECIALTY_LABELS.get(display_spec, "General") if display_spec else "General"
        specialist_contexts.append({
            "subspecialty": spec_key,
            "label": label,
            "context": context,
            "source_count": len(sources),
            "evidence_summary": evidence,
        })

    # If no specialty grouping (broad query), create one "General" block
    if not specialist_contexts and best_by_cid:
        context, sources, evidence = _format_deduplicated_chunks(all_deduped, seen_cids)
        all_sources.extend(sources)
        all_evidence.update(evidence)
        specialist_contexts.append({
            "subspecialty": "general",
            "label": "General",
            "context": context,
            "source_count": len(sources),
            "evidence_summary": evidence,
        })

    if all_sources:
        yield {"event": "sources", "data": {"sources": all_sources}}

    # ── Step 5: Build synthesizer prompt ──────────────────────────────────────
    yield {
        "event": "status",
        "data": {"step": "synthesizing", "detail": "Synthesising evidence…"},
    }

    history_context = f"\n\n{history_summary}" if history_summary else ""

    # Choose base prompt: clinical queries get the structured clinical format
    is_clinical = safety_class == "clinical_query"

    if is_multi:
        # Multi-specialist synthesis prompt
        base_prompt = (
            CLINICAL_DECISION_PROMPT
            if is_clinical
            else MULTI_SPECIALIST_PROMPT.format(n_specs=len(specialist_contexts))
        )
        effective_system = base_prompt + HARD_SYSTEM_SUFFIX + safety_suffix + history_context

        spec_blocks = []
        for sc in specialist_contexts:
            if sc["context"]:
                spec_blocks.append(
                    f"=== {sc['label'].upper()} LITERATURE "
                    f"({sc['source_count']} sources) ===\n{sc['context']}"
                )
            else:
                spec_blocks.append(
                    f"=== {sc['label'].upper()} LITERATURE ===\n"
                    f"[No relevant results found in this subspecialty pool]"
                )

        decomp_note = (
            f"\n\nNote: This query was decomposed into {len(sub_questions)} sub-questions "
            "to improve retrieval coverage."
            if is_complex else ""
        )

        tool_content = (
            "I have retrieved the following subspecialty literature.\n\n"
            + "\n\n".join(spec_blocks)
            + "\n\n"
            + "IMPORTANT: Answer using ONLY these sources. Cite every claim with [cid].\n"
            + f"Original question: {message}"
            + decomp_note
        )
    else:
        # Single-specialty or broad
        base_prompt = (
            CLINICAL_DECISION_PROMPT
            if is_clinical
            else (system_prompt or SINGLE_SPECIALIST_PROMPT)
        )
        effective_system = base_prompt + HARD_SYSTEM_SUFFIX + safety_suffix + history_context

        context_str = specialist_contexts[0]["context"] if specialist_contexts else ""
        spec_key = specialist_contexts[0]["subspecialty"] if specialist_contexts else "general"
        label = specialist_contexts[0]["label"] if specialist_contexts else "Literature"

        tool_content = (
            f"I have retrieved the following {label} sources.\n"
            "=== DOCUMENT SOURCES ===\n"
            f"{context_str}\n"
            "=== END SOURCES ===\n\n"
            "IMPORTANT: Answer using ONLY these sources. Cite every claim with [cid].\n"
            f"Now answer: {message}"
        )

    messages.append(LLMMessage(role="user", content=tool_content))

    # ── Step 4: Stream synthesizer LLM response ───────────────────────────────
    accumulated = ""
    thinking_text = ""

    async for chunk in provider.astream(
        messages,
        system_prompt=effective_system,
        temperature=0.1,
        max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        think=enable_thinking,
    ):
        if chunk.type == "thinking":
            thinking_text += chunk.text
            yield {"event": "thinking", "data": {"text": chunk.text}}
        elif chunk.type == "text":
            accumulated += chunk.text
            yield {"event": "token", "data": {"text": chunk.text}}

    # Fallback if no text produced
    if not accumulated and all_sources:
        fallback = (
            f"Based on the retrieved sources, please answer: {message}\n"
            "Cite every claim with [cid]."
        )
        messages.append(LLMMessage(role="user", content=fallback))
        async for chunk in provider.astream(
            messages,
            system_prompt=effective_system,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        ):
            if chunk.type == "text":
                accumulated += chunk.text
                yield {"event": "token", "data": {"text": chunk.text}}

    accumulated = re.sub(r"<unused\d+>:?\s*", "", accumulated).strip()

    yield {
        "event": "complete",
        "data": {
            "answer": accumulated or "I was unable to generate a response.",
            "sources": all_sources,
            "safety_classification": safety_class,
            "evidence_summary": all_evidence,
            "thinking": thinking_text or None,
            "specialist_contexts": [
                {
                    "subspecialty": sc["subspecialty"],
                    "label": sc["label"],
                    "source_count": sc["source_count"],
                    "evidence_summary": sc["evidence_summary"],
                }
                for sc in specialist_contexts
            ],
        },
    }


# ─── Public endpoint handler ───────────────────────────────────────────────────

async def chat_stream_endpoint(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession,
) -> StreamingResponse:
    """Entry point called from the FastAPI router."""
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    safety_class = safety_classifier.classify(request.message)
    system_prompt = kb.system_prompt or SINGLE_SPECIALIST_PROMPT
    history = [{"role": m.role, "content": m.content} for m in request.history]

    # Persist user message
    try:
        user_row = ChatMessage(
            workspace_id=workspace_id,
            message_id=str(uuid.uuid4()),
            role="user",
            content=request.message,
            safety_classification=safety_class,
        )
        db.add(user_row)
        await db.commit()
    except Exception as exc:
        logger.warning(f"Could not persist user message: {exc}")
        await db.rollback()

    async def _event_generator() -> AsyncGenerator[str, None]:
        final_answer = ""
        final_sources: list = []
        final_evidence: dict = {}
        final_thinking: Optional[str] = None
        agent_steps: list[dict] = []
        step_counter = 0

        if safety_classifier.should_block_query(safety_class):
            warning = safety_classifier.get_warning_message(safety_class)
            yield _sse("token", {"text": warning})
            yield _sse("complete", {
                "answer": warning,
                "sources": [],
                "safety_classification": safety_class,
                "evidence_summary": {},
                "thinking": None,
                "specialist_contexts": [],
            })
            return

        try:
            async for event in _agent_stream(
                workspace_id=workspace_id,
                message=request.message,
                history=history,
                db=db,
                system_prompt=system_prompt,
                enable_thinking=request.enable_thinking,
                safety_class=safety_class,
            ):
                event_type = event["event"]
                event_data = event["data"]

                if event_type == "status":
                    step_counter += 1
                    agent_steps.append({
                        "id": f"step-{step_counter}",
                        "step": event_data.get("step"),
                        "detail": event_data.get("detail", ""),
                        "status": "completed",
                    })
                elif event_type == "complete":
                    final_answer = event_data.get("answer", "")
                    final_sources = event_data.get("sources", [])
                    final_evidence = event_data.get("evidence_summary", {})
                    final_thinking = event_data.get("thinking")

                yield _sse(event_type, event_data)

        except Exception as exc:
            logger.error(f"Chat stream error: {exc}", exc_info=True)
            yield _sse("error", {"message": str(exc)})

        finally:
            if final_answer:
                try:
                    assistant_row = ChatMessage(
                        workspace_id=workspace_id,
                        message_id=str(uuid.uuid4()),
                        role="assistant",
                        content=final_answer,
                        sources=final_sources or None,
                        thinking=final_thinking,
                        agent_steps=agent_steps or None,
                        safety_classification=safety_class,
                        evidence_quality=final_evidence or None,
                    )
                    db.add(assistant_row)
                    await db.commit()
                except Exception as exc:
                    logger.warning(f"Could not persist assistant message: {exc}")
                    await db.rollback()

    return StreamingResponse(
        _sse_heartbeat(_event_generator()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
