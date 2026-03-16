"""
Medical Agentic Chat — Multi-Specialist Architecture

Flow:
  1. Classify which subspecialties the query touches (keyword-based, no LLM call)
  2. Run parallel subspecialty-filtered retrievals (one per matched specialty)
  3. Single synthesizer LLM produces a structured answer:
       - What each specialty's literature says
       - Points of agreement / disagreement
       - Final evidence-based verdict
  4. Stream SSE events throughout

For single-specialty queries the flow degenerates to the simple RAG path
(one retrieval, one LLM call) with no extra overhead.
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
from app.services.rag_service import get_rag_service

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
    "breast",
    "hand",
    "craniofacial",
    "microsurgery",
    "burns",
    "aesthetic",
    "lower_extremity",
]

# Keywords that map a query to one or more subspecialties.
# Overlap is intentional — "free flap" is both breast and microsurgery.
SUBSPECIALTY_KEYWORDS: dict[str, list[str]] = {
    "breast": [
        "breast", "diep", "tram", "tug", "sgap", "igap", "lat flap", "mastectomy",
        "lumpectomy", "nipple", "areola", "implant", "expander", "reconstruction",
        "mammoplasty", "augmentation", "reduction mammoplasty", "ptosis",
    ],
    "hand": [
        "hand", "finger", "thumb", "wrist", "tendon", "carpal tunnel", "dupuytren",
        "trigger finger", "replantation", "digital", "metacarpal", "phalanx",
        "nerve repair", "extensor", "flexor",
    ],
    "craniofacial": [
        "craniofacial", "skull", "orbit", "mandible", "maxilla", "cleft",
        "cranial", "facial bone", "jaw", "zygoma", "forehead", "craniosynostosis",
        "midface", "le fort", "distraction osteogenesis",
    ],
    "microsurgery": [
        "free flap", "perforator", "anastomosis", "microsurg", "alt flap", "gracilis",
        "latissimus", "flap failure", "flap necrosis", "pedicle", "vascular",
        "supermicrosurgery", "lymphedema", "lymphatic", "vascularised",
    ],
    "burns": [
        "burn", "eschar", "scar", "contracture", "skin graft", "split thickness",
        "full thickness graft", "dermis", "inhalation injury", "tbsa",
        "total body surface area", "escharotomy",
    ],
    "aesthetic": [
        "aesthetic", "cosmetic", "rhinoplasty", "facelift", "rhytidectomy",
        "liposuction", "blepharoplasty", "botox", "filler", "hyaluronic",
        "abdominoplasty", "tummy tuck", "brow lift", "otoplasty",
    ],
    "lower_extremity": [
        "lower extremity", "lower limb", "leg", "foot", "ankle", "tibial",
        "limb salvage", "below knee", "above knee", "degloving", "achilles",
    ],
}

# Display names for the UI
SUBSPECIALTY_LABELS: dict[str, str] = {
    "breast": "Breast Surgery",
    "hand": "Hand Surgery",
    "craniofacial": "Craniofacial Surgery",
    "microsurgery": "Microsurgery",
    "burns": "Burns & Wound Care",
    "aesthetic": "Aesthetic Surgery",
    "lower_extremity": "Lower Extremity Reconstruction",
}

# ─── System prompts ───────────────────────────────────────────────────────────

SINGLE_SPECIALIST_PROMPT = """\
You are a medical research assistant specialising in plastic and reconstructive surgery.
You help clinicians and researchers understand evidence from peer-reviewed literature.

## Citation Format
After every factual claim add an inline citation: claim text[cid]
Example: "DIEP flap has a 3-5% flap failure rate[a3x9]."

## Evidence Grading
Mention the evidence level where known:
"A 2022 RCT (Level I) found..." or "A retrospective cohort (Level III) reported..."

## Conservative Language
Use "studies suggest", "evidence indicates" rather than definitive claims.

## Limitations
Always acknowledge: "This is a summary of published research and should not replace
clinical judgment or individual patient assessment."
"""

MULTI_SPECIALIST_PROMPT = """\
You are a senior consultant in plastic and reconstructive surgery synthesising evidence
from multiple subspecialty literature pools.

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
Explain the likely reasons (patient population, indication, surgeon preference).

### Final Verdict
A concise evidence-graded summary answering the original question.
Include the strongest evidence level available (e.g. "supported by Level I–II evidence").

## Rules
- Cite every factual claim with [cid].
- Do NOT invent findings — if a specialty pool has no relevant results, say so.
- Conservative language: "evidence suggests", "studies indicate".
- End with: "This is a research summary — not a substitute for clinical judgment."
"""

HARD_SYSTEM_SUFFIX = """
⚠️  SAFETY RULES (non-negotiable):
- NEVER give patient-specific treatment recommendations.
- NEVER calculate doses for individual patients.
- ALWAYS remind the user this is educational only.
- If asked about emergencies, tell them to call emergency services immediately.
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


# ─── Agent stream ─────────────────────────────────────────────────────────────

async def _agent_stream(
    workspace_id: int,
    message: str,
    history: List[dict],
    db: AsyncSession,
    system_prompt: str,
    enable_thinking: bool,
    force_search: bool,
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

    # ── Step 2: Parallel specialist retrievals ────────────────────────────────
    is_multi = len(subspecialties) > 1

    if not subspecialties:
        # Broad query — single retrieval, no filter
        retrieval_tasks = [(None, SPECIALIST_TOP_K * 2)]
    else:
        retrieval_tasks = [(spec, SPECIALIST_TOP_K) for spec in subspecialties]

    for spec, _ in retrieval_tasks:
        label = SUBSPECIALTY_LABELS.get(spec, "All literature") if spec else "All literature"
        yield {
            "event": "status",
            "data": {"step": "retrieving", "detail": f"Searching {label}…"},
        }

    # Run all retrievals concurrently (use contextualized query for retrieval)
    async_tasks = [
        _retrieve_for_specialty(workspace_id, retrieval_query, spec, db, seen_cids, top_k)
        for spec, top_k in retrieval_tasks
    ]
    retrieval_results = await asyncio.gather(*async_tasks)

    # Collect sources; emit one sources event with subspecialty grouping
    specialist_contexts: list[dict] = []  # for the LLM prompt
    for (spec, _), (context, sources, evidence) in zip(retrieval_tasks, retrieval_results):
        all_sources.extend(sources)
        for k, v in evidence.items():
            all_evidence[k] = all_evidence.get(k, 0) + v

        label = SUBSPECIALTY_LABELS.get(spec, "General") if spec else "General"
        specialist_contexts.append({
            "subspecialty": spec or "general",
            "label": label,
            "context": context,
            "source_count": len(sources),
            "evidence_summary": evidence,
        })

    if all_sources:
        yield {"event": "sources", "data": {"sources": all_sources}}

    # ── Step 3: Build synthesizer prompt ──────────────────────────────────────
    yield {
        "event": "status",
        "data": {"step": "synthesizing", "detail": "Synthesising evidence…"},
    }

    history_context = f"\n\n{history_summary}" if history_summary else ""

    if is_multi:
        # Multi-specialist synthesis prompt
        effective_system = (
            MULTI_SPECIALIST_PROMPT.format(n_specs=len(subspecialties))
            + HARD_SYSTEM_SUFFIX
            + safety_suffix
            + history_context
        )
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

        tool_content = (
            "I have retrieved the following subspecialty literature.\n\n"
            + "\n\n".join(spec_blocks)
            + "\n\n"
            + "IMPORTANT: Answer using ONLY these sources. Cite every claim with [cid].\n"
            + f"Original question: {message}"
        )
    else:
        # Single-specialty or broad — use standard prompt
        effective_system = system_prompt + HARD_SYSTEM_SUFFIX + safety_suffix + history_context
        context_str = retrieval_results[0][0] if retrieval_results else ""
        spec = retrieval_tasks[0][0]
        label = SUBSPECIALTY_LABELS.get(spec, "Literature") if spec else "Literature"
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

        if safety_class == "patient_specific":
            warning_msg = safety_classifier.get_warning_message(safety_class)
            yield _sse("safety_warning", {"message": warning_msg, "classification": safety_class})

        try:
            async for event in _agent_stream(
                workspace_id=workspace_id,
                message=request.message,
                history=history,
                db=db,
                system_prompt=system_prompt,
                enable_thinking=request.enable_thinking,
                force_search=request.force_search,
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
