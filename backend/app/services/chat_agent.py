"""
Medical Agentic Chat

SSE streaming chat with:
- Tool calling (search_documents)
- Medical citation formatting (PMID + evidence level)
- Safety classification checks
- Evidence quality summarisation
- Speculative token streaming with rollback on tool call
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

MAX_ITERATIONS = 3
SSE_HEARTBEAT_INTERVAL = 15  # seconds
_CID_CHARS = string.ascii_lowercase + string.digits


# ─── System prompt ─────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
You are a medical research assistant specialising in plastic and reconstructive surgery.
You help clinicians and researchers understand evidence from peer-reviewed literature.

## Citation Format
After every factual claim, add an inline citation:  claim text[cid]
Example: "DIEP flap has a 3-5% flap failure rate[a3x9]."

## Evidence Grading
When citing, mention the evidence level where known:
"A 2022 RCT (Level I) found that..." or "A retrospective cohort (Level III) reported..."

## Conservative Language
Use "studies suggest", "evidence indicates", "research shows" rather than definitive claims.

## Limitations
Always acknowledge: "This is a summary of published research and should not replace
clinical judgment or individual patient assessment."
"""

HARD_SYSTEM_SUFFIX = """
⚠️  SAFETY RULES (non-negotiable):
- NEVER give patient-specific treatment recommendations.
- NEVER calculate doses for individual patients.
- ALWAYS remind the user this is educational only.
- If asked about emergencies, tell them to call emergency services immediately.
"""


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _new_cid(seen: set[str]) -> str:
    """Generate unique 4-char citation ID."""
    while True:
        cid = "".join(random.choices(_CID_CHARS, k=4))
        if any(c.isalpha() for c in cid) and cid not in seen:
            return cid


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


async def _sse_heartbeat(
    source: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Wrap SSE generator with keep-alive heartbeats."""
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


# ─── Search tool executor ──────────────────────────────────────────────────────

async def _execute_search(
    workspace_id: int,
    query: str,
    top_k: int,
    db: AsyncSession,
    seen_cids: set[str],
) -> tuple[str, list[dict], dict[str, int]]:
    """
    Run hybrid retrieval and return:
      - formatted context string
      - list of source dicts (for SSE sources event)
      - evidence_summary dict
    """
    rag = get_rag_service(db, workspace_id)
    result = await rag.query(question=query, top_k=min(top_k, 10))

    sources = []
    context_parts = []
    evidence_summary: dict[str, int] = {}

    for chunk in result.chunks:
        cid = _new_cid(seen_cids)
        seen_cids.add(cid)

        # Build citation line
        meta_parts = []
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
        })

    context = "\n\n---\n\n".join(context_parts)
    return context, sources, evidence_summary


# ─── Agent loop ────────────────────────────────────────────────────────────────

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

    Modes
    -----
    force_search=True  : pre-retrieve then ask LLM to answer from context
    force_search=False : agentic tool calling (LLM decides when to search)
    """
    provider = get_llm_provider()
    seen_cids: set[str] = set()
    all_sources: list[dict] = []
    all_evidence: dict[str, int] = {}

    # Build conversation history (last 10 turns)
    messages: List[LLMMessage] = [
        LLMMessage(role="user" if m["role"] == "user" else "assistant", content=m["content"])
        for m in history[-10:]
    ]

    # Safety prompt modifier
    safety_suffix = safety_classifier.get_system_prompt_modification(safety_class)
    effective_system = system_prompt + HARD_SYSTEM_SUFFIX + safety_suffix

    # ── Force-search mode ──────────────────────────────────────────────────────
    if force_search:
        yield {"event": "status", "data": {"step": "retrieving", "detail": f"Searching: {message[:80]}…"}}
        context, sources, evidence = await _execute_search(
            workspace_id, message, 8, db, seen_cids
        )
        all_sources.extend(sources)
        all_evidence.update(evidence)

        if sources:
            yield {"event": "sources", "data": {"sources": sources}}

        tool_result = (
            "I have retrieved the following document sources.\n"
            "=== DOCUMENT SOURCES ===\n"
            f"{context}\n"
            "=== END SOURCES ===\n\n"
            "IMPORTANT: Answer using ONLY these sources. Cite every claim with [cid].\n"
            f"Now answer: {message}"
        )
        messages.append(LLMMessage(role="user", content=tool_result))

    else:
        # ── Agentic tool-calling mode ──────────────────────────────────────────
        messages.append(LLMMessage(role="user", content=message))

    yield {"event": "status", "data": {"step": "analyzing", "detail": "Analysing your question…"}}

    accumulated = ""
    thinking_text = ""

    for iteration in range(MAX_ITERATIONS):
        iter_text = ""
        tool_call_detected = False
        tokens_streamed = False

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

            elif chunk.type == "tool_call":
                tool_call_detected = True
                # Rollback any speculatively streamed tokens
                if tokens_streamed:
                    accumulated = ""
                    yield {"event": "token_rollback", "data": {}}

                tool_name = chunk.tool_name
                tool_args = chunk.tool_args

                if tool_name == "search_documents":
                    query = tool_args.get("query", message)
                    top_k = int(tool_args.get("top_k", 8))

                    yield {"event": "status", "data": {
                        "step": "retrieving", "detail": f"Searching: {query[:80]}…"
                    }}

                    context, sources, evidence = await _execute_search(
                        workspace_id, query, top_k, db, seen_cids
                    )
                    all_sources.extend(sources)
                    for k, v in evidence.items():
                        all_evidence[k] = all_evidence.get(k, 0) + v

                    if sources:
                        yield {"event": "sources", "data": {"sources": sources}}

                    tool_result_content = (
                        "I have retrieved the following document sources.\n"
                        "=== DOCUMENT SOURCES ===\n"
                        f"{context}\n"
                        "=== END SOURCES ===\n\n"
                        "IMPORTANT: Answer using ONLY these sources. Cite every claim with [cid].\n"
                        f"Now answer: {message}"
                    )

                    # Append the tool exchange to messages for next iteration
                    messages.append(LLMMessage(role="assistant", content=f"[Called search_documents(query={query!r})]"))
                    messages.append(LLMMessage(role="user", content=tool_result_content))

                    yield {"event": "status", "data": {"step": "generating", "detail": "Generating answer…"}}

            elif chunk.type == "text":
                iter_text += chunk.text
                if not tool_call_detected:
                    accumulated += chunk.text
                    tokens_streamed = True
                    yield {"event": "token", "data": {"text": chunk.text}}

        if not tool_call_detected:
            # LLM answered directly — we're done
            break

    # ── Fallback: no text produced after searching ─────────────────────────────
    if not accumulated and all_sources:
        fallback_prompt = (
            "Based on the sources already retrieved, please provide a comprehensive "
            f"answer to: {message}\n\n"
            "Cite every claim with the provided [cid] identifiers."
        )
        messages.append(LLMMessage(role="user", content=fallback_prompt))
        async for chunk in provider.astream(
            messages,
            system_prompt=system_prompt + HARD_SYSTEM_SUFFIX,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        ):
            if chunk.type == "text":
                accumulated += chunk.text
                yield {"event": "token", "data": {"text": chunk.text}}

    # Clean up model artifacts
    accumulated = re.sub(r"<unused\d+>:?\s*", "", accumulated).strip()

    yield {"event": "complete", "data": {
        "answer": accumulated or "I was unable to generate a response.",
        "sources": all_sources,
        "safety_classification": safety_class,
        "evidence_summary": all_evidence,
        "thinking": thinking_text or None,
    }}


# ─── Public endpoint handler ───────────────────────────────────────────────────

async def chat_stream_endpoint(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession,
) -> StreamingResponse:
    """
    Entry point called from the FastAPI router.
    Returns a StreamingResponse with SSE events.
    """
    # Verify workspace
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    # Safety classification
    safety_class = safety_classifier.classify(request.message)

    # Build effective system prompt
    system_prompt = (kb.system_prompt or DEFAULT_SYSTEM_PROMPT)

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

        # If emergency — block immediately
        if safety_classifier.should_block_query(safety_class):
            warning = safety_classifier.get_warning_message(safety_class)
            yield _sse("token", {"text": warning})
            yield _sse("complete", {
                "answer": warning,
                "sources": [],
                "safety_classification": safety_class,
                "evidence_summary": {},
                "thinking": None,
            })
            return

        # Emit safety warning for patient-specific queries (non-blocking)
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

                # Track steps for persistence
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
            # Persist assistant message
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
