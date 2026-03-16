"""
Ollama Provider

Local LLM with prompt-based tool calling for offline usage.
"""
from __future__ import annotations

import json
import logging
import re
from typing import AsyncGenerator, List, Optional

from app.core.config import settings
from app.services.llm.base import BaseLLMProvider, LLMMessage, StreamChunk

logger = logging.getLogger(__name__)

# Prompt-based tool calling instructions for Ollama models
_TOOL_SYSTEM = """\
## TOOL: search_documents

You have ONE tool: search_documents. Call it by outputting EXACTLY:

<tool_call>{"name": "search_documents", "arguments": {"query": "<rewritten query>"}}</tool_call>

### RULES:
1. For ANY medical question — ALWAYS call search_documents FIRST.
2. Your ENTIRE first response to a medical query must be ONLY the <tool_call> block.
3. Rewrite the query to be specific and detailed.
4. After receiving results, answer using ONLY those sources with citations.
5. Format citations as: claim text[source_id]
6. Simple greetings/thanks do NOT require a tool call.
"""


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    def complete(self, messages: List[LLMMessage], system_prompt: str = "") -> str:
        import ollama

        response = ollama.chat(
            model=settings.LLM_MODEL,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            options={"temperature": settings.LLM_TEMPERATURE},
        )
        return response["message"]["content"]

    async def astream(
        self,
        messages: List[LLMMessage],
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        tools: Optional[list] = None,
        think: bool = False,
    ) -> AsyncGenerator[StreamChunk, None]:
        import asyncio
        import ollama

        # Inject tool calling instructions into system prompt
        full_system = (system_prompt or "") + "\n\n" + _TOOL_SYSTEM

        ollama_messages = [{"role": "system", "content": full_system}]
        for m in messages:
            ollama_messages.append({"role": m.role, "content": m.content})

        def _sync_stream():
            return ollama.chat(
                model=settings.LLM_MODEL,
                messages=ollama_messages,
                stream=True,
                options={"temperature": temperature},
            )

        loop = asyncio.get_event_loop()
        stream = await loop.run_in_executor(None, _sync_stream)

        accumulated = ""
        for chunk in stream:
            text = chunk.get("message", {}).get("content", "")
            if text:
                accumulated += text
                # Check for tool call pattern
                if "<tool_call>" in accumulated and "</tool_call>" in accumulated:
                    match = re.search(
                        r"<tool_call>(.*?)</tool_call>", accumulated, re.DOTALL
                    )
                    if match:
                        try:
                            call_data = json.loads(match.group(1))
                            yield StreamChunk(
                                type="tool_call",
                                tool_name=call_data.get("name", ""),
                                tool_args=call_data.get("arguments", {}),
                            )
                            accumulated = ""
                            return
                        except json.JSONDecodeError:
                            pass
                else:
                    yield StreamChunk(type="text", text=text)
