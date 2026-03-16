"""
Anthropic Claude Provider

Streaming chat with tool calling and optional extended thinking.
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator, List, Optional

from app.core.config import settings
from app.services.llm.base import BaseLLMProvider, LLMMessage, StreamChunk

logger = logging.getLogger(__name__)

# Medical system prompt reinforcement injected on every call
_MEDICAL_TOOL_SYSTEM = """

## Tool Usage (MANDATORY)

You have a tool called `search_documents` that searches the knowledge base.

### ABSOLUTE RULES:
1. For ALL medical questions, requests, or factual queries — call `search_documents` FIRST.
2. Only skip the tool call for:
   - Simple greetings ("hello", "hi", "thanks")
   - Farewells ("bye", "goodbye")
3. NEVER answer medical questions without searching first — your knowledge may be outdated.
4. Rewrite the user's query to be specific and detailed for better retrieval.
5. ALWAYS include PMID and evidence level in citations when available.
"""


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider with streaming and tool calling."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        return self._client

    def complete(self, messages: List[LLMMessage], system_prompt: str = "") -> str:
        client = self._get_client()
        response = client.messages.create(
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            system=system_prompt or "You are a helpful medical research assistant.",
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return response.content[0].text

    async def astream(
        self,
        messages: List[LLMMessage],
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        tools: Optional[list] = None,
        think: bool = False,
    ) -> AsyncGenerator[StreamChunk, None]:
        import anthropic

        client = self._get_client()

        # Build Anthropic messages
        anthropic_messages = []
        for m in messages:
            if m.images:
                content = []
                for img_bytes in m.images:
                    import base64
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(img_bytes).decode(),
                        },
                    })
                content.append({"type": "text", "text": m.content})
                anthropic_messages.append({"role": m.role, "content": content})
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        kwargs = {
            "model": settings.LLM_MODEL,
            "max_tokens": max_tokens,
            "system": system_prompt or "You are a helpful medical research assistant.",
            "messages": anthropic_messages,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools

        # Extended thinking (claude-3-7+ models)
        if think and "claude-3-7" in settings.LLM_MODEL:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 5000}

        # Stream response
        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "type"):
                            if delta.type == "text_delta":
                                yield StreamChunk(type="text", text=delta.text)
                            elif delta.type == "thinking_delta":
                                yield StreamChunk(type="thinking", text=delta.thinking)
                    elif event.type == "content_block_start":
                        block = event.content_block
                        if hasattr(block, "type") and block.type == "tool_use":
                            # Accumulate tool input
                            self._pending_tool = {
                                "name": block.name,
                                "id": block.id,
                                "input_json": "",
                            }
                    elif event.type == "content_block_stop":
                        if hasattr(self, "_pending_tool") and self._pending_tool:
                            import json
                            try:
                                args = json.loads(self._pending_tool["input_json"] or "{}")
                            except Exception:
                                args = {}
                            yield StreamChunk(
                                type="tool_call",
                                tool_name=self._pending_tool["name"],
                                tool_args=args,
                            )
                            self._pending_tool = None
                    # Accumulate tool input JSON
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "type") and delta.type == "input_json_delta":
                            if hasattr(self, "_pending_tool") and self._pending_tool:
                                self._pending_tool["input_json"] += delta.partial_json

    def supports_vision(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True

    def supports_thinking(self) -> bool:
        return "claude-3-7" in settings.LLM_MODEL
