"""
OpenAI Provider

Streaming chat with tool calling via the OpenAI SDK.
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator, List, Optional

from app.core.config import settings
from app.services.llm.base import BaseLLMProvider, LLMMessage, StreamChunk

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider with streaming and tool calling."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def complete(self, messages: List[LLMMessage], system_prompt: str = "") -> str:
        import asyncio
        return asyncio.run(self._async_complete(messages, system_prompt))

    async def _async_complete(
        self, messages: List[LLMMessage], system_prompt: str = ""
    ) -> str:
        client = self._get_client()
        oai_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        oai_messages += [{"role": m.role, "content": m.content} for m in messages]

        response = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=oai_messages,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )
        return response.choices[0].message.content or ""

    async def astream(
        self,
        messages: List[LLMMessage],
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        tools: Optional[list] = None,
        think: bool = False,
    ) -> AsyncGenerator[StreamChunk, None]:
        client = self._get_client()

        oai_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        for m in messages:
            if m.images:
                content = []
                for img_bytes in m.images:
                    import base64
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
                        },
                    })
                content.append({"type": "text", "text": m.content})
                oai_messages.append({"role": m.role, "content": content})
            else:
                oai_messages.append({"role": m.role, "content": m.content})

        kwargs = {
            "model": settings.LLM_MODEL,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        tool_calls_acc: dict[int, dict] = {}

        async with await client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                if delta.content:
                    yield StreamChunk(type="text", text=delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "name": tc.function.name or "",
                                "args_json": "",
                            }
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["args_json"] += tc.function.arguments

        # Emit accumulated tool calls
        import json
        for tc_data in tool_calls_acc.values():
            try:
                args = json.loads(tc_data["args_json"] or "{}")
            except Exception:
                args = {}
            yield StreamChunk(
                type="tool_call",
                tool_name=tc_data["name"],
                tool_args=args,
            )

    def supports_vision(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True
