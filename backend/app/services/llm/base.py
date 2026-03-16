"""
LLM Provider Abstraction

Unified interface for Anthropic Claude, OpenAI, Gemini, and Ollama.
Supports streaming and tool calling.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    """A chat message for the LLM."""

    role: str  # "user" | "assistant" | "system"
    content: str
    images: List[bytes] = field(default_factory=list)  # raw bytes for vision


@dataclass
class StreamChunk:
    """A streamed token or event from the LLM."""

    type: str           # "text" | "thinking" | "tool_call" | "done"
    text: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def complete(self, messages: List[LLMMessage], system_prompt: str = "") -> str:
        """Synchronous completion."""

    @abstractmethod
    async def astream(
        self,
        messages: List[LLMMessage],
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        tools: Optional[list] = None,
        think: bool = False,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async streaming completion."""

    def supports_vision(self) -> bool:
        return False

    def supports_thinking(self) -> bool:
        return False

    def supports_tool_calling(self) -> bool:
        return False
