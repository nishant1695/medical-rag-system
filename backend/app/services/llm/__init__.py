"""
LLM Package

Factory for creating LLM provider instances based on configuration.
"""
from __future__ import annotations

from typing import Optional

from app.core.config import settings
from app.services.llm.base import BaseLLMProvider

_provider: Optional[BaseLLMProvider] = None


def get_llm_provider() -> BaseLLMProvider:
    """Get or create the configured LLM provider."""
    global _provider
    if _provider is None:
        name = settings.LLM_PROVIDER.lower()

        if name == "anthropic":
            from app.services.llm.anthropic_provider import AnthropicProvider
            _provider = AnthropicProvider()

        elif name == "ollama":
            from app.services.llm.ollama_provider import OllamaProvider
            _provider = OllamaProvider()

        elif name == "openai":
            from app.services.llm.openai_provider import OpenAIProvider
            _provider = OpenAIProvider()

        else:
            raise ValueError(
                f"Unsupported LLM provider: {name!r}. "
                "Choose from: anthropic, ollama, openai"
            )

    return _provider
