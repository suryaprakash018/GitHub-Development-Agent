"""LLM provider interfaces and factory."""

from agent.core.config import AppConfig
from agent.core.exceptions import LLMProviderError
from agent.llm.base import LLMProvider
from agent.llm.gemini import GeminiProvider
from agent.llm.groq import GroqProvider

__all__ = ["LLMProvider", "GeminiProvider", "GroqProvider", "get_llm_provider"]


def get_llm_provider(config: AppConfig) -> LLMProvider:
    """Factory function to instantiate the configured LLM provider."""
    provider_name = config.ai.provider.lower()

    if provider_name == "gemini":
        return GeminiProvider(
            api_key=config.ai.gemini_api_key,
            model_name=config.ai.primary_model,
            thinking_level=config.ai.thinking_level,
            max_retries=config.operational_mode.max_retries,
        )

    if provider_name == "groq":
        return GroqProvider(
            api_key=config.ai.groq_api_key,
            model_name=config.ai.groq_model,
            max_retries=config.operational_mode.max_retries,
        )

    raise LLMProviderError(
        f"Unsupported LLM provider '{provider_name}'. Supported: 'gemini', 'groq'."
    )
