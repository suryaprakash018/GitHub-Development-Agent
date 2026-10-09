"""Abstract base provider interface for AI / LLM models."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract interface for all LLM providers (Gemini, OpenAI, Anthropic, etc.)."""

    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generates raw text response from the model."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generates a structured, schema-validated Pydantic model response."""
        pass
