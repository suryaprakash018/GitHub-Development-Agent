"""Google Gemini provider implementation using the official google-genai SDK."""

import os
import time
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from agent.core.exceptions import LLMProviderError
from agent.llm.base import LLMProvider
from agent.utils.logger import setup_logger

logger = setup_logger("agent.llm.gemini")

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider supporting structured JSON schema generation."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3.8-flash",
        thinking_level: str | None = None,
        max_retries: int = 5,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key or self.api_key.strip() == "":
            raise LLMProviderError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your .env file or environment."
            )

        self.model_name = model_name
        self.thinking_level = thinking_level
        self.max_retries = max_retries

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Google GenAI client: {e}") from e

    def _build_thinking_config(self) -> types.ThinkingConfig | None:
        """Constructs ThinkingConfig for Gemini 3.8 models when thinking_level is set."""
        if not self.thinking_level:
            return None
        return types.ThinkingConfig(thinking_level=str(self.thinking_level).upper())

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generates raw text response with exponential backoff on transient errors."""
        config = types.GenerateContentConfig(
            thinking_config=self._build_thinking_config(),
            system_instruction=system_prompt if system_prompt else None,
        )

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                if response.text:
                    return response.text
                raise LLMProviderError("Empty response received from Gemini model.")
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini API attempt {attempt} failed: {e}. Retrying...")
                if attempt < self.max_retries:
                    wait_seconds = (
                        5 * attempt if "503" in str(e) or "UNAVAILABLE" in str(e) else 2**attempt
                    )
                    time.sleep(wait_seconds)

        raise LLMProviderError(
            f"Gemini generate_text failed after {self.max_retries} attempts: {last_error}"
        )

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generates structured response validated strictly against the given Pydantic schema."""
        config = types.GenerateContentConfig(
            thinking_config=self._build_thinking_config(),
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction=system_prompt if system_prompt else None,
        )

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                if not response.text:
                    raise LLMProviderError("Empty structured response from Gemini model.")

                return schema.model_validate_json(response.text)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Gemini structured generation attempt {attempt} failed: {e}. Retrying..."
                )
                if attempt < self.max_retries:
                    wait_seconds = (
                        (10 * attempt) if "503" in str(e) or "UNAVAILABLE" in str(e) else 2**attempt
                    )
                    time.sleep(wait_seconds)

        raise LLMProviderError(
            f"Gemini structured generation failed after {self.max_retries} attempts: {last_error}"
        )
