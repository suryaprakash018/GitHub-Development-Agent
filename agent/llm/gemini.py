"""Google Gemini provider implementation using the official google-genai SDK."""

import os
import time
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from agent.core.exceptions import LLMProviderError
from agent.llm.base import LLMProvider
from agent.telemetry.recorder import redact_secrets
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

    @staticmethod
    def _is_daily_quota_exhaustion(error: Exception) -> bool:
        """Detects whether an exception represents a confirmed non-transient daily quota exhaustion.

        Returns True only when the error explicitly indicates daily/free-tier limit exhaustion,
        distinguishing it from temporary per-minute rate limits (429 RPM/TPM) or transient network errors.
        """
        err_str = str(error).lower()

        # Must represent quota exhaustion, HTTP 429, or resource_exhausted
        is_quota_related = (
            "429" in err_str
            or "resource_exhausted" in err_str
            or "quota exceeded" in err_str
            or "quota_exceeded" in err_str
        )
        if not is_quota_related:
            return False

        # Specific patterns indicating daily / per-day quota exhaustion
        daily_indicators = (
            "generaterequestsperday",
            "perday",
            "per_day",
            "per day",
            "daily quota",
            "requests per day",
        )
        if any(ind in err_str for ind in daily_indicators):
            return True

        return "quota exceeded" in err_str and (
            "freetier" in err_str or "free-tier" in err_str or "day" in err_str
        )

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
                if self._is_daily_quota_exhaustion(e):
                    clean_msg = redact_secrets(
                        str(e), custom_secrets=[self.api_key] if self.api_key else None
                    )
                    logger.error(
                        f"Gemini daily API quota exhausted. Fast-failing immediately without retries: {clean_msg}"
                    )
                    raise LLMProviderError(f"Gemini daily quota exhausted: {clean_msg}") from e

                last_error = e
                clean_msg = redact_secrets(
                    str(e), custom_secrets=[self.api_key] if self.api_key else None
                )
                logger.warning(f"Gemini API attempt {attempt} failed: {clean_msg}. Retrying...")
                if attempt < self.max_retries:
                    wait_seconds = (
                        5 * attempt if "503" in str(e) or "UNAVAILABLE" in str(e) else 2**attempt
                    )
                    time.sleep(wait_seconds)

        clean_last = redact_secrets(
            str(last_error), custom_secrets=[self.api_key] if self.api_key else None
        )
        raise LLMProviderError(
            f"Gemini generate_text failed after {self.max_retries} attempts: {clean_last}"
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
                if self._is_daily_quota_exhaustion(e):
                    clean_msg = redact_secrets(
                        str(e), custom_secrets=[self.api_key] if self.api_key else None
                    )
                    logger.error(
                        f"Gemini daily API quota exhausted. Fast-failing immediately without retries: {clean_msg}"
                    )
                    raise LLMProviderError(f"Gemini daily quota exhausted: {clean_msg}") from e

                last_error = e
                clean_msg = redact_secrets(
                    str(e), custom_secrets=[self.api_key] if self.api_key else None
                )
                logger.warning(
                    f"Gemini structured generation attempt {attempt} failed: {clean_msg}. Retrying..."
                )
                if attempt < self.max_retries:
                    wait_seconds = (
                        (10 * attempt) if "503" in str(e) or "UNAVAILABLE" in str(e) else 2**attempt
                    )
                    time.sleep(wait_seconds)

        clean_last = redact_secrets(
            str(last_error), custom_secrets=[self.api_key] if self.api_key else None
        )
        raise LLMProviderError(
            f"Gemini structured generation failed after {self.max_retries} attempts: {clean_last}"
        )
