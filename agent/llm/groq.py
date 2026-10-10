"""Groq Cloud provider implementation using the official groq SDK."""

import json
import os
import time
from typing import TypeVar

from groq import Groq
from pydantic import BaseModel

from agent.core.exceptions import LLMProviderError
from agent.llm.base import LLMProvider
from agent.telemetry.recorder import redact_secrets
from agent.utils.logger import setup_logger

logger = setup_logger("agent.llm.groq")

T = TypeVar("T", bound=BaseModel)


class GroqProvider(LLMProvider):
    """Groq Cloud LLM provider supporting structured JSON schema generation."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        max_retries: int = 5,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key or self.api_key.strip() == "":
            raise LLMProviderError(
                "GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your .env file or environment."
            )

        if not model_name or model_name.strip() == "":
            raise LLMProviderError(
                "GROQ_MODEL is not configured. Please set GROQ_MODEL in your environment or config."
            )

        self.model_name = model_name
        self.max_retries = max_retries

        try:
            self.client = Groq(api_key=self.api_key)
        except Exception as e:
            clean_err = redact_secrets(
                str(e), custom_secrets=[self.api_key] if self.api_key else None
            )
            raise LLMProviderError(f"Failed to initialize Groq client: {clean_err}") from e

    @staticmethod
    def _is_daily_quota_exhaustion(error: Exception) -> bool:
        """Detects whether an exception represents a confirmed non-transient daily quota exhaustion on Groq.

        Distinguishes daily exhaustion (e.g. Requests Per Day / RPD limit) from transient per-minute limits (RPM/TPM).
        """
        err_str = str(error).lower()

        is_quota_related = (
            "429" in err_str
            or "rate_limit_exceeded" in err_str
            or "ratelimit" in err_str
            or "rate limit" in err_str
            or "quota" in err_str
        )
        if not is_quota_related:
            return False

        daily_indicators = (
            "requests per day",
            "per day",
            "per_day",
            "rpd",
            "daily limit",
            "daily quota",
        )
        return any(ind in err_str for ind in daily_indicators)

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generates raw text response with exponential backoff on transient errors."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                raise LLMProviderError("Empty response received from Groq model.")
            except Exception as e:
                if self._is_daily_quota_exhaustion(e):
                    clean_msg = redact_secrets(
                        str(e), custom_secrets=[self.api_key] if self.api_key else None
                    )
                    logger.error(
                        f"Groq daily API quota exhausted. Fast-failing immediately without retries: {clean_msg}"
                    )
                    raise LLMProviderError(f"Groq daily quota exhausted: {clean_msg}") from e

                if "401" in str(e) or "invalid_api_key" in str(e).lower():
                    clean_msg = redact_secrets(
                        str(e), custom_secrets=[self.api_key] if self.api_key else None
                    )
                    raise LLMProviderError(f"Groq authentication failed: {clean_msg}") from e

                last_error = e
                clean_msg = redact_secrets(
                    str(e), custom_secrets=[self.api_key] if self.api_key else None
                )
                logger.warning(f"Groq API attempt {attempt} failed: {clean_msg}. Retrying...")
                if attempt < self.max_retries:
                    wait_seconds = (
                        (5 * attempt) if "503" in str(e) or "UNAVAILABLE" in str(e) else 2**attempt
                    )
                    time.sleep(wait_seconds)

        clean_last = redact_secrets(
            str(last_error), custom_secrets=[self.api_key] if self.api_key else None
        )
        raise LLMProviderError(
            f"Groq generate_text failed after {self.max_retries} attempts: {clean_last}"
        )

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generates structured response validated strictly against the given Pydantic schema."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_instruction = (
            (f"{system_prompt}\n\n" if system_prompt else "")
            + "You are an automated software development agent.\n"
            + "You MUST output a valid JSON object matching the following JSON Schema exactly.\n"
            + "Do not include any explanation, markdown code fences, or additional text outside the JSON object.\n"
            + f"JSON Schema:\n{schema_json}"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                if not response.choices or not response.choices[0].message.content:
                    raise LLMProviderError("Empty structured response from Groq model.")

                raw_content = response.choices[0].message.content.strip()
                # Clean any markdown fences if present
                if raw_content.startswith("```json"):
                    raw_content = raw_content[7:]
                elif raw_content.startswith("```"):
                    raw_content = raw_content[3:]
                if raw_content.endswith("```"):
                    raw_content = raw_content[:-3]
                raw_content = raw_content.strip()

                return schema.model_validate_json(raw_content)
            except Exception as e:
                if self._is_daily_quota_exhaustion(e):
                    clean_msg = redact_secrets(
                        str(e), custom_secrets=[self.api_key] if self.api_key else None
                    )
                    logger.error(
                        f"Groq daily API quota exhausted. Fast-failing immediately without retries: {clean_msg}"
                    )
                    raise LLMProviderError(f"Groq daily quota exhausted: {clean_msg}") from e

                if "401" in str(e) or "invalid_api_key" in str(e).lower():
                    clean_msg = redact_secrets(
                        str(e), custom_secrets=[self.api_key] if self.api_key else None
                    )
                    raise LLMProviderError(f"Groq authentication failed: {clean_msg}") from e

                last_error = e
                clean_msg = redact_secrets(
                    str(e), custom_secrets=[self.api_key] if self.api_key else None
                )
                logger.warning(
                    f"Groq structured generation attempt {attempt} failed: {clean_msg}. Retrying..."
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
            f"Groq structured generation failed after {self.max_retries} attempts: {clean_last}"
        )
