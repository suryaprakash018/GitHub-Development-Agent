"""Unit tests for LLM provider contract, factory, and error handling."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agent.core.config import AppConfig
from agent.core.exceptions import LLMProviderError
from agent.llm import GeminiProvider, GroqProvider, get_llm_provider


class SampleTaskPlan(BaseModel):
    task_id: str
    summary: str
    target_files: list[str]


def test_gemini_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMProviderError, match="GEMINI_API_KEY is not configured"):
        GeminiProvider(api_key="")


def test_get_llm_provider_unsupported_raises_error():
    config = AppConfig()
    config.ai.provider = "unsupported_provider"
    with pytest.raises(LLMProviderError, match="Unsupported LLM provider"):
        get_llm_provider(config)


def test_gemini_provider_mock_structured_generation(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_mock_key")

    provider = GeminiProvider(api_key="dummy_mock_key")

    # Mock the underlying client generate_content
    mock_response = MagicMock()
    mock_response.text = (
        '{"task_id": "test_01", "summary": "Build test module", "target_files": ["a.py", "b.py"]}'
    )

    provider.client = MagicMock()
    provider.client.models.generate_content.return_value = mock_response

    result = provider.generate_structured(
        prompt="Generate task plan",
        schema=SampleTaskPlan,
    )

    assert isinstance(result, SampleTaskPlan)
    assert result.task_id == "test_01"
    assert result.summary == "Build test module"
    assert result.target_files == ["a.py", "b.py"]


def test_gemini_provider_default_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_mock_key")
    provider = GeminiProvider(api_key="dummy_mock_key")
    assert provider.model_name == "gemini-3.8-flash"


def test_gemini_provider_custom_model():
    provider = GeminiProvider(api_key="dummy_mock_key", model_name="gemini-custom-model")
    assert provider.model_name == "gemini-custom-model"


def test_get_llm_provider_wires_model_from_config(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    config = AppConfig()
    config.ai.provider = "gemini"
    config.ai.primary_model = "gemini-3.8-flash"
    config.ai.thinking_level = "LOW"
    config.ai.gemini_api_key = "dummy_mock_key"
    provider = get_llm_provider(config)
    assert isinstance(provider, GeminiProvider)
    assert provider.model_name == "gemini-3.8-flash"
    assert provider.thinking_level == "LOW"


def test_gemini_provider_thinking_config_construction():
    provider_default = GeminiProvider(api_key="dummy_mock_key")
    assert provider_default.thinking_level is None
    assert provider_default._build_thinking_config() is None

    provider_with_level = GeminiProvider(api_key="dummy_mock_key", thinking_level="LOW")
    thinking_cfg = provider_with_level._build_thinking_config()
    assert thinking_cfg is not None
    assert str(thinking_cfg.thinking_level) == "ThinkingLevel.LOW"
    assert thinking_cfg.thinking_budget is None


def test_gemini_provider_generate_content_config_has_no_deprecated_parameters():
    provider = GeminiProvider(api_key="dummy_mock_key", thinking_level="LOW")
    mock_response = MagicMock()
    mock_response.text = '{"task_id": "test", "summary": "test", "target_files": []}'

    provider.client = MagicMock()
    provider.client.models.generate_content.return_value = mock_response

    # Call generate_text
    provider.generate_text(prompt="hello")
    _, kwargs = provider.client.models.generate_content.call_args
    cfg = kwargs["config"]
    assert cfg.temperature is None
    assert cfg.top_p is None
    assert cfg.top_k is None
    assert cfg.candidate_count is None
    assert cfg.thinking_config.thinking_budget is None
    assert str(cfg.thinking_config.thinking_level) == "ThinkingLevel.LOW"

    # Call generate_structured
    provider.generate_structured(prompt="hello", schema=SampleTaskPlan)
    _, kwargs = provider.client.models.generate_content.call_args
    cfg_structured = kwargs["config"]
    assert cfg_structured.temperature is None
    assert cfg_structured.top_p is None
    assert cfg_structured.top_k is None
    assert cfg_structured.candidate_count is None
    assert cfg_structured.thinking_config.thinking_budget is None
    assert str(cfg_structured.thinking_config.thinking_level) == "ThinkingLevel.LOW"
    assert cfg_structured.response_mime_type == "application/json"


def test_gemini_provider_daily_quota_exhaustion_fails_fast_structured():
    """Confirms that explicit daily free-tier quota exhaustion fails immediately without retrying."""
    provider = GeminiProvider(api_key="dummy_mock_key", max_retries=5)
    provider.client = MagicMock()
    quota_error = Exception(
        "429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric "
        "'GenerateRequestsPerDayPerProjectPerModel-FreeTier' and limit "
        "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"
    )
    provider.client.models.generate_content.side_effect = quota_error

    with pytest.raises(LLMProviderError, match="Gemini daily quota exhausted"):
        provider.generate_structured(prompt="hello", schema=SampleTaskPlan)

    # CRITICAL: Verify generate_content was called ONLY once (failed immediately without retries)
    assert provider.client.models.generate_content.call_count == 1


def test_gemini_provider_daily_quota_exhaustion_fails_fast_text():
    """Confirms that text generation also fast-fails on daily quota exhaustion."""
    provider = GeminiProvider(api_key="dummy_mock_key", max_retries=5)
    provider.client = MagicMock()
    quota_error = Exception("429 Resource has been exhausted: quota exceeded for daily quota limit")
    provider.client.models.generate_content.side_effect = quota_error

    with pytest.raises(LLMProviderError, match="Gemini daily quota exhausted"):
        provider.generate_text(prompt="hello")

    assert provider.client.models.generate_content.call_count == 1


def test_gemini_provider_transient_rate_limit_retries_and_succeeds(monkeypatch):
    """Temporary per-minute rate limits (RPM) should be retried with backoff."""
    monkeypatch.setattr("time.sleep", lambda s: None)  # Avoid sleep delays in unit tests

    provider = GeminiProvider(api_key="dummy_mock_key", max_retries=3)
    provider.client = MagicMock()

    transient_error = Exception(
        "429 RESOURCE_EXHAUSTED: Rate limit exceeded for GenerateRequestsPerMinutePerProject"
    )
    mock_response = MagicMock()
    mock_response.text = '{"task_id": "test_rpm", "summary": "RPM test", "target_files": []}'

    # First attempt fails with transient per-minute limit; second attempt succeeds
    provider.client.models.generate_content.side_effect = [transient_error, mock_response]

    res = provider.generate_structured(prompt="hello", schema=SampleTaskPlan)
    assert res.task_id == "test_rpm"
    assert provider.client.models.generate_content.call_count == 2


def test_gemini_provider_transient_503_retries_and_succeeds(monkeypatch):
    """Confirms genuine transient 503 errors retry with backoff and succeed."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    provider = GeminiProvider(api_key="dummy_mock_key", max_retries=3)
    provider.client = MagicMock()

    service_unavailable = Exception("503 The model is overloaded. Please try again later.")
    mock_response = MagicMock()
    mock_response.text = "Hello world"

    provider.client.models.generate_content.side_effect = [service_unavailable, mock_response]

    text = provider.generate_text(prompt="hello")
    assert text == "Hello world"
    assert provider.client.models.generate_content.call_count == 2


def test_gemini_provider_quota_error_redacts_credentials():
    """Confirms sensitive API keys are never exposed in quota error messages."""
    fake_key = "AIzaSySecretKey98765432101234567890"
    provider = GeminiProvider(api_key=fake_key, max_retries=3)
    provider.client = MagicMock()

    error_with_key = Exception(
        f"429 RESOURCE_EXHAUSTED: Quota exceeded for GenerateRequestsPerDay with key={fake_key}"
    )
    provider.client.models.generate_content.side_effect = error_with_key

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate_text(prompt="test")

    err_msg = str(exc_info.value)
    assert fake_key not in err_msg
    assert "[REDACTED_SECRET]" in err_msg


def test_groq_missing_api_key_raises_error(monkeypatch):
    """Confirms error when GROQ_API_KEY is not configured."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMProviderError, match="GROQ_API_KEY is not configured"):
        GroqProvider(api_key="", model_name="llama-3.3-70b-versatile")


def test_groq_missing_model_raises_error(monkeypatch):
    """Confirms error when model is not configured."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_dummy_key")
    with pytest.raises(LLMProviderError, match="GROQ_MODEL is not configured"):
        GroqProvider(api_key="gsk_mock_dummy_key", model_name="")


def test_groq_mock_text_generation(monkeypatch):
    """Confirms successful text generation with Groq."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_dummy_key")
    provider = GroqProvider(api_key="gsk_mock_dummy_key", model_name="llama-3.3-70b-versatile")
    mock_choice = MagicMock()
    mock_choice.message.content = "def add(a, b): return a + b"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = mock_response

    res = provider.generate_text(prompt="Write add function")
    assert res == "def add(a, b): return a + b"


def test_groq_mock_structured_generation(monkeypatch):
    """Confirms successful structured generation with schema validation on Groq."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_dummy_key")
    provider = GroqProvider(api_key="gsk_mock_dummy_key", model_name="llama-3.3-70b-versatile")
    mock_choice = MagicMock()
    mock_choice.message.content = (
        '{"task_id": "groq_01", "summary": "Build Groq module", "target_files": ["groq.py"]}'
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = mock_response

    plan = provider.generate_structured(prompt="Create plan", schema=SampleTaskPlan)
    assert isinstance(plan, SampleTaskPlan)
    assert plan.task_id == "groq_01"
    assert plan.summary == "Build Groq module"
    assert plan.target_files == ["groq.py"]


def test_groq_provider_daily_quota_exhaustion_fails_fast_structured(monkeypatch):
    """Confirms Groq daily quota exhaustion fails immediately without retrying."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_dummy_key")
    provider = GroqProvider(
        api_key="gsk_mock_dummy_key", model_name="llama-3.3-70b-versatile", max_retries=5
    )
    provider.client = MagicMock()
    quota_err = Exception("429 Rate limit reached for model on requests per day (RPD)")
    provider.client.chat.completions.create.side_effect = quota_err

    with pytest.raises(LLMProviderError, match="Groq daily quota exhausted"):
        provider.generate_structured(prompt="hello", schema=SampleTaskPlan)

    assert provider.client.chat.completions.create.call_count == 1


def test_groq_provider_daily_quota_exhaustion_fails_fast_text(monkeypatch):
    """Confirms Groq daily quota exhaustion fails immediately in text generation."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_dummy_key")
    provider = GroqProvider(
        api_key="gsk_mock_dummy_key", model_name="llama-3.3-70b-versatile", max_retries=5
    )
    provider.client = MagicMock()
    quota_err = Exception("429 Rate limit reached: daily quota limit exceeded")
    provider.client.chat.completions.create.side_effect = quota_err

    with pytest.raises(LLMProviderError, match="Groq daily quota exhausted"):
        provider.generate_text(prompt="hello")

    assert provider.client.chat.completions.create.call_count == 1


def test_groq_provider_transient_rate_limit_retries_and_succeeds(monkeypatch):
    """Confirms Groq transient TPM rate limits retry with backoff and succeed."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_dummy_key")
    provider = GroqProvider(
        api_key="gsk_mock_dummy_key", model_name="llama-3.3-70b-versatile", max_retries=3
    )
    provider.client = MagicMock()

    transient_err = Exception(
        "429 Rate limit reached on tokens per minute (TPM). Please try again in 1.5s"
    )
    mock_choice = MagicMock()
    mock_choice.message.content = (
        '{"task_id": "tpm_01", "summary": "TPM retry", "target_files": []}'
    )
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    provider.client.chat.completions.create.side_effect = [transient_err, mock_resp]
    res = provider.generate_structured(prompt="hello", schema=SampleTaskPlan)
    assert res.task_id == "tpm_01"
    assert provider.client.chat.completions.create.call_count == 2


def test_groq_provider_transient_503_retries_and_succeeds(monkeypatch):
    """Confirms Groq transient 503 service unavailable retries and succeeds."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_dummy_key")
    provider = GroqProvider(
        api_key="gsk_mock_dummy_key", model_name="llama-3.3-70b-versatile", max_retries=3
    )
    provider.client = MagicMock()

    srv_err = Exception("503 Service Unavailable")
    mock_choice = MagicMock()
    mock_choice.message.content = "Recovered text"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    provider.client.chat.completions.create.side_effect = [srv_err, mock_resp]
    res = provider.generate_text(prompt="hello")
    assert res == "Recovered text"
    assert provider.client.chat.completions.create.call_count == 2


def test_groq_provider_quota_error_redacts_credentials():
    """Confirms sensitive Groq API keys are redacted from errors and logs."""
    fake_groq_key = "gsk_1234567890abcdefghijklmnopqrstuvwxyz12345678"
    provider = GroqProvider(
        api_key=fake_groq_key, model_name="llama-3.3-70b-versatile", max_retries=3
    )
    provider.client = MagicMock()

    err_with_key = Exception(f"429 Rate limit reached on requests per day with key={fake_groq_key}")
    provider.client.chat.completions.create.side_effect = err_with_key

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate_text(prompt="hello")

    err_msg = str(exc_info.value)
    assert fake_groq_key not in err_msg
    assert "[REDACTED_SECRET]" in err_msg


def test_get_llm_provider_wires_groq():
    """Confirms provider factory wires Groq correctly when configured."""
    config = AppConfig()
    config.ai.provider = "groq"
    config.ai.groq_api_key = "gsk_dummy_mock_key"
    config.ai.groq_model = "llama-3.3-70b-versatile"
    provider = get_llm_provider(config)
    assert isinstance(provider, GroqProvider)
    assert provider.model_name == "llama-3.3-70b-versatile"


def test_is_llm_configured_handles_gemini_and_groq():
    """Confirms is_llm_configured dynamically evaluates the active provider."""
    cfg = AppConfig()
    cfg.ai.provider = "gemini"
    cfg.ai.gemini_api_key = "dummy_key"
    assert cfg.is_llm_configured() is True

    cfg.ai.gemini_api_key = None
    assert cfg.is_llm_configured() is False

    cfg.ai.provider = "groq"
    cfg.ai.groq_api_key = "gsk_dummy"
    assert cfg.is_llm_configured() is True

    cfg.ai.groq_api_key = None
    assert cfg.is_llm_configured() is False
