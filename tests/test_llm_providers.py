"""
test_llm_providers.py — tests the provider auto-detection logic (not live
API calls, which need real credentials). Run with: pytest tests/test_llm_providers.py -v
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_providers import detect_provider_name, get_provider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no real credentials leak into these tests either way."""
    for var in ["LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "REPLY_GEN_MODEL"]:
        monkeypatch.delenv(var, raising=False)


def test_no_credentials_means_mock_mode(monkeypatch):
    assert detect_provider_name() is None
    assert get_provider() is None


def test_anthropic_key_detected(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-detection-test-only")
    assert detect_provider_name() == "anthropic"


def test_openai_key_detected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-detection-test-only")
    assert detect_provider_name() == "openai"


def test_generic_openai_compatible_key_detected(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake-key-for-detection-test-only")
    assert detect_provider_name() == "openai_compatible"


def test_explicit_provider_override_wins(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert detect_provider_name() == "openai"


def test_priority_order_anthropic_before_openai(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    assert detect_provider_name() == "anthropic"


def test_openai_compatible_without_base_url_raises_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        get_provider()


def test_openai_compatible_with_base_url_constructs_client(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake-key-for-detection-test-only")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    provider = get_provider()
    assert provider.name == "openai_compatible"
    assert provider.model  # a default model was chosen since REPLY_GEN_MODEL wasn't set
