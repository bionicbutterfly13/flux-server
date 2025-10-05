import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from services.llm import provider_registry  # noqa: E402


REQUIRED = {
    "ollama.deepseek-r1": {
        "supports_streaming": True,
        "max_context_tokens": 8192,
        "cost_per_1k_tokens": 0.0,
    },
    "ollama.qwen2.5-14b": {
        "supports_streaming": True,
        "max_context_tokens": 8192,
        "cost_per_1k_tokens": 0.0,
    },
    "openai.gpt-4o": {
        "supports_streaming": True,
        "max_context_tokens": 128_000,
        "cost_per_1k_tokens": 0.01,
    },
    "anthropic.claude-3-sonnet": {
        "supports_streaming": True,
        "max_context_tokens": 200_000,
        "cost_per_1k_tokens": 0.008,
    },
}


def test_registry_contains_required_providers():
    registry = provider_registry.load_provider_registry()
    for name, expected in REQUIRED.items():
        provider = registry.get(name)
        assert provider is not None
        assert provider.supports_streaming is expected["supports_streaming"]
        assert provider.max_context_tokens == expected["max_context_tokens"]
        assert pytest.approx(provider.cost_per_1k_tokens, rel=1e-6) == expected["cost_per_1k_tokens"]


def test_get_provider_returns_copy():
    provider = provider_registry.get_provider("ollama.deepseek-r1")
    provider.preferred_tasks.append("__test__")
    fresh = provider_registry.get_provider("ollama.deepseek-r1")
    assert "__test__" not in fresh.preferred_tasks


def test_reload_does_not_require_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reloaded = importlib.reload(provider_registry)
    assert "openai.gpt-4o" in reloaded.load_provider_registry()


def test_unknown_provider_raises():
    with pytest.raises(provider_registry.ProviderNotFoundError):
        provider_registry.get_provider("does-not-exist")
