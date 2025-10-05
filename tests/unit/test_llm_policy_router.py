import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from services.llm import policies, provider_registry, router  # noqa: E402


def test_router_selects_primary_provider():
    policy_module = importlib.reload(policies)
    router_module = importlib.reload(router)

    selection = router_module.ProviderRouter(policy_module.load_policies()).select("chat")

    assert selection.provider.name == "ollama.deepseek-r1"
    assert selection.attempted_providers["ollama.deepseek-r1"] == "selected"


def test_router_falls_back_when_primary_missing(monkeypatch):
    monkeypatch.setitem(provider_registry._BASE_REGISTRY, "ollama.deepseek-r1", None)  # type: ignore[attr-defined]

    policy_module = importlib.reload(policies)
    router_module = importlib.reload(router)

    selection = router_module.ProviderRouter(policy_module.load_policies()).select("chat")
    assert selection.provider.name == "openai.gpt-4o"
    assert selection.attempted_providers["ollama.deepseek-r1"] == "missing"


def test_router_refresh_reloads_policy(tmp_path, monkeypatch):
    policy_file = tmp_path / "llm_policies.yaml"
    policy_file.write_text(
        """
        policies:
          - task_type: chat
            primary_provider: anthropic.claude-3-sonnet
            fallback_providers: []
            timeout_ms: 5000
            retry_limit: 1
        """
    )

    monkeypatch.setattr(policies, "_POLICY_FILE", policy_file)
    router_instance = router.ProviderRouter()
    router_instance.refresh()

    selection = router_instance.select("chat")
    assert selection.provider.name == "anthropic.claude-3-sonnet"
    assert selection.policy.timeout_ms == 5000
