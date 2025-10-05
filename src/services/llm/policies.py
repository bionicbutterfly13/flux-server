"""Policy loader for multi-provider LLM orchestration (Flux Server)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

_POLICY_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "llm_policies.yaml"


class PolicyNotFoundError(KeyError):
    """Raised when a routing policy is missing."""


@dataclass
class LLMPolicy:
    """Declarative routing policy for a given task type."""

    task_type: str
    primary_provider: str
    fallback_providers: List[str] = field(default_factory=list)
    timeout_ms: int = 10_000
    retry_limit: int = 1

    def copy(self) -> "LLMPolicy":
        return LLMPolicy(
            task_type=self.task_type,
            primary_provider=self.primary_provider,
            fallback_providers=list(self.fallback_providers),
            timeout_ms=self.timeout_ms,
            retry_limit=self.retry_limit,
        )


def _default_policies() -> Dict[str, LLMPolicy]:
    return {
        "chat": LLMPolicy(
            task_type="chat",
            primary_provider="ollama.deepseek-r1",
            fallback_providers=["openai.gpt-4o", "anthropic.claude-3-sonnet"],
            timeout_ms=12_000,
            retry_limit=2,
        ),
        "query_rewrite": LLMPolicy(
            task_type="query_rewrite",
            primary_provider="ollama.deepseek-r1",
            fallback_providers=["openai.gpt-4o"],
            timeout_ms=8_000,
            retry_limit=1,
        ),
        "analysis": LLMPolicy(
            task_type="analysis",
            primary_provider="ollama.qwen2.5-14b",
            fallback_providers=["openai.gpt-4o", "anthropic.claude-3-sonnet"],
            timeout_ms=15_000,
            retry_limit=2,
        ),
    }


def _load_policy_file() -> Dict[str, LLMPolicy]:
    if not _POLICY_FILE.exists():
        return {}

    with open(_POLICY_FILE, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    policies: Dict[str, LLMPolicy] = {}
    for item in data.get("policies", []):
        policy = LLMPolicy(
            task_type=item["task_type"],
            primary_provider=item["primary_provider"],
            fallback_providers=item.get("fallback_providers", []),
            timeout_ms=item.get("timeout_ms", 10_000),
            retry_limit=item.get("retry_limit", 1),
        )
        policies[policy.task_type] = policy
    return policies


def load_policies() -> Dict[str, LLMPolicy]:
    base = _default_policies()
    overrides = _load_policy_file()
    base.update({key: value for key, value in overrides.items()})
    return {key: policy.copy() for key, policy in base.items()}


def get_policy(task_type: str) -> LLMPolicy:
    policies = load_policies()
    policy = policies.get(task_type)
    if policy is None:
        raise PolicyNotFoundError(task_type)
    return policy

