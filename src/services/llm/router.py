"""Routing logic for multi-provider LLM orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .provider_registry import LLMProvider, ProviderNotFoundError, get_provider
from .policies import LLMPolicy, PolicyNotFoundError, load_policies


@dataclass
class ProviderSelection:
    """Result of choosing a provider for a task."""

    provider: LLMProvider
    policy: LLMPolicy
    attempted_providers: Dict[str, str]


class ProviderRouter:
    """Determines which provider should handle a task based on policies."""

    def __init__(self, policies: Optional[Dict[str, LLMPolicy]] = None) -> None:
        self._policies = policies or load_policies()

    def select(self, task_type: str) -> ProviderSelection:
        policy = self._policies.get(task_type)
        if policy is None:
            raise PolicyNotFoundError(task_type)

        attempted: Dict[str, str] = {}
        candidates = [policy.primary_provider, *policy.fallback_providers]

        for candidate in candidates:
            try:
                provider = get_provider(candidate)
                attempted[candidate] = "selected"
                return ProviderSelection(provider=provider, policy=policy.copy(), attempted_providers=attempted)
            except ProviderNotFoundError:
                attempted[candidate] = "missing"
                continue

        raise ProviderNotFoundError(
            f"No valid providers available for task {task_type} (checked: {attempted})"
        )

    def refresh(self) -> None:
        """Reload policies from disk."""
        self._policies = load_policies()

