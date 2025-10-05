"""Static provider registry used by the multi-provider LLM orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


class ProviderNotFoundError(KeyError):
    """Raised when a requested provider is not present in the registry."""


@dataclass
class LLMProvider:
    """Capability metadata for a single LLM provider."""

    name: str
    supports_streaming: bool
    max_context_tokens: int
    cost_per_1k_tokens: float
    preferred_tasks: List[str] = field(default_factory=list)

    def copy(self) -> "LLMProvider":
        """Return a deep copy suitable for safe mutation by callers."""
        return LLMProvider(
            name=self.name,
            supports_streaming=self.supports_streaming,
            max_context_tokens=self.max_context_tokens,
            cost_per_1k_tokens=self.cost_per_1k_tokens,
            preferred_tasks=list(self.preferred_tasks),
        )


_BASE_REGISTRY: Dict[str, LLMProvider] = {
    "ollama.deepseek-r1": LLMProvider(
        name="ollama.deepseek-r1",
        supports_streaming=True,
        max_context_tokens=8192,
        cost_per_1k_tokens=0.0,
        preferred_tasks=["analysis", "rewrite", "synthesis"],
    ),
    "ollama.qwen2.5-14b": LLMProvider(
        name="ollama.qwen2.5-14b",
        supports_streaming=True,
        max_context_tokens=8192,
        cost_per_1k_tokens=0.0,
        preferred_tasks=["analysis", "rewrite", "synthesis"],
    ),
    "openai.gpt-4o": LLMProvider(
        name="openai.gpt-4o",
        supports_streaming=True,
        max_context_tokens=128_000,
        cost_per_1k_tokens=0.01,
        preferred_tasks=["analysis", "rewrite", "synthesis"],
    ),
    "anthropic.claude-3-sonnet": LLMProvider(
        name="anthropic.claude-3-sonnet",
        supports_streaming=True,
        max_context_tokens=200_000,
        cost_per_1k_tokens=0.008,
        preferred_tasks=["analysis", "rewrite", "synthesis"],
    ),
}


def _clone_registry(providers: Iterable[LLMProvider]) -> Dict[str, LLMProvider]:
    """Create a fresh mapping of provider name to a safe copy."""
    return {provider.name: provider.copy() for provider in providers}


def load_provider_registry() -> Dict[str, LLMProvider]:
    """Return a fully cloned provider registry."""
    return _clone_registry(_BASE_REGISTRY.values())


def get_provider(name: str) -> LLMProvider:
    """Return a copy of the requested provider."""
    provider = _BASE_REGISTRY.get(name)
    if provider is None:
        raise ProviderNotFoundError(name)
    return provider.copy()


def registry_keys() -> List[str]:  # pragma: no cover - helper
    """List names of available providers."""
    return list(_BASE_REGISTRY.keys())
