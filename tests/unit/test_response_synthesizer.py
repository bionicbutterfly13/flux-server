import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.query import Query  # noqa: E402
from models.response import SearchResult, SearchSource  # noqa: E402
from services.response_synthesizer import (  # noqa: E402
    AnswerGenerationError,
    PolicyRoutedAnswerGenerator,
    ResponseSynthesizer,
)


class DummyRouter:
    def __init__(self, provider_name: str):
        from services.llm.provider_registry import LLMProvider
        from services.llm.policies import LLMPolicy
        from services.llm.router import ProviderSelection

        provider = LLMProvider(
            name=provider_name,
            supports_streaming=True,
            max_context_tokens=8192,
            cost_per_1k_tokens=0.0,
            preferred_tasks=["analysis"],
        )
        policy = LLMPolicy(
            task_type="chat",
            primary_provider=provider_name,
            fallback_providers=[],
            timeout_ms=10_000,
            retry_limit=1,
        )
        self.selection = ProviderSelection(
            provider=provider,
            policy=policy,
            attempted_providers={provider_name: "selected"},
        )

    def select(self, task_type: str):  # pragma: no cover - simple stub
        assert task_type == "chat"
        return self.selection


class DummyOllamaManager:
    def __init__(self):
        self.calls = []

    async def generate_text(self, model_name: str, prompt: str, max_tokens: int = 768):  # pragma: no cover
        self.calls.append((model_name, prompt, max_tokens))
        return {"success": True, "response": "Generated answer with citation [1]."}


class DummyTelemetry:
    def __init__(self):
        self.success_records = []
        self.failure_records = []

    def record_success(self, provider: str, latency_ms: float, cost: float = 0.0):  # pragma: no cover
        self.success_records.append((provider, latency_ms, cost))

    def record_failure(self, provider: str, latency_ms: float, error: str, cost: float = 0.0):  # pragma: no cover
        self.failure_records.append((provider, latency_ms, error, cost))


@pytest.mark.asyncio
async def test_policy_routed_generator_records_success():
    router = DummyRouter("ollama.qwen2.5-14b")
    manager = DummyOllamaManager()
    telemetry = DummyTelemetry()
    generator = PolicyRoutedAnswerGenerator(router=router, ollama_manager=manager, telemetry_store=telemetry)

    sources = [
        SearchResult(
            source=SearchSource.NEO4J,
            content="Basins stabilise consciousness across iterative updates.",
            relevance_score=0.9,
            metadata={"title": "Consciousness Basins"},
            relationships=["STABILISES"],
        )
    ]

    result = await generator.generate("Explain consciousness basins", sources)

    assert manager.calls
    assert "[1]" in result
    assert telemetry.success_records
    provider_name, latency_ms, cost = telemetry.success_records[0]
    assert provider_name == "ollama.qwen2.5-14b"
    assert cost >= 0


@pytest.mark.asyncio
async def test_policy_routed_generator_records_failure():
    router = DummyRouter("openai.gpt-4o")
    manager = DummyOllamaManager()
    telemetry = DummyTelemetry()
    generator = PolicyRoutedAnswerGenerator(router=router, ollama_manager=manager, telemetry_store=telemetry)

    sources = [
        SearchResult(
            source=SearchSource.NEO4J,
            content="Basins stabilise consciousness across iterative updates.",
            relevance_score=0.9,
            metadata={"title": "Consciousness Basins"},
            relationships=["STABILISES"],
        )
    ]

    with pytest.raises(AnswerGenerationError):
        await generator.generate("Explain consciousness basins", sources)

    assert telemetry.failure_records
    provider_name, latency_ms, error, _ = telemetry.failure_records[0]
    assert provider_name == "openai.gpt-4o"
    assert "not yet supported" in error
