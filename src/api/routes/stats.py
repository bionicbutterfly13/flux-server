"""Stats endpoints for Flux Server telemetry."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter

from src.services.llm import LLMProvider, ProviderMetrics, load_provider_registry, get_telemetry_store

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/llm/providers")
async def get_llm_provider_metrics() -> Dict[str, Any]:
    """Expose provider telemetry for dashboards and tooling."""

    registry = load_provider_registry()
    telemetry_store = get_telemetry_store()
    metrics_map = telemetry_store.get_all_metrics()

    providers: List[Dict[str, Any]] = []
    for name, provider in registry.items():
        metrics: ProviderMetrics = metrics_map.get(name, ProviderMetrics(provider=name))
        providers.append({
            "provider": name,
            "supports_streaming": provider.supports_streaming,
            "max_context_tokens": provider.max_context_tokens,
            "cost_per_1k_tokens": provider.cost_per_1k_tokens,
            "preferred_tasks": provider.preferred_tasks,
            "total_calls": metrics.total_calls,
            "successes": metrics.successes,
            "failures": metrics.failures,
            "success_rate": metrics.success_rate,
            "average_latency_ms": metrics.average_latency_ms,
            "total_cost": metrics.total_cost,
            "last_error": metrics.last_error,
            "last_updated": metrics.last_updated,
        })

    return {
        "providers": providers,
        "timestamp": datetime.utcnow().isoformat(),
    }
