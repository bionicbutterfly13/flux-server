"""Telemetry collection for LLM providers."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Optional

try:  # Optional dependency
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class ProviderMetrics:
    """Aggregate metrics for a single provider."""

    provider: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    last_error: Optional[str] = None
    last_updated: Optional[str] = None

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successes / self.total_calls

    @property
    def average_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls


class TelemetryStore:
    """Persist provider telemetry in Redis with in-memory fallback."""

    def __init__(self, redis_client: Optional["redis.Redis"] = None, namespace: str = "llm:telemetry") -> None:
        self.redis_client = redis_client
        self.namespace = namespace
        self._cache: Dict[str, ProviderMetrics] = {}

    def _key(self, provider: str) -> str:
        return f"{self.namespace}:{provider}"

    def _load_from_cache(self, provider: str) -> ProviderMetrics:
        if provider in self._cache:
            return self._cache[provider]

        metrics = ProviderMetrics(provider=provider)

        if self.redis_client is not None:
            try:
                data = self.redis_client.get(self._key(provider))
                if data:
                    payload = json.loads(data)
                    metrics = ProviderMetrics(**payload)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load telemetry for %s: %s", provider, exc)

        self._cache[provider] = metrics
        return metrics

    def _persist(self, metrics: ProviderMetrics) -> None:
        self._cache[metrics.provider] = metrics

        if self.redis_client is not None:
            try:
                self.redis_client.set(self._key(metrics.provider), json.dumps(asdict(metrics)))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to persist telemetry for %s: %s", metrics.provider, exc)

    def record_success(self, provider: str, latency_ms: float, cost: float = 0.0) -> ProviderMetrics:
        metrics = self._load_from_cache(provider)
        metrics.total_calls += 1
        metrics.successes += 1
        metrics.total_latency_ms += latency_ms
        metrics.total_cost += cost
        metrics.last_error = None
        metrics.last_updated = datetime.utcnow().isoformat()
        self._persist(metrics)
        return metrics

    def record_failure(self, provider: str, latency_ms: float, error: str, cost: float = 0.0) -> ProviderMetrics:
        metrics = self._load_from_cache(provider)
        metrics.total_calls += 1
        metrics.failures += 1
        metrics.total_latency_ms += latency_ms
        metrics.total_cost += cost
        metrics.last_error = error
        metrics.last_updated = datetime.utcnow().isoformat()
        self._persist(metrics)
        return metrics

    def get_metrics(self, provider: str) -> ProviderMetrics:
        return self._load_from_cache(provider)

    def get_all_metrics(self) -> Dict[str, ProviderMetrics]:
        if self.redis_client is not None:
            try:
                keys = [key.decode("utf-8") for key in self.redis_client.keys(f"{self.namespace}:*")]
                for key in keys:
                    provider = key.split(":", 1)[-1]
                    self._load_from_cache(provider)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to enumerate telemetry keys: %s", exc)
        return dict(self._cache)


_default_store: Optional[TelemetryStore] = None


def get_telemetry_store() -> TelemetryStore:
    """Return global telemetry store singleton."""

    global _default_store
    if _default_store is None:
        redis_client = None
        if redis is not None:
            try:
                redis_client = redis.Redis(host="localhost", port=6379, decode_responses=False)
                redis_client.ping()
            except Exception:
                redis_client = None
        _default_store = TelemetryStore(redis_client=redis_client)
    return _default_store


__all__ = [
    "ProviderMetrics",
    "TelemetryStore",
    "get_telemetry_store",
]
