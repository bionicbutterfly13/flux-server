import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from services.llm.telemetry import ProviderMetrics, TelemetryStore  # noqa: E402


def test_record_success_updates_metrics():
    store = TelemetryStore(redis_client=None, namespace="test:telemetry")
    metrics = store.record_success("ollama.deepseek-r1", latency_ms=100.0, cost=0.05)

    assert metrics.total_calls == 1
    assert metrics.successes == 1
    assert metrics.failures == 0
    assert metrics.average_latency_ms == 100.0
    assert metrics.total_cost == 0.05
    assert metrics.success_rate == 1.0


def test_record_failure_tracks_error():
    store = TelemetryStore(redis_client=None, namespace="test:telemetry")
    store.record_failure("ollama.deepseek-r1", latency_ms=80.0, error="timeout")
    metrics = store.get_metrics("ollama.deepseek-r1")

    assert metrics.failures == 1
    assert metrics.last_error == "timeout"
    assert metrics.total_calls == 1


def test_get_all_metrics_returns_cache():
    store = TelemetryStore(redis_client=None, namespace="test:telemetry")
    store.record_success("ollama.qwen2.5-14b", latency_ms=90.0)
    store.record_failure("ollama.qwen2.5-14b", latency_ms=120.0, error="rate limit")

    metrics_map = store.get_all_metrics()
    metrics = metrics_map["ollama.qwen2.5-14b"]
    assert metrics.total_calls == 2
    assert metrics.successes == 1
    assert metrics.failures == 1
