import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.routes import stats as stats_module  # noqa: E402
from services.llm.telemetry import ProviderMetrics  # noqa: E402


class StubTelemetry:
    def __init__(self):
        self._metrics = {
            "ollama.deepseek-r1": ProviderMetrics(
                provider="ollama.deepseek-r1",
                total_calls=5,
                successes=4,
                failures=1,
                total_latency_ms=450.0,
                total_cost=0.1,
                last_error="timeout",
                last_updated="2025-10-07T16:55:00Z",
            )
        }

    def get_all_metrics(self):  # pragma: no cover - simple stub
        return dict(self._metrics)


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(stats_module.router)

    from services.llm import provider_registry

    monkeypatch.setattr(stats_module, "get_telemetry_store", lambda: StubTelemetry())
    monkeypatch.setattr(
        stats_module,
        "load_provider_registry",
        lambda: provider_registry.load_provider_registry(),
    )

    return TestClient(app)


def test_llm_provider_metrics(client):
    response = client.get("/api/stats/llm/providers")
    assert response.status_code == 200
    payload = response.json()
    providers = {entry["provider"]: entry for entry in payload["providers"]}
    assert "ollama.deepseek-r1" in providers
    metrics = providers["ollama.deepseek-r1"]
    assert metrics["total_calls"] == 5
    assert metrics["success_rate"] == pytest.approx(0.8)
