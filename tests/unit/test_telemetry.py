from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.infrastructure.telemetry.metrics import PrometheusMetrics
from src.interfaces.api.middleware import CorrelationAndTelemetryMiddleware


def test_prometheus_metrics_counter_gauge_and_histogram() -> None:
    metrics = PrometheusMetrics()

    # Counter
    metrics.inc_counter("test_counter", 1.0, labels={"status": "200"})
    metrics.inc_counter("test_counter", 2.0, labels={"status": "200"})

    # Gauge
    metrics.set_gauge("test_gauge", 42.0, labels={"collection": "radar_active"})

    # Histogram
    with metrics.measure_time("test_histogram", labels={"phase": "embed"}):
        _ = 1 + 1

    text = metrics.generate_prometheus_text()
    assert 'test_counter{status="200"} 3.0' in text
    assert 'test_gauge{collection="radar_active"} 42.0' in text
    assert 'test_histogram_count{phase="embed"} 1' in text
    assert 'test_histogram_sum{phase="embed"}' in text


def test_correlation_and_telemetry_middleware() -> None:
    app = FastAPI()
    app.add_middleware(CorrelationAndTelemetryMiddleware)

    @app.get("/test-endpoint")
    def sample_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)

    # 1. Petición sin Correlation-ID previo (debe generarlo)
    resp = client.get("/test-endpoint")
    assert resp.status_code == 200
    assert "X-Correlation-ID" in resp.headers
    assert "X-Response-Time-Seconds" in resp.headers

    # 2. Petición con Correlation-ID explícito
    custom_id = "custom-trace-uuid-1234"
    resp2 = client.get("/test-endpoint", headers={"X-Correlation-ID": custom_id})
    assert resp2.headers["X-Correlation-ID"] == custom_id


def test_root_metrics_endpoint() -> None:
    from src.interfaces.api.app import app

    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "# HELP" in resp.text


def test_middleware_uses_route_template_for_path_label() -> None:
    from src.infrastructure.telemetry.metrics import get_metrics

    app = FastAPI()
    app.add_middleware(CorrelationAndTelemetryMiddleware)

    @app.get("/items/{item_id}")
    def get_item(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    client = TestClient(app)
    client.get("/items/abc")
    client.get("/items/xyz")

    text = get_metrics().generate_prometheus_text()
    assert 'path="/items/{item_id}"' in text
    assert 'path="/items/abc"' not in text
    assert "# TYPE http_request_duration_seconds summary" in text
