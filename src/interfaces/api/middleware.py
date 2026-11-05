import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.infrastructure.telemetry.metrics import get_metrics


def _route_template(request: Request) -> str:
    """Etiqueta de ruta de baja cardinalidad (plantilla, no la URL concreta)."""
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    return route_path if isinstance(route_path, str) else "unmatched"


class CorrelationAndTelemetryMiddleware(BaseHTTPMiddleware):
    """Middleware que asigna un Correlation-ID a cada petición y mide latencias para Prometheus."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        metrics = get_metrics()
        method = request.method
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception:
            path = _route_template(request)
            metrics.inc_counter(
                "http_requests_total",
                labels={"method": method, "path": path, "status": "500"},
            )
            raise

        elapsed = time.perf_counter() - start_time
        path = _route_template(request)
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time-Seconds"] = f"{elapsed:.4f}"

        metrics.inc_counter(
            "http_requests_total",
            labels={"method": method, "path": path, "status": status_code},
        )
        metrics.observe_histogram(
            "http_request_duration_seconds",
            elapsed,
            labels={"method": method, "path": path},
        )

        return response
