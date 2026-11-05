from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.infrastructure.telemetry.metrics import get_metrics
from src.interfaces.api.middleware import CorrelationAndTelemetryMiddleware
from src.interfaces.api.v1.health import router as health_router
from src.interfaces.api.v1.search import router as search_router
from src.interfaces.api.v1.webhook import router as webhook_router


def create_app() -> FastAPI:
    """Crea y configura la aplicación FastAPI para LangChain Radar."""
    app = FastAPI(
        title="LangChain Radar API",
        description="Pipeline RAG de Ingesta Continua y Búsqueda Semántica Híbrida",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(CorrelationAndTelemetryMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        # Con origen comodín no deben permitirse credenciales (cookies / Authorization)
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Registrar routers
    app.include_router(webhook_router)
    app.include_router(search_router)
    app.include_router(health_router)

    @app.get(
        "/metrics",
        response_class=PlainTextResponse,
        include_in_schema=True,
        tags=["Observability"],
        summary="Métricas globales Prometheus",
    )
    async def root_metrics() -> PlainTextResponse:
        metrics = get_metrics()
        return PlainTextResponse(
            content=metrics.generate_prometheus_text(),
            media_type="text/plain; version=0.0.4",
        )

    return app


app = create_app()
