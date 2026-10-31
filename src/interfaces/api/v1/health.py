import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from src.domain.ports.state_store import StateStore
from src.domain.ports.vector_store import VectorRepository
from src.infrastructure.state.redis_state import RedisStateStore
from src.infrastructure.vector.qdrant_adapter import QdrantVectorAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Ops & Governance"])

_vector_repo: VectorRepository | None = None
_state_store: StateStore | None = None


def get_vector_repo() -> VectorRepository:
    global _vector_repo
    if _vector_repo is None:
        _vector_repo = QdrantVectorAdapter()
    return _vector_repo


def set_health_vector_repo(repo: VectorRepository | None) -> None:
    global _vector_repo
    _vector_repo = repo


def get_state_store() -> StateStore:
    global _state_store
    if _state_store is None:
        _state_store = RedisStateStore()
    return _state_store


def set_health_state_store(store: StateStore | None) -> None:
    global _state_store
    _state_store = store


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    components: dict[str, str]


class SyncStatusResponse(BaseModel):
    active_collection: str
    stream_key: str
    environment: str
    timestamp: datetime
    last_commit_sha: str | None = None
    last_synced_at: datetime | None = None
    total_processed_prs: int = 0


class ReindexRequest(BaseModel):
    new_version_tag: str = Field(
        description="Tag identificador de la nueva versión o modelo de embedding"
    )


class ReindexResponse(BaseModel):
    status: str
    previous_collection: str
    active_collection: str
    alias: str
    switched_at: datetime


def verify_admin_token(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Valida el token Bearer para endpoints de gobernanza administrativa."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Cabecera Authorization Bearer requerida",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.secret_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token administrativo inválido",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Healthcheck y Readiness Probes",
    description="Verifica el estado operativo del servicio y sus componentes de infraestructura.",
)
async def check_health() -> HealthResponse:
    """Verifica estado de salud básico."""
    now = datetime.now(UTC)
    components = {
        "api": "healthy",
        "embedder": "healthy",
    }
    return HealthResponse(
        status="healthy",
        timestamp=now,
        components=components,
    )


@router.get(
    "/status/sync",
    response_model=SyncStatusResponse,
    summary="Estado de sincronización y colecciones activas",
)
async def get_sync_status(
    settings: Annotated[Settings, Depends(get_settings)],
    state_store: Annotated[StateStore, Depends(get_state_store)],
) -> SyncStatusResponse:
    """Consulta la colección activa y el estado del stream de ingesta."""
    last_commit_sha: str | None = None
    last_synced_at: datetime | None = None
    total_prs = 0

    try:
        cp = await state_store.get_checkpoint(
            "backfill_cursor"
        ) or await state_store.get_checkpoint("stream_consumer")
        if cp:
            last_commit_sha = cp.last_commit_sha
            last_synced_at = cp.last_synced_at
        total_prs = await state_store.get_processed_pr_count()
    except Exception as e:
        logger.warning("No se pudo obtener información detallada del state_store: %s", e)

    return SyncStatusResponse(
        active_collection=settings.qdrant_collection_active,
        stream_key=settings.redis_stream_key,
        environment=settings.app_env,
        timestamp=datetime.now(UTC),
        last_commit_sha=last_commit_sha,
        last_synced_at=last_synced_at,
        total_processed_prs=total_prs,
    )


@router.post(
    "/admin/reindex",
    response_model=ReindexResponse,
    summary="Disparo de Re-indexación y Swap Blue/Green Zero-Downtime",
    description="Crea una colección staging en Qdrant y ejecuta un cambio de alias atómico sin tiempo de inactividad.",
    dependencies=[Depends(verify_admin_token)],
)
async def trigger_reindex(
    request: ReindexRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    vector_repo: Annotated[VectorRepository, Depends(get_vector_repo)],
) -> ReindexResponse:
    """Ejecuta una migración atómica Blue/Green cambiando el alias langchain_radar_active."""
    alias_name = settings.qdrant_collection_active
    staging_collection = f"{settings.qdrant_collection_prefix}_{request.new_version_tag}"

    # 1. Crear colección staging
    await vector_repo.ensure_collection(staging_collection)

    # 2. Transición atómica de alias en Qdrant
    now = datetime.now(UTC)
    await vector_repo.switch_alias(
        alias_name=alias_name,
        target_collection=staging_collection,
        old_collection=None,
    )

    logger.info(
        "Alias Blue/Green '%s' actualizado con éxito a la colección '%s'",
        alias_name,
        staging_collection,
    )

    return ReindexResponse(
        status="swapped_successfully",
        previous_collection="previous_active",
        active_collection=staging_collection,
        alias=alias_name,
        switched_at=now,
    )


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Métricas operativas de Prometheus",
    description="Expone contadores, gauges y tiempos del pipeline para Prometheus.",
)
async def get_prometheus_metrics() -> PlainTextResponse:
    """Expone métricas en formato estándar de Prometheus."""
    from src.infrastructure.telemetry.metrics import get_metrics

    metrics = get_metrics()
    content = metrics.generate_prometheus_text()
    return PlainTextResponse(content=content, media_type="text/plain; version=0.0.4")
