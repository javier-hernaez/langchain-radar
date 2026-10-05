from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SearchFilters(BaseModel):
    """Filtros estructurados para la búsqueda semántica e híbrida."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    module_prefix: str | None = Field(
        default=None, description="Prefijo del módulo a filtrar, ej. langchain_core"
    )
    semver_major: int | None = Field(default=None, ge=0)
    semver_minor: int | None = Field(default=None, ge=0)
    is_deprecated: bool | None = Field(default=None)
    since_date: datetime | None = Field(default=None)
    entity_type: str | None = Field(default=None)


class SearchQuery(BaseModel):
    """Consulta de búsqueda entrante."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(
        min_length=2, description="Texto de la consulta en lenguaje natural o código"
    )
    limit: int = Field(
        default=5, ge=1, le=50, description="Número de resultados finales a devolver"
    )
    filters: SearchFilters = Field(default_factory=SearchFilters)
    search_mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="hybrid", description="Estrategia de recuperación en Qdrant"
    )
    use_reranker: bool = Field(
        default=True, description="Indica si se aplica Cross-Encoder Re-ranking local"
    )
    include_parent_context: bool = Field(
        default=True, description="Si es True, inyecta el esqueleto padre de la clase"
    )
    include_diffs: bool = Field(default=False, description="Si incluye hunks de cambios de git")


class SearchResultItem(BaseModel):
    """Elemento individual de resultado recuperado y re-rankeado."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: UUID
    parent_chunk_id: UUID | None = None
    rerank_score: float | None = None
    hybrid_score: float
    entity_name: str
    canonical_import: str
    file_path: str
    release_version: str
    last_updated: datetime
    parent_context: str | None = None
    content: str
    author: str | None = None
    pr_number: int | None = None
    is_deprecated: bool = False
    breaking_change: bool = False


class SearchTimings(BaseModel):
    """Métricas de latencia por etapa de búsqueda en milisegundos."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hybrid_retrieval_ms: float
    cross_encoder_rerank_ms: float = 0.0


class SearchResponse(BaseModel):
    """Respuesta consolidada de búsqueda."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    took_ms: float
    timings: SearchTimings
    total_hits: int
    results: tuple[SearchResultItem, ...]
