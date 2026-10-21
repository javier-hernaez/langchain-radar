import logging
import time

from config.settings import Settings, get_settings
from src.domain.models.search import (
    SearchQuery,
    SearchResponse,
    SearchResultItem,
    SearchTimings,
)
from src.domain.ports.embedder import EmbeddingEngine
from src.domain.ports.reranker import RerankerEngine
from src.domain.ports.vector_store import VectorRepository

logger = logging.getLogger(__name__)


class SearchCodeUseCase:
    """Caso de uso para búsqueda semántica híbrida (First-Stage) y Re-ranking local (Second-Stage)."""

    def __init__(
        self,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        reranker_engine: RerankerEngine,
        settings: Settings | None = None,
    ) -> None:
        self._vector_repo = vector_repo
        self._embedding_engine = embedding_engine
        self._reranker_engine = reranker_engine
        self._settings = settings or get_settings()

    async def execute(
        self, query: SearchQuery, collection_name: str | None = None
    ) -> SearchResponse:
        """Ejecuta el pipeline de recuperación híbrida en dos etapas con enriquecimiento contextual."""
        overall_start = time.monotonic()
        target_collection = collection_name or self._settings.qdrant_collection_active

        # -----------------------------------------------------------------
        # Etapa 1: Embeddings y Búsqueda Híbrida en Qdrant (First-Stage)
        # -----------------------------------------------------------------
        stage1_start = time.monotonic()

        # Generar embedding denso
        dense_vecs = self._embedding_engine.embed_dense([query.query])
        dense_vector = dense_vecs[0] if dense_vecs else [0.0] * self._embedding_engine.dimension

        sparse_indices: list[int] | None = None
        sparse_values: list[float] | None = None

        if query.search_mode == "hybrid":
            sparse_vecs = self._embedding_engine.embed_sparse([query.query])
            if sparse_vecs:
                sparse_indices, sparse_values = sparse_vecs[0]

        # Extraer Top-25 candidatos (o proporcional al límite solicitado)
        candidate_limit = max(25, query.limit * 5)
        candidates = await self._vector_repo.search_hybrid(
            collection_name=target_collection,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            limit=candidate_limit,
            filters=query.filters,
        )

        stage1_duration_ms = (time.monotonic() - stage1_start) * 1000.0

        # -----------------------------------------------------------------
        # Etapa 2: Re-ranking Semántico Profundo con Cross-Encoder (Second-Stage)
        # -----------------------------------------------------------------
        stage2_start = time.monotonic()

        if query.use_reranker and len(candidates) > 0:
            reranked_items = self._reranker_engine.rerank(
                query=query.query,
                candidates=candidates,
                top_k=query.limit,
            )
        else:
            reranked_items = candidates[: query.limit]

        stage2_duration_ms = (time.monotonic() - stage2_start) * 1000.0

        # -----------------------------------------------------------------
        # Etapa 3: Inyección de Contexto Padre si fue solicitado
        # -----------------------------------------------------------------
        enriched_results: list[SearchResultItem] = []
        for item in reranked_items:
            parent_context: str | None = None
            if query.include_parent_context and item.parent_chunk_id is not None:
                parent_chunk = await self._vector_repo.get_chunk_by_id(
                    collection_name=target_collection,
                    chunk_id=item.parent_chunk_id,
                )
                if parent_chunk:
                    parent_context = parent_chunk.content

            enriched_results.append(item.model_copy(update={"parent_context": parent_context}))

        total_duration_ms = (time.monotonic() - overall_start) * 1000.0

        from src.infrastructure.telemetry.metrics import get_metrics

        metrics = get_metrics()
        metrics.inc_counter("search_requests_total", labels={"mode": query.search_mode})
        metrics.observe_histogram(
            "search_latency_seconds",
            (time.monotonic() - overall_start),
            labels={"mode": query.search_mode},
        )

        return SearchResponse(
            query=query.query,
            took_ms=round(total_duration_ms, 2),
            timings=SearchTimings(
                hybrid_retrieval_ms=round(stage1_duration_ms, 2),
                cross_encoder_rerank_ms=round(stage2_duration_ms, 2),
            ),
            total_hits=len(enriched_results),
            results=tuple(enriched_results),
        )
