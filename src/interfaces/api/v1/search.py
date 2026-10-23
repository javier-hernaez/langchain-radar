import logging

from fastapi import APIRouter, status

from config.settings import get_settings
from src.application.use_cases.search_code import SearchCodeUseCase
from src.domain.models.search import SearchQuery, SearchResponse
from src.domain.ports.embedder import EmbeddingEngine
from src.domain.ports.reranker import RerankerEngine
from src.domain.ports.vector_store import VectorRepository
from src.infrastructure.embeddings.bge_m3_local import FastembedEmbeddingAdapter
from src.infrastructure.embeddings.bge_reranker_local import FastembedRerankerAdapter
from src.infrastructure.vector.qdrant_adapter import QdrantVectorAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Search"])

_search_use_case: SearchCodeUseCase | None = None


def get_search_use_case() -> SearchCodeUseCase:
    global _search_use_case
    if _search_use_case is None:
        settings = get_settings()
        vector_repo: VectorRepository = QdrantVectorAdapter(settings=settings)
        embedding_engine: EmbeddingEngine = FastembedEmbeddingAdapter(settings=settings)
        reranker_engine: RerankerEngine = FastembedRerankerAdapter(settings=settings)
        _search_use_case = SearchCodeUseCase(
            vector_repo=vector_repo,
            embedding_engine=embedding_engine,
            reranker_engine=reranker_engine,
            settings=settings,
        )
    return _search_use_case


def set_search_use_case(use_case: SearchCodeUseCase | None) -> None:
    """Permite inyectar un caso de uso mockeado en tests herméticos."""
    global _search_use_case
    _search_use_case = use_case


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Búsqueda Semántica Híbrida con Re-ranking Local",
    description="Recupera código idiomático y documentación precisa fusionando similitud densa y dispersa con Cross-Encoder local.",
)
async def search_code(query: SearchQuery) -> SearchResponse:
    """Ejecuta búsqueda híbrida en dos etapas con enriquecimiento contextual."""
    use_case = get_search_use_case()
    return await use_case.execute(query)
