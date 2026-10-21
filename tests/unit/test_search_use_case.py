import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.use_cases.search_code import SearchCodeUseCase
from src.domain.models.search import SearchFilters, SearchQuery, SearchResultItem


@pytest.mark.asyncio
async def test_search_use_case_execution_with_rerank_and_parent_context() -> None:
    mock_vector = AsyncMock()
    mock_embedding = MagicMock()
    mock_reranker = MagicMock()

    mock_embedding.dimension = 4
    mock_embedding.embed_dense.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_embedding.embed_sparse.return_value = [([1, 2], [0.5, 0.8])]

    parent_id = uuid.uuid4()
    child_id = uuid.uuid4()
    now = datetime.now(UTC)

    candidate_child = SearchResultItem(
        chunk_id=child_id,
        parent_chunk_id=parent_id,
        hybrid_score=0.88,
        entity_name="BaseChatModel.bind_tools",
        canonical_import="from langchain_core.language_models import BaseChatModel",
        file_path="libs/core/chat.py",
        release_version="v0.3.1",
        last_updated=now,
        content="def bind_tools(): pass",
    )

    parent_item = SearchResultItem(
        chunk_id=parent_id,
        parent_chunk_id=None,
        hybrid_score=0.95,
        entity_name="BaseChatModel",
        canonical_import="from langchain_core.language_models import BaseChatModel",
        file_path="libs/core/chat.py",
        release_version="v0.3.1",
        last_updated=now,
        content="class BaseChatModel: pass",
    )

    mock_vector.search_hybrid.return_value = [candidate_child]
    # Re-ranker devuelve el child con rerank_score asignado
    reranked_child = candidate_child.model_copy(update={"rerank_score": 0.98})
    mock_reranker.rerank.return_value = [reranked_child]

    # Simular recuperación del padre para inyección de contexto
    mock_vector.get_chunk_by_id.return_value = parent_item

    use_case = SearchCodeUseCase(
        vector_repo=mock_vector,
        embedding_engine=mock_embedding,
        reranker_engine=mock_reranker,
    )

    query = SearchQuery(
        query="bind tools with lcel",
        limit=5,
        filters=SearchFilters(module_prefix="langchain_core"),
        search_mode="hybrid",
        use_reranker=True,
        include_parent_context=True,
    )

    response = await use_case.execute(query)

    assert response.query == "bind tools with lcel"
    assert response.total_hits == 1
    assert len(response.results) == 1

    item = response.results[0]
    assert item.entity_name == "BaseChatModel.bind_tools"
    assert item.rerank_score == 0.98
    assert item.parent_context == "class BaseChatModel: pass"
    assert response.timings.hybrid_retrieval_ms >= 0
    assert response.timings.cross_encoder_rerank_ms >= 0
