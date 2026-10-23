import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from src.domain.models.search import SearchResponse, SearchResultItem, SearchTimings
from src.interfaces.api.app import app
from src.interfaces.api.v1.search import set_search_use_case


@pytest.mark.asyncio
async def test_search_endpoint_contract() -> None:
    now = datetime.now(UTC)
    chunk_id = uuid.uuid4()
    parent_id = uuid.uuid4()

    mock_result_item = SearchResultItem(
        chunk_id=chunk_id,
        parent_chunk_id=parent_id,
        rerank_score=0.968,
        hybrid_score=0.912,
        entity_name="BaseChatModel.bind_tools",
        canonical_import="from langchain_core.language_models import BaseChatModel",
        file_path="libs/core/langchain_core/language_models/chat_models.py",
        release_version="v0.3.1",
        last_updated=now,
        parent_context="class BaseChatModel: pass",
        content="def bind_tools(...): pass",
        author="hwchase17",
        pr_number=24501,
        is_deprecated=False,
        breaking_change=False,
    )

    mock_response = SearchResponse(
        query="¿Cómo vincular herramientas estructuradas a ChatAnthropic usando LCEL?",
        took_ms=38.0,
        timings=SearchTimings(hybrid_retrieval_ms=19.0, cross_encoder_rerank_ms=19.0),
        total_hits=1,
        results=(mock_result_item,),
    )

    mock_use_case = AsyncMock()
    mock_use_case.execute.return_value = mock_response
    set_search_use_case(mock_use_case)

    payload = {
        "query": "¿Cómo vincular herramientas estructuradas a ChatAnthropic usando LCEL?",
        "limit": 5,
        "filters": {
            "module_prefix": "langchain_core",
            "semver_major": 0,
            "semver_minor": 3,
            "is_deprecated": False,
        },
        "search_mode": "hybrid",
        "use_reranker": True,
        "include_parent_context": True,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/search", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["query"] == payload["query"]
        assert data["took_ms"] == 38.0
        assert data["timings"]["hybrid_retrieval_ms"] == 19.0
        assert data["timings"]["cross_encoder_rerank_ms"] == 19.0
        assert data["total_hits"] == 1
        assert len(data["results"]) == 1

        res_item = data["results"][0]
        assert res_item["entity_name"] == "BaseChatModel.bind_tools"
        assert (
            res_item["canonical_import"]
            == "from langchain_core.language_models import BaseChatModel"
        )
        assert res_item["parent_context"] == "class BaseChatModel: pass"
        assert res_item["rerank_score"] == 0.968


def test_get_search_use_case_default() -> None:
    from unittest.mock import patch

    from src.interfaces.api.v1.search import get_search_use_case, set_search_use_case

    set_search_use_case(None)
    with (
        patch("src.interfaces.api.v1.search.QdrantVectorAdapter"),
        patch("src.interfaces.api.v1.search.FastembedEmbeddingAdapter"),
        patch("src.interfaces.api.v1.search.FastembedRerankerAdapter"),
    ):
        use_case = get_search_use_case()
        assert use_case is not None
