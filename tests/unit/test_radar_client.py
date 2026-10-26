import pytest
import respx
from httpx import Response

from src.interfaces.client.radar_client import AsyncRadarClient, RadarClient


@respx.mock
def test_sync_radar_client_search_and_health() -> None:
    respx.post("http://test-radar:8000/api/v1/search").mock(
        return_value=Response(
            200,
            json={
                "query": "test query",
                "took_ms": 15.2,
                "timings": {
                    "hybrid_retrieval_ms": 10.0,
                    "cross_encoder_rerank_ms": 5.2,
                },
                "total_hits": 1,
                "results": [
                    {
                        "chunk_id": "c7a8b9e0-6f21-4d9a-9e1b-3f2d1e0a9b8c",
                        "parent_chunk_id": None,
                        "rerank_score": 0.95,
                        "hybrid_score": 0.90,
                        "entity_name": "BaseChatModel",
                        "canonical_import": "from langchain_core.language_models import BaseChatModel",
                        "file_path": "libs/core/chat.py",
                        "release_version": "v0.3.0",
                        "last_updated": "2026-10-14T00:00:00Z",
                        "parent_context": None,
                        "content": "class BaseChatModel: pass",
                        "pr_number": 101,
                        "author": "hwchase17",
                        "is_deprecated": False,
                        "breaking_change": False,
                    }
                ],
            },
        )
    )

    respx.get("http://test-radar:8000/api/v1/health").mock(
        return_value=Response(200, json={"status": "healthy"})
    )
    respx.get("http://test-radar:8000/api/v1/status/sync").mock(
        return_value=Response(200, json={"active_collection": "langchain_radar_active"})
    )
    respx.post("http://test-radar:8000/api/v1/admin/reindex").mock(
        return_value=Response(200, json={"status": "swapped_successfully"})
    )

    with RadarClient(base_url="http://test-radar:8000", admin_token="adm-token") as client:
        search_res = client.search("test query")
        assert search_res.total_hits == 1
        assert search_res.results[0].entity_name == "BaseChatModel"

        health_res = client.get_health()
        assert health_res["status"] == "healthy"

        sync_res = client.get_sync_status()
        assert sync_res["active_collection"] == "langchain_radar_active"

        reindex_res = client.trigger_reindex(version_tag="v2")
        assert reindex_res["status"] == "swapped_successfully"

    client.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_radar_client_methods() -> None:
    respx.post("http://test-radar:8000/api/v1/search").mock(
        return_value=Response(
            200,
            json={
                "query": "async query",
                "took_ms": 12.0,
                "timings": {
                    "hybrid_retrieval_ms": 8.0,
                    "cross_encoder_rerank_ms": 4.0,
                },
                "total_hits": 1,
                "results": [
                    {
                        "chunk_id": "c7a8b9e0-6f21-4d9a-9e1b-3f2d1e0a9b8c",
                        "parent_chunk_id": None,
                        "rerank_score": 0.98,
                        "hybrid_score": 0.92,
                        "entity_name": "bind_tools",
                        "canonical_import": "from langchain_core.language_models import BaseChatModel",
                        "file_path": "libs/core/chat.py",
                        "release_version": "v0.3.0",
                        "last_updated": "2026-10-14T00:00:00Z",
                        "parent_context": None,
                        "content": "def bind_tools(): pass",
                        "pr_number": 102,
                        "author": "hwchase17",
                        "is_deprecated": False,
                        "breaking_change": False,
                    }
                ],
            },
        )
    )
    respx.get("http://test-radar:8000/api/v1/health").mock(
        return_value=Response(200, json={"status": "healthy"})
    )
    respx.get("http://test-radar:8000/api/v1/status/sync").mock(
        return_value=Response(200, json={"active_collection": "langchain_radar_active"})
    )
    respx.post("http://test-radar:8000/api/v1/admin/reindex").mock(
        return_value=Response(200, json={"status": "swapped_successfully"})
    )

    async with AsyncRadarClient(
        base_url="http://test-radar:8000", admin_token="secret-123"
    ) as client:
        search_res = await client.search("async query")
        assert search_res.total_hits == 1

        health_res = await client.get_health()
        assert health_res["status"] == "healthy"

        sync_res = await client.get_sync_status()
        assert sync_res["active_collection"] == "langchain_radar_active"

        reindex_res = await client.trigger_reindex(version_tag="v2")
        assert reindex_res["status"] == "swapped_successfully"

    await client.close()
