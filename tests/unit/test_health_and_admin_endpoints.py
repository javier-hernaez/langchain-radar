from unittest.mock import AsyncMock

import httpx
import pytest

from config.settings import get_settings
from src.interfaces.api.app import app
from src.interfaces.api.v1.health import set_health_vector_repo


@pytest.mark.asyncio
async def test_health_check_endpoint() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data


@pytest.mark.asyncio
async def test_sync_status_endpoint() -> None:
    from datetime import UTC, datetime

    from src.domain.models.state import SyncCheckpoint
    from src.interfaces.api.v1.health import set_health_state_store

    mock_state = AsyncMock()
    now = datetime.now(UTC)
    mock_state.get_checkpoint.return_value = SyncCheckpoint(
        checkpoint_id="backfill_cursor",
        last_commit_sha="a" * 40,
        last_synced_at=now,
    )
    mock_state.get_processed_pr_count.return_value = 15
    set_health_state_store(mock_state)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/status/sync")
        assert response.status_code == 200
        data = response.json()
        assert data["active_collection"] == "langchain_radar_active"
        assert "stream_key" in data
        assert data["total_processed_prs"] == 15
        assert data["last_commit_sha"] == "a" * 40


@pytest.mark.asyncio
async def test_admin_reindex_blue_green_swap() -> None:
    settings = get_settings()
    mock_vector = AsyncMock()
    set_health_vector_repo(mock_vector)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Petición sin token -> 401
        res_no_auth = await client.post(
            "/api/v1/admin/reindex", json={"new_version_tag": "202610_v2"}
        )
        assert res_no_auth.status_code == 401

        # 2. Petición con token incorrecto -> 403
        res_bad_auth = await client.post(
            "/api/v1/admin/reindex",
            headers={"Authorization": "Bearer wrong_token"},
            json={"new_version_tag": "202610_v2"},
        )
        assert res_bad_auth.status_code == 403

        # 3. Petición autorizada -> 200 y swap de alias ejecutado
        res_ok = await client.post(
            "/api/v1/admin/reindex",
            headers={"Authorization": f"Bearer {settings.secret_key}"},
            json={"new_version_tag": "202610_v2"},
        )
        assert res_ok.status_code == 200
        data = res_ok.json()
        assert data["status"] == "swapped_successfully"
        assert "langchain_radar_202610_v2" in data["active_collection"]
        assert mock_vector.switch_alias.called
