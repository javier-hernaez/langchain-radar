from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.interfaces.api.app import app
from src.interfaces.api.v1.webhook import set_dependencies


@pytest.mark.asyncio
async def test_webhook_endpoint_ping() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/webhook",
            headers={"X-GitHub-Event": "ping"},
            content=b"{}",
        )
        assert response.status_code == 200
        assert response.json() == {"status": "pong"}


@pytest.mark.asyncio
async def test_webhook_endpoint_accepted_flow() -> None:
    mock_validator = MagicMock()
    mock_validator.verify_signature.return_value = True

    mock_queue = AsyncMock()
    mock_queue.is_duplicate_or_mark.return_value = False  # Nuevo evento

    mock_bronze = AsyncMock()

    set_dependencies(validator=mock_validator, queue=mock_queue, bronze=mock_bronze)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/webhook",
            headers={
                "X-GitHub-Delivery": "delivery_123",
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": "sha256=valid_dummy",
            },
            content=b'{"action": "closed", "pull_request": {"number": 10}}',
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "accepted"
        assert data["event_id"] == "delivery_123"

        assert mock_bronze.save_raw_event.called
        assert mock_queue.enqueue_event.called


@pytest.mark.asyncio
async def test_webhook_endpoint_duplicate_flow() -> None:
    mock_validator = MagicMock()
    mock_validator.verify_signature.return_value = True

    mock_queue = AsyncMock()
    mock_queue.is_duplicate_or_mark.return_value = True  # Duplicado detectado por Redis SETNX

    mock_bronze = AsyncMock()

    set_dependencies(validator=mock_validator, queue=mock_queue, bronze=mock_bronze)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/webhook",
            headers={
                "X-GitHub-Delivery": "delivery_dup",
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": "sha256=valid_dummy",
            },
            content=b'{"action": "closed"}',
        )
        # Retorna 200 sin reprocesar
        assert response.status_code == 200
        assert response.json()["status"] == "already_processed"
        assert not mock_bronze.save_raw_event.called
        assert not mock_queue.enqueue_event.called


@pytest.mark.asyncio
async def test_webhook_endpoint_unauthorized_bad_signature() -> None:
    mock_validator = MagicMock()
    mock_validator.verify_signature.return_value = False  # Firma rechazada

    set_dependencies(validator=mock_validator)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/webhook",
            headers={"X-Hub-Signature-256": "sha256=bad_sig"},
            content=b'{"action": "closed"}',
        )
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_webhook_releases_dedup_mark_when_enqueue_fails() -> None:
    mock_validator = MagicMock()
    mock_validator.verify_signature.return_value = True
    mock_queue = AsyncMock()
    mock_queue.is_duplicate_or_mark.return_value = False
    mock_queue.enqueue_event.side_effect = RuntimeError("redis down")
    mock_bronze = AsyncMock()

    set_dependencies(validator=mock_validator, queue=mock_queue, bronze=mock_bronze)

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/webhook",
            headers={"X-GitHub-Delivery": "delivery_fail", "X-GitHub-Event": "pull_request"},
            content=b'{"action": "closed"}',
        )
    assert response.status_code == 500
    # Sin liberar la marca, el reintento de GitHub se descartaría como duplicado
    mock_queue.unmark_event.assert_awaited_once_with("delivery_fail")


@pytest.mark.asyncio
async def test_webhook_malformed_json_is_rejected_before_dedup() -> None:
    mock_validator = MagicMock()
    mock_validator.verify_signature.return_value = True
    mock_queue = AsyncMock()

    set_dependencies(validator=mock_validator, queue=mock_queue, bronze=AsyncMock())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/webhook",
            headers={"X-GitHub-Delivery": "delivery_bad", "X-GitHub-Event": "pull_request"},
            content=b"not-json{",
        )
    assert response.status_code == 400
    assert not mock_queue.is_duplicate_or_mark.called
