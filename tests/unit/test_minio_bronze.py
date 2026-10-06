from unittest.mock import MagicMock

import pytest

from src.domain.exceptions import StorageError
from src.infrastructure.storage.minio_bronze import MinioBronzeStorage


@pytest.mark.asyncio
async def test_minio_bronze_save_raw_event_success() -> None:
    mock_client = MagicMock()
    storage = MinioBronzeStorage(client=mock_client)

    s3_uri = await storage.save_raw_event(
        event_id="evt_12345",
        year=2026,
        month=10,
        day=5,
        raw_payload='{"action": "closed", "number": 100}',
    )

    assert "s3://bronze/github/langchain/year=2026/month=10/day=05/evt_12345.json" in s3_uri
    assert mock_client.put_object.called
    call_kwargs = mock_client.put_object.call_args[1]
    assert call_kwargs["bucket_name"] == "bronze"
    assert "evt_12345.json" in call_kwargs["object_name"]


@pytest.mark.asyncio
async def test_minio_bronze_get_raw_event_success() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"hello": "world"}'
    mock_client.get_object.return_value = mock_response

    storage = MinioBronzeStorage(client=mock_client)
    content = await storage.get_raw_event("s3://bronze/some/path/event.json")

    assert content == '{"hello": "world"}'
    assert mock_response.close.called
    assert mock_response.release_conn.called


@pytest.mark.asyncio
async def test_minio_bronze_invalid_uri_scheme() -> None:
    mock_client = MagicMock()
    storage = MinioBronzeStorage(client=mock_client)

    with pytest.raises(StorageError) as exc_info:
        await storage.get_raw_event("http://example.com/event.json")

    assert "Esquema de URI no soportado" in str(exc_info.value)
