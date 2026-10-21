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


def test_minio_bronze_ensure_bucket_flow() -> None:
    # 1. Bucket no existe -> crear
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = False
    storage = MinioBronzeStorage(client=mock_client)
    storage.ensure_bucket()
    assert mock_client.make_bucket.called

    # 2. Bucket ya existe -> no llamar a make_bucket
    mock_client.reset_mock()
    mock_client.bucket_exists.return_value = True
    storage.ensure_bucket()
    assert not mock_client.make_bucket.called

    # 3. Error en verificación -> StorageError
    mock_client.reset_mock()
    mock_client.bucket_exists.side_effect = RuntimeError("Conexión perdida")
    with pytest.raises(StorageError):
        storage.ensure_bucket()


@pytest.mark.asyncio
async def test_minio_bronze_save_raw_event_failure() -> None:
    mock_client = MagicMock()
    mock_client.put_object.side_effect = RuntimeError("S3 write failure")
    storage = MinioBronzeStorage(client=mock_client)

    with pytest.raises(StorageError) as exc_info:
        await storage.save_raw_event(
            event_id="err_1",
            year=2026,
            month=10,
            day=14,
            raw_payload="{}",
        )

    assert "Error al guardar evento crudo en MinIO" in str(exc_info.value)


@pytest.mark.asyncio
async def test_minio_bronze_get_raw_event_failures() -> None:
    from minio.error import S3Error

    mock_client = MagicMock()
    storage = MinioBronzeStorage(client=mock_client)

    # Error S3 (objeto no encontrado)
    mock_client.get_object.side_effect = S3Error(
        code="NoSuchKey",
        message="The specified key does not exist",
        resource="bronze/key.json",
        request_id="req1",
        host_id="host1",
        response=MagicMock(status=404),
    )
    with pytest.raises(StorageError) as exc_info:
        await storage.get_raw_event("s3://bronze/key.json")
    assert "Objeto no encontrado o error S3" in str(exc_info.value)

    # Error inesperado
    mock_client.get_object.side_effect = ConnectionResetError("Network reset")
    with pytest.raises(StorageError) as exc_info2:
        await storage.get_raw_event("s3://bronze/key.json")
    assert "Error inesperado al leer objeto de MinIO" in str(exc_info2.value)
