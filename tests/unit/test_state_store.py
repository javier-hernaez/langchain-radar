from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.exceptions import StorageError
from src.domain.models.state import SyncCheckpoint
from src.infrastructure.state.redis_state import RedisStateStore


@pytest.mark.asyncio
async def test_redis_state_store_in_memory() -> None:
    store = RedisStateStore(client=None)

    # 1. Initially empty
    checkpoint = await store.get_checkpoint("job_1")
    assert checkpoint is None

    # 2. Save checkpoint
    now = datetime.now(UTC)
    cp = SyncCheckpoint(
        checkpoint_id="job_1",
        last_commit_sha="a" * 40,
        cursor="cursor_123",
        last_synced_at=now,
        total_processed_items=10,
        metadata={"batch": 1},
    )
    await store.save_checkpoint(cp)

    saved = await store.get_checkpoint("job_1")
    assert saved is not None
    assert saved.checkpoint_id == "job_1"
    assert saved.last_commit_sha == "a" * 40
    assert saved.total_processed_items == 10

    # 3. Processed PRs tracking
    assert not await store.is_pr_processed(101)
    await store.record_processed_pr(101, "b" * 40)
    assert await store.is_pr_processed(101)
    assert await store.get_processed_pr_count() == 1


@pytest.mark.asyncio
async def test_redis_state_store_with_mocked_redis() -> None:
    mock_client = AsyncMock()
    store = RedisStateStore(client=mock_client)

    now = datetime.now(UTC)
    cp = SyncCheckpoint(
        checkpoint_id="backfill_cursor",
        last_commit_sha="c" * 40,
        cursor="Y3Vyc29yMQ==",
        last_synced_at=now,
        total_processed_items=25,
    )

    # Save
    await store.save_checkpoint(cp)
    mock_client.set.assert_called_once()

    # Get
    mock_client.get.return_value = cp.model_dump_json().encode("utf-8")
    loaded = await store.get_checkpoint("backfill_cursor")
    assert loaded is not None
    assert loaded.last_commit_sha == "c" * 40

    # Get non-existent
    mock_client.get.return_value = None
    empty = await store.get_checkpoint("non_existent")
    assert empty is None

    # Record PR
    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=[1, 1])
    mock_client.pipeline = MagicMock(return_value=mock_pipe)
    await store.record_processed_pr(200, "d" * 40)
    assert mock_pipe.sadd.called
    assert mock_pipe.hset.called
    assert mock_pipe.execute.called

    # Is processed
    mock_client.sismember.return_value = 1
    assert await store.is_pr_processed(200)

    # Count
    mock_client.scard.return_value = 42
    assert await store.get_processed_pr_count() == 42


@pytest.mark.asyncio
async def test_redis_state_store_error_handling() -> None:
    mock_client = AsyncMock()
    mock_client.get.side_effect = RuntimeError("Redis disconnected")
    mock_client.set.side_effect = RuntimeError("Redis disconnected")
    mock_client.sismember.side_effect = RuntimeError("Redis disconnected")
    mock_client.scard.side_effect = RuntimeError("Redis disconnected")

    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock(side_effect=RuntimeError("Pipeline failed"))
    mock_client.pipeline = MagicMock(return_value=mock_pipe)

    store = RedisStateStore(client=mock_client)

    with pytest.raises(StorageError):
        await store.get_checkpoint("err")

    now = datetime.now(UTC)
    cp = SyncCheckpoint(
        checkpoint_id="err",
        last_commit_sha="e" * 40,
        last_synced_at=now,
    )
    with pytest.raises(StorageError):
        await store.save_checkpoint(cp)

    with pytest.raises(StorageError):
        await store.record_processed_pr(300, "f" * 40)

    with pytest.raises(StorageError):
        await store.is_pr_processed(300)

    with pytest.raises(StorageError):
        await store.get_processed_pr_count()
