from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ResponseError

from src.infrastructure.queue.redis_streams import RedisStreamsAdapter


@pytest.mark.asyncio
async def test_redis_streams_deduplication() -> None:
    mock_redis = AsyncMock()
    # Primera vez: SETNX devuelve True (nuevo evento)
    mock_redis.set.return_value = True
    adapter = RedisStreamsAdapter(client=mock_redis)

    is_dup_1 = await adapter.is_duplicate_or_mark("event_abc", ttl_seconds=86400)
    assert is_dup_1 is False
    assert mock_redis.set.called

    # Segunda vez: SETNX devuelve None/False (duplicado)
    mock_redis.set.return_value = None
    is_dup_2 = await adapter.is_duplicate_or_mark("event_abc", ttl_seconds=86400)
    assert is_dup_2 is True


@pytest.mark.asyncio
async def test_redis_streams_enqueue_and_ack() -> None:
    mock_redis = AsyncMock()
    mock_redis.xadd.return_value = "1720000000000-0"
    adapter = RedisStreamsAdapter(client=mock_redis)

    msg_id = await adapter.enqueue_event("stream:events", "evt_1", '{"test": 1}')
    assert msg_id == "1720000000000-0"
    assert mock_redis.xadd.called

    await adapter.ack_event("stream:events", "group_workers", "1720000000000-0")
    assert mock_redis.xack.called


@pytest.mark.asyncio
async def test_redis_streams_consumer_group_idempotency() -> None:
    mock_redis = AsyncMock()
    # Simular que el grupo ya existía (BUSYGROUP)
    mock_redis.xgroup_create.side_effect = ResponseError(
        "BUSYGROUP Consumer Group name already exists"
    )
    adapter = RedisStreamsAdapter(client=mock_redis)

    # No debe lanzar excepción
    await adapter.ensure_consumer_group("stream:events", "group_workers")
    assert mock_redis.xgroup_create.called


@pytest.mark.asyncio
async def test_redis_streams_read_events_and_dlq() -> None:
    mock_redis = AsyncMock()
    mock_redis.xreadgroup.return_value = [
        ("stream:events", [("msg_1", {"event_id": "e1", "payload": "{}"})])
    ]
    mock_redis.xadd.return_value = "dlq_msg_1"
    adapter = RedisStreamsAdapter(client=mock_redis)

    events = await adapter.read_events("stream:events", "group", "c1", count=1)
    assert len(events) == 1
    assert events[0][0] == "msg_1"

    dlq_id = await adapter.send_to_dlq("stream:events", "e1", "error detail", "{}")
    assert dlq_id == "dlq_msg_1"


@pytest.mark.asyncio
async def test_redis_streams_error_cases() -> None:
    from src.domain.exceptions import EventQueueError

    mock_redis = AsyncMock()
    adapter = RedisStreamsAdapter(client=mock_redis)

    # 1. is_duplicate_or_mark failure
    mock_redis.set.side_effect = RuntimeError("Redis connection dropped")
    with pytest.raises(EventQueueError):
        await adapter.is_duplicate_or_mark("e_err")

    # 2. enqueue_event failure
    mock_redis.xadd.side_effect = RuntimeError("Redis write error")
    with pytest.raises(EventQueueError):
        await adapter.enqueue_event("stream:err", "e_err", "{}")

    # 3. ensure_consumer_group other ResponseError & general exception
    mock_redis.xgroup_create.side_effect = ResponseError("NOGROUP No such key")
    with pytest.raises(EventQueueError):
        await adapter.ensure_consumer_group("stream:err", "grp_err")

    mock_redis.xgroup_create.side_effect = RuntimeError("Unexpected group error")
    with pytest.raises(EventQueueError):
        await adapter.ensure_consumer_group("stream:err", "grp_err")

    # 4. read_events empty response and failure
    mock_redis.xreadgroup.side_effect = None
    mock_redis.xreadgroup.return_value = None
    assert await adapter.read_events("stream:err", "grp", "c1") == []

    mock_redis.xreadgroup.side_effect = RuntimeError("Read failure")
    with pytest.raises(EventQueueError):
        await adapter.read_events("stream:err", "grp", "c1")

    # 5. ack_event failure
    mock_redis.xack.side_effect = RuntimeError("Ack failure")
    with pytest.raises(EventQueueError):
        await adapter.ack_event("stream:err", "grp", "msg_1")

    # 6. send_to_dlq failure
    mock_redis.xadd.side_effect = RuntimeError("DLQ write failure")
    with pytest.raises(EventQueueError):
        await adapter.send_to_dlq("stream:err", "e_err", "detail", "{}")


@pytest.mark.asyncio
async def test_redis_streams_claim_stale_and_unmark() -> None:
    mock_redis = AsyncMock()
    mock_redis.xautoclaim.return_value = [
        "0-0",
        [("1-0", {"event_id": "e1", "payload": "{}"}), ("2-0", None)],
        [],
    ]
    adapter = RedisStreamsAdapter(client=mock_redis)

    claimed = await adapter.claim_stale_events("s", "g", "c", min_idle_ms=1000, count=5)
    assert claimed == [("1-0", {"event_id": "e1", "payload": "{}"})]
    assert mock_redis.xautoclaim.call_args.kwargs["min_idle_time"] == 1000

    await adapter.unmark_event("e1")
    mock_redis.delete.assert_awaited_once_with("event:e1")
