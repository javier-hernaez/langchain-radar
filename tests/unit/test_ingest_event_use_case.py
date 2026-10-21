import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.use_cases.ingest_event import IngestEventUseCase


@pytest.mark.asyncio
async def test_ingest_event_use_case_merged_pr_success() -> None:
    mock_silver = AsyncMock()
    mock_vector = AsyncMock()
    mock_embedding = MagicMock()
    mock_queue = AsyncMock()

    mock_embedding.embed_dense.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
    mock_embedding.embed_sparse.side_effect = lambda texts: [([1], [0.5]) for _ in texts]

    use_case = IngestEventUseCase(
        silver_storage=mock_silver,
        vector_repo=mock_vector,
        embedding_engine=mock_embedding,
        event_queue=mock_queue,
    )

    pr_payload = {
        "action": "closed",
        "pull_request": {
            "number": 555,
            "title": "feat(core): dynamic prompt template",
            "merged": True,
            "merged_at": "2026-09-08T10:00:00Z",
            "merge_commit_sha": "3333444455556666777788889999000011112222",
            "user": {"login": "ai_architect"},
            "files": [
                {
                    "filename": "libs/core/prompt.py",
                    "status": "modified",
                    "patch": "class DynamicPrompt: pass",
                },
                {
                    "filename": "libs/core/obsolete_module.py",
                    "status": "removed",
                    "patch": "",
                },
            ],
        },
    }

    fields = {"payload": json.dumps(pr_payload)}

    processed = await use_case.process_stream_message(
        stream_name="stream:github_events",
        group_name="workers",
        message_id="msg_123",
        fields=fields,
    )

    assert processed is True
    # Tombstoning para el archivo eliminado
    mock_vector.delete_by_file_path.assert_any_call(
        "langchain_radar_active", "libs/core/obsolete_module.py"
    )
    # Idempotencia previa para el archivo modificado
    mock_vector.delete_by_file_path.assert_any_call("langchain_radar_active", "libs/core/prompt.py")
    # Persistencia en Silver y Gold
    assert mock_silver.append_chunks.called
    assert mock_vector.upsert_chunks.called
    # ACK ejecutado
    mock_queue.ack_event.assert_called_with("stream:github_events", "workers", "msg_123")


@pytest.mark.asyncio
async def test_ingest_event_use_case_ignores_unmerged_pr() -> None:
    mock_silver = AsyncMock()
    mock_vector = AsyncMock()
    mock_embedding = MagicMock()
    mock_queue = AsyncMock()

    use_case = IngestEventUseCase(
        silver_storage=mock_silver,
        vector_repo=mock_vector,
        embedding_engine=mock_embedding,
        event_queue=mock_queue,
    )

    # PR cerrado sin mergear (merged == False)
    pr_payload = {
        "action": "closed",
        "pull_request": {
            "number": 556,
            "merged": False,
        },
    }
    fields = {"payload": json.dumps(pr_payload)}

    processed = await use_case.process_stream_message(
        stream_name="stream:github_events",
        group_name="workers",
        message_id="msg_999",
        fields=fields,
    )

    assert processed is True
    assert not mock_silver.append_chunks.called
    # Debe hacer ACK para no atascar el stream
    mock_queue.ack_event.assert_called_with("stream:github_events", "workers", "msg_999")


@pytest.mark.asyncio
async def test_ingest_event_malformed_json() -> None:
    mock_silver = AsyncMock()
    mock_vector = AsyncMock()
    mock_embedding = MagicMock()
    mock_queue = AsyncMock()

    use_case = IngestEventUseCase(
        silver_storage=mock_silver,
        vector_repo=mock_vector,
        embedding_engine=mock_embedding,
        event_queue=mock_queue,
    )

    res = await use_case.process_stream_message(
        stream_name="stream:github_events",
        group_name="workers",
        message_id="bad_json_msg",
        fields={"payload": "invalid-json{{{"},
    )
    assert res is False
    assert mock_queue.ack_event.called


@pytest.mark.asyncio
async def test_ingest_event_various_file_types() -> None:
    mock_silver = AsyncMock()
    mock_vector = AsyncMock()
    mock_embedding = MagicMock()
    mock_embedding.embed_dense.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_embedding.embed_sparse.return_value = [([0], [1.0]), ([1], [1.0])]
    mock_queue = AsyncMock()

    use_case = IngestEventUseCase(
        silver_storage=mock_silver,
        vector_repo=mock_vector,
        embedding_engine=mock_embedding,
        event_queue=mock_queue,
    )

    pr_payload = {
        "action": "closed",
        "pull_request": {
            "number": 777,
            "title": "feat(core): various files",
            "merged": True,
            "merged_at": "2026-10-26T12:00:00Z",
            "merge_commit_sha": "4" * 40,
            "user": {"login": "dev"},
            "files": [
                # 1. Excluded
                {"filename": "poetry.lock", "status": "modified", "patch": "lock"},
                # 2. Non-target prefix
                {"filename": "tests/test_x.py", "status": "modified", "patch": "pass"},
                # 3. Empty patch
                {"filename": "libs/core/empty.py", "status": "modified", "patch": ""},
                # 4. __init__.py
                {
                    "filename": "libs/core/__init__.py",
                    "status": "modified",
                    "patch": "__all__ = ['Foo']\nclass Foo: pass",
                },
                # 5. Markdown/MDX
                {
                    "filename": "docs/guide.md",
                    "status": "modified",
                    "patch": "## Guide\n```python\nprint(1)\n```",
                },
            ],
        },
    }

    fields = {"payload": json.dumps(pr_payload)}
    res = await use_case.process_stream_message(
        stream_name="stream:github_events",
        group_name="workers",
        message_id="multi_msg",
        fields=fields,
    )
    assert res is True
    assert mock_silver.append_chunks.called
    assert mock_vector.upsert_chunks.called
