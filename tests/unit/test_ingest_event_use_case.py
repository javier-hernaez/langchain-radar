import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.use_cases.ingest_event import IngestEventUseCase
from src.domain.models.events import CommitFileChange


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


class FakeSourceHost:
    """SourceCodeHost en memoria que registra las llamadas recibidas."""

    def __init__(self, files: list[CommitFileChange], contents: dict[str, str]) -> None:
        self._files = files
        self._contents = contents
        self.pr_calls: list[tuple[str, str, int]] = []
        self.content_calls: list[tuple[str, str, str, str]] = []

    async def fetch_pr_files(self, owner: str, repo: str, pr_number: int) -> list[CommitFileChange]:
        self.pr_calls.append((owner, repo, pr_number))
        return self._files

    async def fetch_file_content(self, owner: str, repo: str, path: str, ref: str) -> str | None:
        self.content_calls.append((owner, repo, path, ref))
        return self._contents.get(path)


@pytest.mark.asyncio
async def test_ingest_real_webhook_payload_fetches_files_and_full_content() -> None:
    mock_silver = AsyncMock()
    mock_vector = AsyncMock()
    mock_vector.upsert_chunks.return_value = 3
    mock_embedding = MagicMock()
    mock_embedding.embed_dense.side_effect = lambda texts: [[0.1] for _ in texts]
    mock_embedding.embed_sparse.side_effect = lambda texts: [([1], [0.5]) for _ in texts]
    mock_queue = AsyncMock()

    merge_sha = "5" * 40
    host = FakeSourceHost(
        files=[
            CommitFileChange(
                file_path="libs/core/prompt.py",
                status="MODIFIED",
                patch="@@ -1,2 +1,3 @@ class Prompt:\n     pass\n+    x = 1",
            ),
            CommitFileChange(
                file_path="libs/core/new_name.py",
                status="RENAMED",
                previous_filename="libs/core/old_name.py",
            ),
        ],
        contents={
            "libs/core/prompt.py": "class Prompt:\n    def render(self) -> str:\n        return ''\n",
            "libs/core/new_name.py": "def moved() -> None:\n    pass\n",
        },
    )

    use_case = IngestEventUseCase(
        silver_storage=mock_silver,
        vector_repo=mock_vector,
        embedding_engine=mock_embedding,
        event_queue=mock_queue,
        source_host=host,
    )

    # Estructura real encolada por el webhook: el PR no trae "files"
    queued = {
        "event_id": "d-1",
        "event_type": "pull_request",
        "action": "closed",
        "payload": {
            "action": "closed",
            "repository": {"full_name": "acme/widgets"},
            "pull_request": {
                "number": 42,
                "title": "feat: prompt render",
                "merged": True,
                "merged_at": "2026-10-01T10:00:00Z",
                "merge_commit_sha": merge_sha,
                "user": {"login": "dev"},
                "labels": [{"name": "breaking"}],
            },
        },
    }

    processed = await use_case.process_stream_message(
        stream_name="s", group_name="g", message_id="m-1", fields={"payload": json.dumps(queued)}
    )

    assert processed is True
    assert host.pr_calls == [("acme", "widgets", 42)]
    assert ("acme", "widgets", "libs/core/prompt.py", merge_sha) in host.content_calls
    # El renombrado elimina los vectores de la ruta anterior
    mock_vector.delete_by_file_path.assert_any_call(
        "langchain_radar_active", "libs/core/old_name.py"
    )

    chunks = mock_silver.append_chunks.call_args.args[0]
    entity_types = {c.metadata.entity_type for c in chunks}
    assert {"class", "method", "function", "diff_hunk"} <= entity_types
    # La etiqueta "breaking" se propaga a los chunks estructurales
    assert any(c.metadata.breaking_change for c in chunks if c.metadata.entity_type == "class")
    mock_queue.ack_event.assert_called_with("s", "g", "m-1")
