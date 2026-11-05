import argparse
from unittest.mock import AsyncMock, patch

import pytest

from src.interfaces.cli.manage import (
    main,
    run_backfill,
    run_eval,
    run_reindex,
    run_serve,
    run_worker,
)


@pytest.mark.asyncio
async def test_cli_run_backfill() -> None:
    args = argparse.Namespace(
        owner="langchain-ai",
        repo="langchain",
        max_prs=5,
        batch_size=32,
        collection="test_coll",
    )

    with (
        patch("src.interfaces.cli.manage.BackfillUseCase") as mock_use_case_cls,
        patch("src.interfaces.cli.manage.FastembedEmbeddingAdapter"),
        patch("src.interfaces.cli.manage.RedisStateStore"),
    ):
        mock_instance = AsyncMock()
        mock_instance.execute.return_value = argparse.Namespace(
            prs_processed=5,
            files_processed=10,
            chunks_created=25,
            chunks_indexed=25,
            chunks_skipped_by_hash=0,
            duration_seconds=1.2,
        )
        mock_use_case_cls.return_value = mock_instance

        await run_backfill(args)
        assert mock_instance.execute.called


def test_cli_run_serve() -> None:
    args = argparse.Namespace(
        host="127.0.0.1",
        port=8080,
        reload=False,
    )
    with patch("uvicorn.run") as mock_uvicorn:
        run_serve(args)
        mock_uvicorn.assert_called_once_with(
            "src.interfaces.api.app:app", host="127.0.0.1", port=8080, reload=False
        )


@pytest.mark.asyncio
async def test_cli_run_worker() -> None:
    args = argparse.Namespace(
        consumer_name="test-worker",
        batch_size=5,
        block_ms=500,
        once=True,
    )
    with (
        patch("src.interfaces.cli.manage.RedisStreamsAdapter") as mock_queue_cls,
        patch("src.interfaces.cli.manage.IngestEventUseCase") as mock_use_case_cls,
        patch("src.interfaces.cli.manage.FastembedEmbeddingAdapter"),
        patch("src.interfaces.cli.manage.RedisStateStore"),
    ):
        mock_queue = AsyncMock()
        mock_queue.claim_stale_events.return_value = []
        mock_queue.read_events.return_value = [
            ("msg-1", {"event_id": "evt-1", "payload": "{}"}),
        ]
        mock_queue_cls.return_value = mock_queue

        mock_use_case = AsyncMock()
        mock_use_case.process_stream_message.return_value = True
        mock_use_case_cls.return_value = mock_use_case

        await run_worker(args)
        assert mock_queue.ensure_consumer_group.called
        assert mock_queue.read_events.called
        assert mock_use_case.process_stream_message.called


@pytest.mark.asyncio
async def test_cli_run_reindex() -> None:
    args = argparse.Namespace(
        version_tag="2026_test",
        swap_alias=True,
    )
    with patch("src.interfaces.cli.manage.QdrantVectorAdapter") as mock_vector_cls:
        mock_vector = AsyncMock()
        mock_vector_cls.return_value = mock_vector

        await run_reindex(args)
        assert mock_vector.ensure_collection.called
        assert mock_vector.switch_alias.called


@pytest.mark.asyncio
async def test_cli_run_eval() -> None:
    args = argparse.Namespace(
        dataset="tests/evaluation/golden_dataset.json",
    )
    with patch("tests.evaluation.run_rag_eval.run_evaluation", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = {"hit_rate_at_5": 1.0}
        await run_eval(args)
        assert mock_eval.called


@pytest.mark.asyncio
async def test_cli_run_status(capsys: pytest.CaptureFixture[str]) -> None:
    from src.interfaces.cli.manage import run_status

    args = argparse.Namespace()
    with patch("src.interfaces.cli.manage.RedisStateStore") as mock_state_cls:
        mock_state = AsyncMock()
        mock_state.get_processed_pr_count.return_value = 42
        mock_state.get_checkpoint.return_value = None
        mock_state_cls.from_settings.return_value = mock_state

        await run_status(args)
        captured = capsys.readouterr()
        assert "Total PRs procesados: 42" in captured.out


def test_cli_main_dispatch() -> None:
    with (
        patch("sys.argv", ["manage.py", "serve", "--port", "9000"]),
        patch("src.interfaces.cli.manage.run_serve") as mock_serve,
    ):
        main()
        assert mock_serve.called


@pytest.mark.asyncio
async def test_cli_run_worker_sends_failed_message_to_dlq() -> None:
    args = argparse.Namespace(consumer_name="w", batch_size=5, block_ms=10, once=True)
    with (
        patch("src.interfaces.cli.manage.RedisStreamsAdapter") as mock_queue_cls,
        patch("src.interfaces.cli.manage.IngestEventUseCase") as mock_use_case_cls,
        patch("src.interfaces.cli.manage.FastembedEmbeddingAdapter"),
        patch("src.interfaces.cli.manage.RedisStateStore"),
    ):
        mock_queue = AsyncMock()
        # Un mensaje huérfano reclamado y uno nuevo que falla
        mock_queue.claim_stale_events.return_value = [("msg-0", {"payload": "{}"})]
        mock_queue.read_events.return_value = [("msg-1", {"event_id": "evt-1", "payload": "{}"})]
        mock_queue_cls.return_value = mock_queue

        mock_use_case = AsyncMock()
        mock_use_case.process_stream_message.side_effect = [True, RuntimeError("boom")]
        mock_use_case_cls.return_value = mock_use_case

        await run_worker(args)

        assert mock_use_case.process_stream_message.call_count == 2
        mock_queue.send_to_dlq.assert_awaited_once()
        assert mock_queue.send_to_dlq.call_args.kwargs["event_id"] == "evt-1"
        mock_queue.ack_event.assert_awaited_once_with(
            "stream:github_events", "radar_workers", "msg-1"
        )
