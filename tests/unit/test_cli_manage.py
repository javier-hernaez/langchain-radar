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

    with patch("src.interfaces.cli.manage.BackfillUseCase") as mock_use_case_cls:
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
    ):
        mock_queue = AsyncMock()
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


def test_cli_main_dispatch() -> None:
    with (
        patch("sys.argv", ["manage.py", "serve", "--port", "9000"]),
        patch("src.interfaces.cli.manage.run_serve") as mock_serve,
    ):
        main()
        assert mock_serve.called
