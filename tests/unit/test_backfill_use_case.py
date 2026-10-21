from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.use_cases.backfill import BackfillRequest, BackfillUseCase
from src.domain.models.events import CommitFileChange, PRMergedPayload


@pytest.mark.asyncio
async def test_backfill_use_case_execution() -> None:
    # 1. Mocks de componentes
    mock_gh_client = AsyncMock()
    mock_bronze = AsyncMock()
    mock_silver = AsyncMock()
    mock_vector = AsyncMock()
    mock_embedding = MagicMock()

    # Configurar mock de embeddings
    mock_embedding.embed_dense.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
    mock_embedding.embed_sparse.side_effect = lambda texts: [([1], [0.5]) for _ in texts]

    # Simular PR devuelto por GitHub
    sample_pr = PRMergedPayload(
        pr_number=999,
        title="feat(core): add math tool",
        body="Adds a calculator tool",
        author="agent_dev",
        merged_at=datetime.now(UTC),
        head_branch="feat-math",
        merge_commit_sha="2222333344445555666677778888999900001111",
        files_changed=(
            CommitFileChange(
                file_path="libs/core/tools/math.py",
                status="ADDED",
                patch="def calculator(expr: str) -> float: return 42.0",
            ),
            CommitFileChange(
                file_path="docs/tools/math.md",
                status="ADDED",
                patch="# Math Tool\nUsage examples for calculator.",
            ),
            CommitFileChange(
                file_path="poetry.lock",
                status="MODIFIED",
                patch="lockfile content",
            ),
        ),
    )

    # fetch_merged_prs retorna (lista_prs, cursor, has_next)
    mock_gh_client.fetch_merged_prs.return_value = ([sample_pr], None, False)
    mock_vector.upsert_chunks.return_value = 2  # 1 python + 1 doc

    use_case = BackfillUseCase(
        github_client=mock_gh_client,
        bronze_storage=mock_bronze,
        silver_storage=mock_silver,
        vector_repo=mock_vector,
        embedding_engine=mock_embedding,
    )

    request = BackfillRequest(
        repo_owner="langchain-ai",
        repo_name="langchain",
        max_prs=1,
        target_prefixes=("libs/core/", "docs/"),
    )

    summary = await use_case.execute(request)

    # Verificaciones
    assert summary.prs_processed == 1
    assert summary.files_processed == 2  # poetry.lock ignorado
    assert summary.chunks_created >= 2
    assert summary.chunks_indexed >= 2

    # Verificar que Bronze guardó el evento
    assert mock_bronze.save_raw_event.called

    # Verificar que Silver guardó en Delta Lake
    assert mock_silver.append_chunks.called

    # Verificar que Qdrant hizo upsert
    assert mock_vector.upsert_chunks.called


@pytest.mark.asyncio
async def test_backfill_content_hash_deduplication_and_filters() -> None:
    mock_gh_client = AsyncMock()
    mock_bronze = AsyncMock()
    mock_silver = AsyncMock()
    mock_vector = AsyncMock()
    mock_embedding = MagicMock()
    mock_embedding.embed_dense.return_value = [[0.1, 0.2]]
    mock_embedding.embed_sparse.return_value = [([0], [1.0])]

    # Mismo código en dos PRs distintos
    pr_1 = PRMergedPayload(
        pr_number=101,
        title="feat: tool",
        body="",
        author="dev",
        merged_at=datetime.now(UTC),
        head_branch="b1",
        merge_commit_sha="1" * 40,
        files_changed=(
            CommitFileChange(
                file_path="libs/core/__init__.py",
                status="MODIFIED",
                patch="__all__ = ['run']\ndef run(): pass",
            ),
            CommitFileChange(
                file_path="libs/core/empty.py",
                status="MODIFIED",
                patch="",
            ),
            CommitFileChange(
                file_path="other/ignored.py",
                status="MODIFIED",
                patch="pass",
            ),
        ),
    )
    pr_2 = PRMergedPayload(
        pr_number=102,
        title="fix: touch same file",
        body="",
        author="dev",
        merged_at=datetime.now(UTC),
        head_branch="b2",
        merge_commit_sha="2" * 40,
        files_changed=(
            CommitFileChange(
                file_path="libs/core/__init__.py",
                status="MODIFIED",
                patch="__all__ = ['run']\ndef run(): pass",
            ),
        ),
    )

    mock_gh_client.fetch_merged_prs.return_value = ([pr_1, pr_2], None, False)
    mock_vector.upsert_chunks.return_value = 1

    use_case = BackfillUseCase(
        github_client=mock_gh_client,
        bronze_storage=mock_bronze,
        silver_storage=mock_silver,
        vector_repo=mock_vector,
        embedding_engine=mock_embedding,
    )

    summary = await use_case.execute(
        BackfillRequest(
            repo_owner="langchain-ai",
            repo_name="langchain",
            max_prs=2,
            target_prefixes=("libs/core/",),
        )
    )

    assert summary.prs_processed == 2
    assert summary.chunks_skipped_by_hash >= 1
