from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.use_cases.backfill import BackfillRequest, BackfillUseCase
from src.domain.models.events import CommitFileChange, PRMergedPayload


class FakeSourceHost:
    """SourceCodeHost en memoria: devuelve el contenido completo por (ruta, ref)."""

    def __init__(self, contents: dict[str, str]) -> None:
        self._contents = contents
        self.requested: list[tuple[str, str]] = []
        self.repos: set[tuple[str, str]] = set()

    async def fetch_pr_files(self, owner: str, repo: str, pr_number: int) -> list[CommitFileChange]:
        raise AssertionError(f"El backfill no debe listar ficheros ({owner}/{repo}#{pr_number})")

    async def fetch_file_content(self, owner: str, repo: str, path: str, ref: str) -> str | None:
        self.repos.add((owner, repo))
        self.requested.append((path, ref))
        return self._contents.get(path)


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
            ),
            CommitFileChange(
                file_path="docs/tools/math.md",
                status="ADDED",
            ),
            CommitFileChange(
                file_path="poetry.lock",
                status="MODIFIED",
            ),
        ),
    )

    # fetch_merged_prs retorna (lista_prs, cursor, has_next)
    mock_gh_client.fetch_merged_prs.return_value = ([sample_pr], None, False)
    mock_vector.upsert_chunks.return_value = 2  # 1 python + 1 doc

    source_host = FakeSourceHost(
        {
            "libs/core/tools/math.py": "def calculator(expr: str) -> float:\n    return 42.0\n",
            "docs/tools/math.md": "# Math Tool\nUsage examples for calculator.",
            "poetry.lock": "lockfile content",
        }
    )

    use_case = BackfillUseCase(
        github_client=mock_gh_client,
        bronze_storage=mock_bronze,
        silver_storage=mock_silver,
        vector_repo=mock_vector,
        embedding_engine=mock_embedding,
        source_host=source_host,
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

    # El contenido se pide en el commit de merge y nunca para ficheros excluidos
    assert ("libs/core/tools/math.py", sample_pr.merge_commit_sha) in source_host.requested
    assert all(path != "poetry.lock" for path, _ in source_host.requested)
    assert source_host.repos == {("langchain-ai", "langchain")}

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
            ),
            CommitFileChange(
                file_path="libs/core/empty.py",
                status="MODIFIED",
            ),
            CommitFileChange(
                file_path="other/ignored.py",
                status="MODIFIED",
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
        source_host=FakeSourceHost(
            {
                "libs/core/__init__.py": "__all__ = ['run']\ndef run(): pass",
                "libs/core/empty.py": "",
                "other/ignored.py": "pass",
            }
        ),
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


@pytest.mark.asyncio
async def test_backfill_skips_unparseable_file_without_aborting() -> None:
    mock_gh_client = AsyncMock()
    mock_vector = AsyncMock()
    mock_vector.upsert_chunks.return_value = 1
    mock_embedding = MagicMock()
    mock_embedding.embed_dense.side_effect = lambda texts: [[0.1] for _ in texts]
    mock_embedding.embed_sparse.side_effect = lambda texts: [([0], [1.0]) for _ in texts]

    pr = PRMergedPayload(
        pr_number=7,
        title="fix: things",
        author="dev",
        merged_at=datetime.now(UTC),
        head_branch="b",
        merge_commit_sha="7" * 40,
        files_changed=(
            CommitFileChange(file_path="libs/core/broken.py", status="MODIFIED"),
            CommitFileChange(file_path="libs/core/ok.py", status="MODIFIED"),
            CommitFileChange(file_path="libs/core/gone.py", status="REMOVED"),
        ),
    )
    mock_gh_client.fetch_merged_prs.return_value = ([pr], None, False)
    source_host = FakeSourceHost(
        {
            "libs/core/broken.py": "def broken(:\n",
            "libs/core/ok.py": "def ok() -> int:\n    return 1\n",
        }
    )

    use_case = BackfillUseCase(
        github_client=mock_gh_client,
        bronze_storage=AsyncMock(),
        silver_storage=AsyncMock(),
        vector_repo=mock_vector,
        embedding_engine=mock_embedding,
        source_host=source_host,
    )
    summary = await use_case.execute(
        BackfillRequest(repo_owner="o", repo_name="r", max_prs=1, target_prefixes=("libs/core/",))
    )

    assert summary.prs_processed == 1
    assert summary.files_processed == 2
    assert summary.chunks_created == 1  # solo ok.py; broken.py se omite sin abortar
    assert all(path != "libs/core/gone.py" for path, _ in source_host.requested)
