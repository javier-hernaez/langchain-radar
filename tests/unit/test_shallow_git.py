from typing import Any

import pytest

from src.infrastructure.github.shallow_git import ShallowGitEngine


def test_shallow_git_exclusion_rules() -> None:
    engine = ShallowGitEngine()

    # Lockfiles deben ser excluidos
    assert engine.is_excluded_file("poetry.lock") is True
    assert engine.is_excluded_file("subfolder/uv.lock") is True
    assert engine.is_excluded_file("package-lock.json") is True

    # Binarios y assets estáticos minificados
    assert engine.is_excluded_file("static/bundle.min.js") is True
    assert engine.is_excluded_file("assets/logo.png") is True
    assert engine.is_excluded_file("compiled/lib.so") is True
    assert engine.is_excluded_file("data/dump.parquet") is True

    # Test fixtures
    assert engine.is_excluded_file("tests/fixtures/sample_payload.json") is True

    # Archivos válidos de código y documentación
    assert engine.is_excluded_file("libs/core/langchain_core/tools.py") is False
    assert engine.is_excluded_file("docs/guides/quickstart.md") is False
    assert engine.is_excluded_file("docs/tutorial.mdx") is False


def test_shallow_git_threshold_trigger() -> None:
    engine = ShallowGitEngine()

    # Si la API indica que está truncado, debe usar fallback
    assert (
        engine.should_use_shallow_fallback(total_files=5, total_lines=50, is_truncated=True) is True
    )

    # Si excede 300 archivos
    assert engine.should_use_shallow_fallback(total_files=350, total_lines=500) is True

    # Si excede 3000 líneas
    assert engine.should_use_shallow_fallback(total_files=20, total_lines=4500) is True

    # PR típico pequeño
    assert engine.should_use_shallow_fallback(total_files=5, total_lines=120) is False


@pytest.mark.asyncio
async def test_fetch_commit_files_shallow_success() -> None:
    from unittest.mock import AsyncMock, patch

    engine = ShallowGitEngine()

    mock_proc_clone = AsyncMock()
    mock_proc_clone.returncode = 0
    mock_proc_clone.communicate.return_value = (b"", b"")

    mock_proc_ls = AsyncMock()
    mock_proc_ls.returncode = 0
    mock_proc_ls.communicate.return_value = (
        b"libs/core/langchain_core/tools.py\npoetry.lock\nother/file.py\n",
        b"",
    )

    mock_proc_cat = AsyncMock()
    mock_proc_cat.returncode = 0
    mock_proc_cat.communicate.return_value = (b"def dummy(): pass", b"")

    async def fake_subprocess_exec(*cmd: Any, **_kwargs: Any) -> Any:
        if "clone" in cmd:
            return mock_proc_clone
        elif "ls-tree" in cmd:
            return mock_proc_ls
        elif "show" in cmd:
            return mock_proc_cat
        return mock_proc_clone

    with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess_exec):
        files = await engine.fetch_commit_files_shallow(
            repo_url="https://github.com/langchain-ai/langchain.git",
            commit_sha="a" * 40,
            target_prefixes=("libs/core/",),
        )

    assert len(files) == 1
    assert files[0][0] == "libs/core/langchain_core/tools.py"
    assert files[0][1] == "def dummy(): pass"


@pytest.mark.asyncio
async def test_fetch_commit_files_shallow_clone_failure() -> None:
    from unittest.mock import AsyncMock, patch

    from src.domain.exceptions import StorageError

    engine = ShallowGitEngine()

    mock_proc = AsyncMock()
    mock_proc.returncode = 128
    mock_proc.communicate.return_value = (b"", b"fatal: repository not found")

    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        pytest.raises(StorageError) as exc_info,
    ):
        await engine.fetch_commit_files_shallow(
            repo_url="https://github.com/bad/repo.git",
            commit_sha="a" * 40,
        )

    assert "Fallo al ejecutar git clone superficial" in str(exc_info.value)


@pytest.mark.asyncio
async def test_fetch_commit_files_shallow_ls_tree_failure() -> None:
    from unittest.mock import AsyncMock, patch

    from src.domain.exceptions import StorageError

    engine = ShallowGitEngine()

    mock_proc_clone = AsyncMock()
    mock_proc_clone.returncode = 0
    mock_proc_clone.communicate.return_value = (b"", b"")

    mock_proc_ls = AsyncMock()
    mock_proc_ls.returncode = 1
    mock_proc_ls.communicate.return_value = (b"", b"fatal: Not a valid object name")

    async def fake_subprocess_exec(*cmd: Any, **_kwargs: Any) -> Any:
        if "clone" in cmd:
            return mock_proc_clone
        return mock_proc_ls

    with (
        patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess_exec),
        pytest.raises(StorageError) as exc_info,
    ):
        await engine.fetch_commit_files_shallow(
            repo_url="https://github.com/langchain-ai/langchain.git",
            commit_sha="badsha",
        )

    assert "Fallo al listar archivos del commit" in str(exc_info.value)
