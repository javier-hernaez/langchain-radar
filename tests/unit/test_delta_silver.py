import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.domain.models.chunk import ChunkMetadata, CodeChunk, DocChunk
from src.infrastructure.storage.delta_silver import DeltaSilverStorage


@pytest.mark.asyncio
async def test_delta_silver_append_and_read_chunks(tmp_path: Path) -> None:
    table_dir = str(tmp_path / "delta_test")
    storage = DeltaSilverStorage(table_uri=table_dir)

    chunk_id_1 = uuid.uuid4()
    meta_1 = ChunkMetadata(
        chunk_id=chunk_id_1,
        parent_chunk_id=None,
        file_path="libs/core/langchain_core/chat_models.py",
        canonical_import="from langchain_core.chat_models import BaseChatModel",
        module_name="langchain_core.chat_models",
        language="python",
        entity_type="class",
        entity_name="BaseChatModel",
        is_parent_class=True,
        commit_sha="a1b2c3d4e5f67890123456789abcdef012345678",
        commit_date=datetime.now(UTC),
        release_version="v0.3.1",
        semver_major=0,
        semver_minor=3,
        pr_number=24501,
        pr_title="feat: chat model",
        author="hwchase17",
        is_deprecated=False,
        breaking_change=False,
        content_hash="a" * 64,
    )
    code_chunk = CodeChunk(
        content="class BaseChatModel: pass",
        raw_code="class BaseChatModel: pass",
        start_line=1,
        end_line=10,
        metadata=meta_1,
    )

    chunk_id_2 = uuid.uuid4()
    meta_2 = ChunkMetadata(
        chunk_id=chunk_id_2,
        parent_chunk_id=None,
        file_path="docs/overview.md",
        canonical_import="docs/overview.md",
        module_name="docs.overview",
        language="markdown",
        entity_type="doc_section",
        entity_name="Overview",
        is_parent_class=False,
        commit_sha="a1b2c3d4e5f67890123456789abcdef012345678",
        commit_date=datetime.now(UTC),
        release_version="v0.3.1",
        semver_major=0,
        semver_minor=3,
        author="docs-writer",
        content_hash="b" * 64,
    )
    doc_chunk = DocChunk(
        content="# Overview\nWelcome to LangChain.",
        header_path=("Overview",),
        start_line=1,
        end_line=5,
        metadata=meta_2,
    )

    # Escribir en Delta Lake
    uri = await storage.append_chunks([code_chunk, doc_chunk])
    assert uri == table_dir

    # Leer por commit_sha
    read_chunks = await storage.get_chunks_by_commit("a1b2c3d4e5f67890123456789abcdef012345678")
    assert len(read_chunks) == 2

    # Verificar deserialización correcta
    code_found = [c for c in read_chunks if isinstance(c, CodeChunk)][0]
    doc_found = [c for c in read_chunks if isinstance(c, DocChunk)][0]

    assert code_found.metadata.entity_name == "BaseChatModel"
    assert code_found.start_line == 1
    assert doc_found.metadata.entity_name == "Overview"
    assert doc_found.header_path == ("Overview",)
