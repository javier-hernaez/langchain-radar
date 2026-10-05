import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from config.settings import Settings
from src.domain.exceptions import SecurityViolationError, WebhookValidationError
from src.domain.models.chunk import ChunkMetadata, CodeChunk
from src.domain.models.search import (
    SearchFilters,
    SearchQuery,
    SearchResponse,
    SearchResultItem,
    SearchTimings,
)


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.api_port == 8000
    assert settings.minio_bronze_bucket == "bronze"
    assert settings.minio_silver_bucket == "silver"
    assert settings.qdrant_collection_active == "langchain_radar_active"
    assert settings.redis_port == 6379


def test_chunk_metadata_immutability_and_extra_fields() -> None:
    meta = ChunkMetadata(
        chunk_id=uuid.uuid4(),
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
        pr_title="feat: enhance chat model",
        author="hwchase17",
        is_deprecated=False,
        breaking_change=False,
        content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )

    # Immutability check
    with pytest.raises(ValidationError):
        # Direct attribute assignment is blocked at runtime by frozen=True
        meta.is_deprecated = True

    # Extra fields forbidden check
    with pytest.raises(ValidationError):
        ChunkMetadata.model_validate(
            {**meta.model_dump(), "extra_field": "disallowed"}
        )


def test_code_chunk_structure() -> None:
    meta = ChunkMetadata(
        chunk_id=uuid.uuid4(),
        file_path="libs/core/langchain_core/tools.py",
        canonical_import="from langchain_core.tools import tool",
        module_name="langchain_core.tools",
        language="python",
        entity_type="function",
        entity_name="tool",
        commit_sha="abcdef1234567890abcdef1234567890abcdef12",
        commit_date=datetime.now(UTC),
        release_version="v0.3.0",
        author="developer",
        content_hash="a" * 64,
    )
    chunk = CodeChunk(
        content="def tool(): pass",
        raw_code="def tool(): pass",
        start_line=10,
        end_line=20,
        metadata=meta,
    )
    assert chunk.metadata.entity_name == "tool"
    assert chunk.start_line == 10


def test_search_models() -> None:
    query = SearchQuery(
        query="bind_tools with LCEL",
        limit=5,
        filters=SearchFilters(module_prefix="langchain_core"),
    )
    assert query.limit == 5
    assert query.filters.module_prefix == "langchain_core"

    item = SearchResultItem(
        chunk_id=uuid.uuid4(),
        hybrid_score=0.95,
        entity_name="BaseChatModel.bind_tools",
        canonical_import="from langchain_core.chat_models import BaseChatModel",
        file_path="libs/core/langchain_core/chat_models.py",
        release_version="v0.3.1",
        last_updated=datetime.now(UTC),
        content="def bind_tools(...): ...",
    )
    response = SearchResponse(
        query=query.query,
        took_ms=35.5,
        timings=SearchTimings(hybrid_retrieval_ms=20.0, cross_encoder_rerank_ms=15.5),
        total_hits=1,
        results=(item,),
    )
    assert response.total_hits == 1
    assert len(response.results) == 1


def test_domain_exceptions() -> None:
    err = SecurityViolationError("Secret detected in commit", {"pattern": "ghp_***"})
    assert isinstance(err, Exception)
    assert err.details["pattern"] == "ghp_***"

    val_err = WebhookValidationError("Invalid HMAC signature")
    assert "Invalid HMAC" in str(val_err)
