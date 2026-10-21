import uuid
from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient

from src.domain.models.chunk import ChunkMetadata, CodeChunk
from src.domain.models.search import SearchFilters
from src.infrastructure.vector.qdrant_adapter import QdrantVectorAdapter


@pytest.mark.asyncio
async def test_qdrant_adapter_crud_hybrid_search_and_blue_green() -> None:
    # Cliente en memoria para testing ultrarrápido y hermético
    memory_client = QdrantClient(":memory:")
    adapter = QdrantVectorAdapter(client=memory_client, dimension=4)

    collection_v1 = "langchain_radar_v1"
    collection_v2 = "langchain_radar_v2"
    alias_active = "langchain_radar_active"

    # 1. ensure_collection
    await adapter.ensure_collection(collection_v1)
    await adapter.ensure_collection(collection_v2)

    # 2. Crear chunks de prueba
    chunk_id_1 = uuid.uuid4()
    meta_1 = ChunkMetadata(
        chunk_id=chunk_id_1,
        file_path="libs/core/chat.py",
        canonical_import="from langchain_core.chat import ChatOpenAI",
        module_name="langchain_core.chat",
        language="python",
        entity_type="class",
        entity_name="ChatOpenAI",
        commit_sha="1111222233334444555566667777888899990000",
        commit_date=datetime.now(UTC),
        release_version="v0.3.0",
        semver_major=0,
        semver_minor=3,
        author="hwchase",
        content_hash="c" * 64,
        is_deprecated=False,
    )
    chunk_1 = CodeChunk(
        content="class ChatOpenAI: pass",
        raw_code="class ChatOpenAI: pass",
        start_line=1,
        end_line=5,
        metadata=meta_1,
    )

    chunk_id_2 = uuid.uuid4()
    meta_2 = ChunkMetadata(
        chunk_id=chunk_id_2,
        file_path="libs/community/old_llm.py",
        canonical_import="from langchain_community.old import OldLLM",
        module_name="langchain_community.old",
        language="python",
        entity_type="class",
        entity_name="OldLLM",
        commit_sha="1111222233334444555566667777888899990000",
        commit_date=datetime.now(UTC),
        release_version="v0.1.0",
        semver_major=0,
        semver_minor=1,
        author="legacy-user",
        content_hash="d" * 64,
        is_deprecated=True,
    )
    chunk_2 = CodeChunk(
        content="class OldLLM: pass",
        raw_code="class OldLLM: pass",
        start_line=1,
        end_line=10,
        metadata=meta_2,
    )

    dense_vectors = [
        [0.1, 0.2, 0.3, 0.4],
        [0.9, 0.8, 0.7, 0.6],
    ]
    sparse_vectors = [
        ([1, 10], [0.5, 0.8]),
        ([2, 20], [0.3, 0.9]),
    ]

    # 3. Upsert
    inserted = await adapter.upsert_chunks(
        collection_name=collection_v1,
        chunks=[chunk_1, chunk_2],
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
    )
    assert inserted == 2

    # 4. Get by ID
    item = await adapter.get_chunk_by_id(collection_v1, chunk_id_1)
    assert item is not None
    assert item.entity_name == "ChatOpenAI"
    assert item.file_path == "libs/core/chat.py"

    # 5. Hybrid Search sin filtros
    results = await adapter.search_hybrid(
        collection_name=collection_v1,
        dense_vector=[0.1, 0.2, 0.3, 0.4],
        sparse_indices=[1, 10],
        sparse_values=[0.5, 0.8],
        limit=5,
    )
    assert len(results) >= 1
    assert results[0].entity_name == "ChatOpenAI"

    # 6. Search con filtro (is_deprecated=False)
    filtered_results = await adapter.search_hybrid(
        collection_name=collection_v1,
        dense_vector=[0.1, 0.2, 0.3, 0.4],
        limit=5,
        filters=SearchFilters(is_deprecated=False),
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].entity_name == "ChatOpenAI"

    # 7. Delete by file_path (Tombstoning)
    deleted = await adapter.delete_by_file_path(collection_v1, "libs/core/chat.py")
    assert deleted == 1
    deleted_item = await adapter.get_chunk_by_id(collection_v1, chunk_id_1)
    assert deleted_item is None

    # 8. Blue / Green Alias Switch
    # Inicialmente apuntamos al v1
    await adapter.switch_alias(alias_name=alias_active, target_collection=collection_v1)
    aliases = memory_client.get_aliases().aliases
    assert any(a.alias_name == alias_active and a.collection_name == collection_v1 for a in aliases)

    # Migramos a v2 de forma atómica eliminando el alias previo
    await adapter.switch_alias(
        alias_name=alias_active, target_collection=collection_v2, old_collection=collection_v1
    )
    aliases_after = memory_client.get_aliases().aliases
    assert any(
        a.alias_name == alias_active and a.collection_name == collection_v2 for a in aliases_after
    )


@pytest.mark.asyncio
async def test_qdrant_adapter_advanced_filters_and_hashes() -> None:
    from unittest.mock import MagicMock

    from src.domain.exceptions import VectorStoreError

    memory_client = QdrantClient(":memory:")
    adapter = QdrantVectorAdapter(client=memory_client, dimension=4)
    coll = "test_adv_filters"
    await adapter.ensure_collection(coll)

    # Empty upsert
    assert await adapter.upsert_chunks(coll, [], dense_vectors=[]) == 0

    cid = uuid.uuid4()
    meta = ChunkMetadata(
        chunk_id=cid,
        file_path="libs/core/prompts.py",
        canonical_import="from langchain_core.prompts import PromptTemplate",
        module_name="langchain_core.prompts",
        language="python",
        entity_type="class",
        entity_name="PromptTemplate",
        commit_sha="2" * 40,
        commit_date=datetime(2026, 5, 1, tzinfo=UTC),
        release_version="v0.2.0",
        semver_major=0,
        semver_minor=2,
        author="alice",
        content_hash="h" * 64,
        is_deprecated=True,
    )
    chunk = CodeChunk(
        content="class PromptTemplate: pass",
        raw_code="class PromptTemplate: pass",
        start_line=1,
        end_line=10,
        metadata=meta,
    )

    await adapter.upsert_chunks(
        coll,
        [chunk],
        dense_vectors=[[0.5, 0.5, 0.5, 0.5]],
        sparse_vectors=[([0, 1], [1.0, 2.0])],
    )

    # Test all SearchFilters (module_prefix, semver, since_date)
    filters = SearchFilters(
        module_prefix="langchain_core",
        semver_major=0,
        semver_minor=2,
        is_deprecated=True,
        since_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    res = await adapter.search_hybrid(
        collection_name=coll,
        dense_vector=[0.5, 0.5, 0.5, 0.5],
        sparse_indices=[0, 1],
        sparse_values=[1.0, 2.0],
        limit=5,
        filters=filters,
    )
    assert len(res) == 1
    assert res[0].entity_name == "PromptTemplate"

    # Test VectorStoreError propagation on error
    bad_client = MagicMock()
    bad_client.get_collections.side_effect = RuntimeError("Qdrant offline")
    bad_adapter = QdrantVectorAdapter(client=bad_client)
    with pytest.raises(VectorStoreError):
        await bad_adapter.ensure_collection("bad_coll")
