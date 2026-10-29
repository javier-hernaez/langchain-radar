import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import QdrantClient

from src.application.use_cases.search_code import SearchCodeUseCase
from src.domain.models.chunk import ChunkMetadata, CodeChunk
from src.domain.models.search import SearchQuery, SearchResultItem
from src.infrastructure.vector.qdrant_adapter import QdrantVectorAdapter


@pytest.mark.asyncio
async def test_qdrant_adapter_get_chunks_by_ids() -> None:
    memory_client = QdrantClient(":memory:")
    adapter = QdrantVectorAdapter(client=memory_client, dimension=4)
    coll = "test_batch_retrieval"
    await adapter.ensure_collection(coll)

    # Empty list
    assert await adapter.get_chunks_by_ids(coll, []) == {}

    id_parent = uuid.uuid4()
    id_child = uuid.uuid4()

    meta_parent = ChunkMetadata(
        chunk_id=id_parent,
        parent_chunk_id=None,
        file_path="libs/core/chat.py",
        canonical_import="from langchain_core.chat import BaseChatModel",
        module_name="langchain_core.chat",
        language="python",
        entity_type="class",
        entity_name="BaseChatModel",
        is_parent_class=True,
        commit_sha="1" * 40,
        commit_date=datetime.now(UTC),
        release_version="v0.3.0",
        author="alice",
        content_hash="p" * 64,
    )
    chunk_parent = CodeChunk(
        content="class BaseChatModel: pass",
        raw_code="class BaseChatModel: pass",
        start_line=1,
        end_line=5,
        metadata=meta_parent,
    )

    meta_child = ChunkMetadata(
        chunk_id=id_child,
        parent_chunk_id=id_parent,
        file_path="libs/core/chat.py",
        canonical_import="from langchain_core.chat import BaseChatModel",
        module_name="langchain_core.chat",
        language="python",
        entity_type="method",
        entity_name="BaseChatModel.bind_tools",
        commit_sha="1" * 40,
        commit_date=datetime.now(UTC),
        release_version="v0.3.0",
        author="alice",
        content_hash="c" * 64,
    )
    chunk_child = CodeChunk(
        content="def bind_tools(self): pass",
        raw_code="def bind_tools(self): pass",
        start_line=6,
        end_line=10,
        metadata=meta_child,
    )

    await adapter.upsert_chunks(
        coll,
        [chunk_parent, chunk_child],
        dense_vectors=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        sparse_vectors=[([0], [1.0]), ([1], [1.0])],
    )

    # Batch retrieve
    items = await adapter.get_chunks_by_ids(coll, [id_parent, id_child, uuid.uuid4()])
    assert len(items) == 2
    assert id_parent in items
    assert id_child in items
    assert items[id_parent].entity_name == "BaseChatModel"
    assert items[id_child].entity_name == "BaseChatModel.bind_tools"

    # get_chunk_by_id delegation
    single = await adapter.get_chunk_by_id(coll, id_parent)
    assert single is not None
    assert single.entity_name == "BaseChatModel"


@pytest.mark.asyncio
async def test_search_code_use_case_batch_enrichment() -> None:
    mock_vector = AsyncMock()
    mock_embed = MagicMock()
    mock_rerank = MagicMock()

    id_parent = uuid.uuid4()
    id_child1 = uuid.uuid4()
    id_child2 = uuid.uuid4()

    child1_item = SearchResultItem(
        chunk_id=id_child1,
        parent_chunk_id=id_parent,
        hybrid_score=0.9,
        entity_name="BaseChatModel.bind_tools",
        canonical_import="from langchain_core.chat import BaseChatModel",
        file_path="chat.py",
        release_version="v0.3.0",
        last_updated=datetime.now(UTC),
        content="def bind_tools(self): pass",
    )
    child2_item = SearchResultItem(
        chunk_id=id_child2,
        parent_chunk_id=id_parent,
        hybrid_score=0.85,
        entity_name="BaseChatModel.invoke",
        canonical_import="from langchain_core.chat import BaseChatModel",
        file_path="chat.py",
        release_version="v0.3.0",
        last_updated=datetime.now(UTC),
        content="def invoke(self): pass",
    )

    mock_vector.search_hybrid.return_value = [child1_item, child2_item]
    mock_embed.embed_dense.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_embed.embed_sparse.return_value = [([0], [1.0])]
    mock_rerank.rerank.return_value = [child1_item, child2_item]

    parent_item = SearchResultItem(
        chunk_id=id_parent,
        hybrid_score=1.0,
        entity_name="BaseChatModel",
        canonical_import="from langchain_core.chat import BaseChatModel",
        file_path="chat.py",
        release_version="v0.3.0",
        last_updated=datetime.now(UTC),
        content="class BaseChatModel: ...",
    )
    mock_vector.get_chunks_by_ids.return_value = {id_parent: parent_item}

    use_case = SearchCodeUseCase(
        vector_repo=mock_vector,
        embedding_engine=mock_embed,
        reranker_engine=mock_rerank,
    )

    query = SearchQuery(query="bind tools to chat model", include_parent_context=True)
    res = await use_case.execute(query)

    assert len(res.results) == 2
    # Verify get_chunks_by_ids was called exactly once in batch
    mock_vector.get_chunks_by_ids.assert_called_once()
    assert res.results[0].parent_context == "class BaseChatModel: ..."
    assert res.results[1].parent_context == "class BaseChatModel: ..."
