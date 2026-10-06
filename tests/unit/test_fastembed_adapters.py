import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

from src.domain.models.search import SearchResultItem
from src.infrastructure.embeddings.bge_m3_local import FastembedEmbeddingAdapter
from src.infrastructure.embeddings.bge_reranker_local import FastembedRerankerAdapter


def test_fastembed_embedding_adapter_dense_and_sparse() -> None:
    adapter = FastembedEmbeddingAdapter(lazy_load=True)

    # Mock inner models
    mock_dense = MagicMock()
    mock_dense.embed.return_value = [[0.1, 0.2, 0.3]]
    adapter._dense_model = mock_dense
    adapter._dimension = 3

    mock_sparse_point = MagicMock()
    mock_sparse_point.indices = [1, 5]
    mock_sparse_point.values = [0.4, 0.9]
    mock_sparse = MagicMock()
    mock_sparse.embed.return_value = [mock_sparse_point]
    adapter._sparse_model = mock_sparse

    dense_out = adapter.embed_dense(["hello world"])
    assert dense_out == [[0.1, 0.2, 0.3]]

    sparse_out = adapter.embed_sparse(["hello world"])
    assert len(sparse_out) == 1
    assert sparse_out[0] == ([1, 5], [0.4, 0.9])
    assert adapter.dimension == 3


def test_fastembed_reranker_adapter() -> None:
    adapter = FastembedRerankerAdapter(lazy_load=True)

    mock_model = MagicMock()
    # Puntuaciones: primer doc obtiene 0.2, segundo doc obtiene 0.95
    mock_model.rerank.return_value = [0.2, 0.95]
    adapter._model = mock_model

    now = datetime.now(UTC)
    item_1 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        hybrid_score=0.5,
        entity_name="ClassLow",
        canonical_import="from test import ClassLow",
        file_path="test/low.py",
        release_version="v0.1",
        last_updated=now,
        content="low relevance content",
    )
    item_2 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        hybrid_score=0.6,
        entity_name="ClassHigh",
        canonical_import="from test import ClassHigh",
        file_path="test/high.py",
        release_version="v0.1",
        last_updated=now,
        content="high relevance content",
    )

    reranked = adapter.rerank("relevant query", [item_1, item_2], top_k=2)

    assert len(reranked) == 2
    # El de mayor puntuación (0.95) debe quedar de primero
    assert reranked[0].entity_name == "ClassHigh"
    assert reranked[0].rerank_score == 0.95
    assert reranked[1].entity_name == "ClassLow"
    assert reranked[1].rerank_score == 0.2
