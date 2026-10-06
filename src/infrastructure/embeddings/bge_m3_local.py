from collections.abc import Iterable
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding

from config.settings import Settings, get_settings
from src.domain.ports.embedder import EmbeddingEngine


class FastembedEmbeddingAdapter(EmbeddingEngine):
    """Adaptador de inferencia local en CPU para embeddings densos y dispersos vía FastEmbed / ONNX."""

    def __init__(
        self,
        dense_model_name: str | None = None,
        sparse_model_name: str | None = None,
        settings: Settings | None = None,
        lazy_load: bool = True,
    ) -> None:
        self._settings = settings or get_settings()
        self._dense_model_name = dense_model_name or "BAAI/bge-small-en-v1.5"
        self._sparse_model_name = sparse_model_name or "Qdrant/bm25"
        self._dense_model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None
        self._dimension = 384 if "small" in self._dense_model_name else 768

        if not lazy_load:
            self._init_models()

    def _init_models(self) -> None:
        if self._dense_model is None:
            self._dense_model = TextEmbedding(model_name=self._dense_model_name)
            # Determinar dimensión exacta dinámicamente
            sample = list(self._dense_model.embed(["probe"]))[0]
            self._dimension = len(sample)

        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(model_name=self._sparse_model_name)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Calcula vectores densos float normalizados para una lista de textos."""
        if not texts:
            return []
        self._init_models()
        assert self._dense_model is not None
        embeddings: Iterable[Any] = self._dense_model.embed(
            texts, batch_size=self._settings.embedding_batch_size
        )
        return [list(vec.tolist() if hasattr(vec, "tolist") else vec) for vec in embeddings]

    def embed_sparse(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Calcula representaciones dispersas (índices y valores léxicos)."""
        if not texts:
            return []
        self._init_models()
        assert self._sparse_model is not None
        sparse_embeddings: Iterable[Any] = self._sparse_model.embed(
            texts, batch_size=self._settings.embedding_batch_size
        )
        result: list[tuple[list[int], list[float]]] = []
        for sparse_vec in sparse_embeddings:
            indices = list(
                sparse_vec.indices.tolist()
                if hasattr(sparse_vec.indices, "tolist")
                else sparse_vec.indices
            )
            values = list(
                sparse_vec.values.tolist()
                if hasattr(sparse_vec.values, "tolist")
                else sparse_vec.values
            )
            result.append((indices, values))
        return result
