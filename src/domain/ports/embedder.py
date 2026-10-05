from typing import Protocol


class EmbeddingEngine(Protocol):
    """Puerto de dominio para generación local de embeddings densos y dispersos."""

    @property
    def dimension(self) -> int:
        """Dimensión del vector denso (ej. 1024 para bge-m3)."""
        ...

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Calcula vectores densos normalizados para un lote de textos."""
        ...

    def embed_sparse(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Calcula pesos léxicos dispersos (índices de tokens y pesos de importancia)."""
        ...
