from typing import Protocol

from src.domain.models.search import SearchResultItem


class RerankerEngine(Protocol):
    """Puerto de dominio para Re-ranking semántico profundo con Cross-Encoder local."""

    def rerank(
        self, query: str, candidates: list[SearchResultItem], top_k: int = 5
    ) -> list[SearchResultItem]:
        """Calcula scores de atención cruzada (Query, Chunk) y devuelve el top_k ordenado."""
        ...
