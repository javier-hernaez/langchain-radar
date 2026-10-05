from typing import Protocol
from uuid import UUID

from src.domain.models.chunk import CodeChunk, DocChunk
from src.domain.models.search import SearchFilters, SearchResultItem


class VectorRepository(Protocol):
    """Puerto de dominio para operaciones vectoriales e índices híbridos (Qdrant)."""

    async def ensure_collection(self, collection_name: str) -> None:
        """Crea la colección si no existe, configurando índices densos y dispersos."""
        ...

    async def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[CodeChunk | DocChunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[tuple[list[int], list[float]]] | None = None,
    ) -> int:
        """Inserta o actualiza un lote de chunks con sus representaciones vectoriales."""
        ...

    async def delete_by_file_path(self, collection_name: str, file_path: str) -> int:
        """Elimina todos los vectores asociados a un archivo modificado o borrado."""
        ...

    async def search_hybrid(
        self,
        collection_name: str,
        dense_vector: list[float],
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        limit: int = 25,
        filters: SearchFilters | None = None,
    ) -> list[SearchResultItem]:
        """Ejecuta búsqueda híbrida (Dense + Sparse) fusionada con RRF en Qdrant."""
        ...

    async def get_chunk_by_id(
        self, collection_name: str, chunk_id: UUID
    ) -> SearchResultItem | None:
        """Recupera un chunk específico por su ID."""
        ...

    async def switch_alias(
        self, alias_name: str, target_collection: str, old_collection: str | None = None
    ) -> None:
        """Cambio atómico de alias de colección para migraciones Blue/Green con zero-downtime."""
        ...
