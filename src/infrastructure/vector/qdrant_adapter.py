import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import Settings, get_settings
from src.domain.exceptions import VectorStoreError
from src.domain.models.chunk import CodeChunk, DocChunk
from src.domain.models.search import SearchFilters, SearchResultItem
from src.domain.ports.vector_store import VectorRepository


class QdrantVectorAdapter(VectorRepository):
    """Adaptador de infraestructura para Qdrant Vector DB con soporte híbrido y Blue/Green."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        settings: Settings | None = None,
        dimension: int = 384,
    ) -> None:
        self._settings = settings or get_settings()
        self._dimension = dimension
        if client:
            self._client = client
        else:
            self._client = QdrantClient(
                host=self._settings.qdrant_host,
                port=self._settings.qdrant_port,
                api_key=self._settings.qdrant_api_key,
            )

    async def ensure_collection(self, collection_name: str) -> None:
        """Crea la colección si no existe, configurando índices densos y dispersos."""
        try:
            await asyncio.to_thread(self._ensure_collection_sync, collection_name)
        except Exception as e:
            raise VectorStoreError(
                f"Error al verificar/crear la colección '{collection_name}' en Qdrant: {e}",
                {"collection_name": collection_name},
            ) from e

    def _ensure_collection_sync(self, collection_name: str) -> None:
        if self._client.collection_exists(collection_name):
            return
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=self._dimension, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"sparse_lexical": models.SparseVectorParams()},
        )
        # Índices de payload para filtros rápidos
        payload_indexes: dict[str, models.PayloadSchemaType] = {
            "file_path": models.PayloadSchemaType.KEYWORD,
            "module_name": models.PayloadSchemaType.KEYWORD,
            "entity_name": models.PayloadSchemaType.KEYWORD,
            "entity_type": models.PayloadSchemaType.KEYWORD,
            "semver_major": models.PayloadSchemaType.INTEGER,
            "semver_minor": models.PayloadSchemaType.INTEGER,
            "is_deprecated": models.PayloadSchemaType.BOOL,
            "commit_date": models.PayloadSchemaType.DATETIME,
        }
        for field_name, schema in payload_indexes.items():
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
            )

    async def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[CodeChunk | DocChunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[tuple[list[int], list[float]]] | None = None,
    ) -> int:
        """Inserta o actualiza un lote de chunks con sus representaciones vectoriales."""
        if not chunks:
            return 0

        if len(chunks) != len(dense_vectors):
            raise VectorStoreError(
                f"Discrepancia de tamaño: {len(chunks)} chunks vs {len(dense_vectors)} vectores densos"
            )

        points: list[models.PointStruct] = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            point_id = str(meta.chunk_id)
            vector_data: dict[str, Any] = {"dense": dense_vectors[i]}

            if sparse_vectors and i < len(sparse_vectors):
                s_indices, s_values = sparse_vectors[i]
                if s_indices:
                    vector_data["sparse_lexical"] = models.SparseVector(
                        indices=s_indices, values=s_values
                    )

            payload = {
                "chunk_id": point_id,
                "parent_chunk_id": str(meta.parent_chunk_id) if meta.parent_chunk_id else None,
                "file_path": meta.file_path,
                "canonical_import": meta.canonical_import,
                "module_name": meta.module_name,
                "language": meta.language,
                "entity_type": meta.entity_type,
                "entity_name": meta.entity_name,
                "is_parent_class": meta.is_parent_class,
                "commit_sha": meta.commit_sha,
                "commit_date": meta.commit_date.isoformat(),
                "release_version": meta.release_version,
                "semver_major": meta.semver_major,
                "semver_minor": meta.semver_minor,
                "pr_number": meta.pr_number,
                "pr_title": meta.pr_title,
                "author": meta.author,
                "is_deprecated": meta.is_deprecated,
                "breaking_change": meta.breaking_change,
                "content_hash": meta.content_hash,
                "content": chunk.content,
                "raw_code": chunk.raw_code if isinstance(chunk, CodeChunk) else None,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }

            points.append(models.PointStruct(id=point_id, vector=vector_data, payload=payload))

        try:
            await asyncio.to_thread(
                self._client.upsert, collection_name=collection_name, points=points
            )
            return len(points)
        except Exception as e:
            raise VectorStoreError(
                f"Error al hacer upsert de puntos en Qdrant: {e}",
                {"collection_name": collection_name, "count": len(points)},
            ) from e

    async def delete_by_file_path(self, collection_name: str, file_path: str) -> int:
        """Elimina todos los vectores asociados a un archivo modificado o borrado (Tombstoning)."""
        try:
            await asyncio.to_thread(
                self._client.delete,
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_path",
                                match=models.MatchValue(value=file_path),
                            )
                        ]
                    )
                ),
            )
            return 1
        except Exception as e:
            raise VectorStoreError(
                f"Error al eliminar vectores de '{file_path}' en Qdrant: {e}",
                {"collection_name": collection_name, "file_path": file_path},
            ) from e

    def _build_filter(self, filters: SearchFilters | None) -> models.Filter | None:
        if not filters:
            return None

        must_conditions: list[models.Condition] = []
        if filters.module_prefix:
            must_conditions.append(
                models.FieldCondition(
                    key="module_name",
                    match=models.MatchText(text=filters.module_prefix),
                )
            )
        if filters.semver_major is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="semver_major",
                    match=models.MatchValue(value=filters.semver_major),
                )
            )
        if filters.semver_minor is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="semver_minor",
                    match=models.MatchValue(value=filters.semver_minor),
                )
            )
        if filters.is_deprecated is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="is_deprecated",
                    match=models.MatchValue(value=filters.is_deprecated),
                )
            )
        if filters.entity_type is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="entity_type",
                    match=models.MatchValue(value=filters.entity_type),
                )
            )
        if filters.since_date is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="commit_date",
                    range=models.DatetimeRange(gte=filters.since_date),
                )
            )

        if not must_conditions:
            return None
        return models.Filter(must=must_conditions)

    @staticmethod
    def _payload_to_item(payload: dict[str, Any], score: float) -> SearchResultItem:
        commit_date_raw = payload.get("commit_date")
        last_updated = (
            datetime.fromisoformat(commit_date_raw)
            if isinstance(commit_date_raw, str)
            else datetime.now(UTC)
        )
        return SearchResultItem(
            chunk_id=UUID(payload["chunk_id"]),
            parent_chunk_id=UUID(payload["parent_chunk_id"])
            if payload.get("parent_chunk_id")
            else None,
            rerank_score=None,
            hybrid_score=score,
            entity_name=payload.get("entity_name", ""),
            canonical_import=payload.get("canonical_import", ""),
            file_path=payload.get("file_path", ""),
            release_version=payload.get("release_version", "dev"),
            last_updated=last_updated,
            parent_context=None,
            content=payload.get("content", ""),
            author=payload.get("author"),
            pr_number=payload.get("pr_number"),
            is_deprecated=payload.get("is_deprecated", False),
            breaking_change=payload.get("breaking_change", False),
        )

    async def search_hybrid(
        self,
        collection_name: str,
        dense_vector: list[float] | None,
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        limit: int = 25,
        filters: SearchFilters | None = None,
    ) -> list[SearchResultItem]:
        """Búsqueda híbrida (Dense + Sparse con RRF), solo densa o solo dispersa según los vectores."""
        query_filter = self._build_filter(filters)
        sparse_query = (
            models.SparseVector(indices=sparse_indices, values=sparse_values)
            if sparse_indices and sparse_values
            else None
        )
        if dense_vector is None and sparse_query is None:
            raise VectorStoreError("Se requiere al menos un vector denso o disperso para buscar")

        try:
            if dense_vector is not None and sparse_query is not None:
                query_res = await asyncio.to_thread(
                    self._client.query_points,
                    collection_name=collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=dense_vector, using="dense", limit=limit, filter=query_filter
                        ),
                        models.Prefetch(
                            query=sparse_query,
                            using="sparse_lexical",
                            limit=limit,
                            filter=query_filter,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                )
            else:
                query_res = await asyncio.to_thread(
                    self._client.query_points,
                    collection_name=collection_name,
                    query=dense_vector if dense_vector is not None else sparse_query,
                    using="dense" if dense_vector is not None else "sparse_lexical",
                    query_filter=query_filter,
                    limit=limit,
                )

            return [
                self._payload_to_item(sp.payload, float(sp.score))
                for sp in query_res.points
                if sp.payload and "chunk_id" in sp.payload
            ]
        except Exception as e:
            raise VectorStoreError(
                f"Error al ejecutar búsqueda híbrida en Qdrant: {e}",
                {"collection_name": collection_name, "limit": limit},
            ) from e

    async def get_chunks_by_ids(
        self, collection_name: str, chunk_ids: list[UUID]
    ) -> dict[UUID, SearchResultItem]:
        """Recupera un lote de chunks específicos por sus IDs en una única operación."""
        if not chunk_ids:
            return {}

        try:
            points = await asyncio.to_thread(
                self._client.retrieve,
                collection_name=collection_name,
                ids=[str(cid) for cid in chunk_ids],
                with_payload=True,
            )
            items_by_id: dict[UUID, SearchResultItem] = {}
            for point in points:
                payload = point.payload or {}
                if "chunk_id" not in payload:
                    continue
                item = self._payload_to_item(payload, 1.0)
                items_by_id[item.chunk_id] = item
            return items_by_id
        except Exception as e:
            raise VectorStoreError(
                f"Error al recuperar lote de chunks por IDs en Qdrant: {e}",
                {"collection_name": collection_name, "count": len(chunk_ids)},
            ) from e

    async def get_chunk_by_id(
        self, collection_name: str, chunk_id: UUID
    ) -> SearchResultItem | None:
        """Recupera un chunk específico por su ID."""
        results = await self.get_chunks_by_ids(collection_name, [chunk_id])
        return results.get(chunk_id)

    async def switch_alias(
        self, alias_name: str, target_collection: str, old_collection: str | None = None
    ) -> None:
        """Cambio atómico de alias de colección para migraciones Blue/Green con zero-downtime."""
        operations: list[models.AliasOperations] = []
        if old_collection:
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias_name))
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    alias_name=alias_name, collection_name=target_collection
                )
            )
        )
        try:
            await asyncio.to_thread(
                self._client.update_collection_aliases, change_aliases_operations=operations
            )
        except UnexpectedResponse as e:
            raise VectorStoreError(
                f"Error al cambiar alias Blue/Green en Qdrant: {e}",
                {"alias": alias_name, "target": target_collection},
            ) from e
        except Exception as e:
            raise VectorStoreError(
                f"Error inesperado al cambiar alias en Qdrant: {e}",
                {"alias": alias_name, "target": target_collection},
            ) from e
