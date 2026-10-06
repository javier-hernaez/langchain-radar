from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient, QdrantClient, models
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
        async_client: AsyncQdrantClient | None = None,
        settings: Settings | None = None,
        dimension: int = 384,
    ) -> None:
        self._settings = settings or get_settings()
        self._dimension = dimension
        self._async_client = async_client
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
            collections = self._client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            if not exists:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self._dimension, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={"sparse_lexical": models.SparseVectorParams()},
                )
                # Crear índices de payload para filtros rápidos
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name="file_path",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name="module_name",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name="entity_name",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name="semver_major",
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name="semver_minor",
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name="is_deprecated",
                    field_schema=models.PayloadSchemaType.BOOL,
                )
        except Exception as e:
            raise VectorStoreError(
                f"Error al verificar/crear la colección '{collection_name}' en Qdrant: {e}",
                {"collection_name": collection_name},
            ) from e

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
            self._client.upsert(collection_name=collection_name, points=points)
            return len(points)
        except Exception as e:
            raise VectorStoreError(
                f"Error al hacer upsert de puntos en Qdrant: {e}",
                {"collection_name": collection_name, "count": len(points)},
            ) from e

    async def delete_by_file_path(self, collection_name: str, file_path: str) -> int:
        """Elimina todos los vectores asociados a un archivo modificado o borrado (Tombstoning)."""
        try:
            self._client.delete(
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
        query_filter = self._build_filter(filters)

        try:
            if sparse_indices and sparse_values and len(sparse_indices) > 0:
                prefetch = [
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=limit,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                        using="sparse_lexical",
                        limit=limit,
                        filter=query_filter,
                    ),
                ]
                query_res = self._client.query_points(
                    collection_name=collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                )
                scored_points = query_res.points
            else:
                query_res = self._client.query_points(
                    collection_name=collection_name,
                    query=dense_vector,
                    using="dense",
                    query_filter=query_filter,
                    limit=limit,
                )
                scored_points = query_res.points

            results: list[SearchResultItem] = []
            for sp in scored_points:
                payload = sp.payload or {}
                commit_date_raw = payload.get("commit_date")
                if isinstance(commit_date_raw, str):
                    last_updated = datetime.fromisoformat(commit_date_raw)
                else:
                    last_updated = datetime.now(UTC)

                item = SearchResultItem(
                    chunk_id=UUID(payload["chunk_id"]),
                    parent_chunk_id=UUID(payload["parent_chunk_id"])
                    if payload.get("parent_chunk_id")
                    else None,
                    rerank_score=None,
                    hybrid_score=float(sp.score),
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
                results.append(item)
            return results
        except Exception as e:
            raise VectorStoreError(
                f"Error al ejecutar búsqueda híbrida en Qdrant: {e}",
                {"collection_name": collection_name, "limit": limit},
            ) from e

    async def get_chunk_by_id(
        self, collection_name: str, chunk_id: UUID
    ) -> SearchResultItem | None:
        """Recupera un chunk específico por su ID."""
        try:
            points = self._client.retrieve(
                collection_name=collection_name,
                ids=[str(chunk_id)],
                with_payload=True,
            )
            if not points:
                return None
            point = points[0]
            payload = point.payload or {}
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
                hybrid_score=1.0,
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
        except Exception as e:
            raise VectorStoreError(
                f"Error al recuperar chunk por ID en Qdrant: {e}",
                {"collection_name": collection_name, "chunk_id": str(chunk_id)},
            ) from e

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
            self._client.update_collection_aliases(change_aliases_operations=operations)
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
