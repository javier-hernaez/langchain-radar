import uuid
from typing import Any

import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from config.settings import Settings, get_settings
from src.domain.exceptions import StorageError
from src.domain.models.chunk import ChunkMetadata, CodeChunk, DocChunk
from src.domain.ports.object_store import SilverStorage

SILVER_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("parent_chunk_id", pa.string()),
        ("file_path", pa.string()),
        ("canonical_import", pa.string()),
        ("module_name", pa.string()),
        ("language", pa.string()),
        ("entity_type", pa.string()),
        ("entity_name", pa.string()),
        ("is_parent_class", pa.bool_()),
        ("commit_sha", pa.string()),
        ("commit_date", pa.timestamp("us", tz="UTC")),
        ("release_version", pa.string()),
        ("semver_major", pa.int32()),
        ("semver_minor", pa.int32()),
        ("pr_number", pa.int32()),
        ("pr_title", pa.string()),
        ("author", pa.string()),
        ("is_deprecated", pa.bool_()),
        ("breaking_change", pa.bool_()),
        ("content_hash", pa.string()),
        ("content", pa.string()),
        ("raw_code", pa.string()),
        ("start_line", pa.int32()),
        ("end_line", pa.int32()),
        ("header_path", pa.list_(pa.string())),
    ]
)


class DeltaSilverStorage(SilverStorage):
    """Adaptador de infraestructura para la capa Silver utilizando Delta Lake (delta-rs)."""

    def __init__(
        self,
        table_uri: str | None = None,
        storage_options: dict[str, str] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if table_uri:
            self._table_uri = table_uri
            self._storage_options = storage_options or {}
        else:
            self._table_uri = f"s3://{self._settings.minio_silver_bucket}/github/{self._settings.github_repo_name}/delta"
            protocol = "http" if not self._settings.minio_secure else "https"
            endpoint = f"{protocol}://{self._settings.minio_endpoint}"
            self._storage_options = storage_options or {
                "AWS_ENDPOINT_URL": endpoint,
                "AWS_ACCESS_KEY_ID": self._settings.minio_access_key,
                "AWS_SECRET_ACCESS_KEY": self._settings.minio_secret_key,
                "AWS_ALLOW_HTTP": "true" if not self._settings.minio_secure else "false",
                "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
            }

    async def append_chunks(self, chunks: list[CodeChunk | DocChunk]) -> str:
        """Escribe un lote de chunks normalizados en la tabla Delta."""
        if not chunks:
            return self._table_uri

        rows: list[dict[str, Any]] = []
        for ch in chunks:
            meta = ch.metadata
            row: dict[str, Any] = {
                "chunk_id": str(meta.chunk_id),
                "parent_chunk_id": str(meta.parent_chunk_id) if meta.parent_chunk_id else None,
                "file_path": meta.file_path,
                "canonical_import": meta.canonical_import,
                "module_name": meta.module_name,
                "language": meta.language,
                "entity_type": meta.entity_type,
                "entity_name": meta.entity_name,
                "is_parent_class": meta.is_parent_class,
                "commit_sha": meta.commit_sha,
                "commit_date": meta.commit_date,
                "release_version": meta.release_version,
                "semver_major": meta.semver_major,
                "semver_minor": meta.semver_minor,
                "pr_number": meta.pr_number,
                "pr_title": meta.pr_title,
                "author": meta.author,
                "is_deprecated": meta.is_deprecated,
                "breaking_change": meta.breaking_change,
                "content_hash": meta.content_hash,
                "content": ch.content,
                "raw_code": ch.raw_code if isinstance(ch, CodeChunk) else None,
                "start_line": ch.start_line,
                "end_line": ch.end_line,
                "header_path": list(ch.header_path) if isinstance(ch, DocChunk) else None,
            }
            rows.append(row)

        try:
            arrow_table = pa.Table.from_pylist(rows, schema=SILVER_SCHEMA)
            write_deltalake(
                table_or_uri=self._table_uri,
                data=arrow_table,
                mode="append",
                schema_mode="merge",
                storage_options=self._storage_options if self._storage_options else None,
            )
            return self._table_uri
        except Exception as e:
            raise StorageError(
                f"Error al escribir chunks en Delta Lake Silver: {e}",
                {"table_uri": self._table_uri, "count": len(chunks)},
            ) from e

    async def get_chunks_by_commit(self, commit_sha: str) -> list[CodeChunk | DocChunk]:
        """Recupera los chunks asociados a un commit específico desde Silver."""
        try:
            dt = DeltaTable(
                table_uri=self._table_uri,
                storage_options=self._storage_options if self._storage_options else None,
            )
            pyarrow_dataset = dt.to_pyarrow_dataset()
            filtered = pyarrow_dataset.to_table(
                filter=(pa.compute.field("commit_sha") == commit_sha)
            )
            pylist = filtered.to_pylist()

            result: list[CodeChunk | DocChunk] = []
            for row in pylist:
                meta = ChunkMetadata(
                    chunk_id=uuid.UUID(row["chunk_id"]),
                    parent_chunk_id=uuid.UUID(row["parent_chunk_id"])
                    if row["parent_chunk_id"]
                    else None,
                    file_path=row["file_path"],
                    canonical_import=row["canonical_import"],
                    module_name=row["module_name"],
                    language=row["language"],
                    entity_type=row["entity_type"],
                    entity_name=row["entity_name"],
                    is_parent_class=row["is_parent_class"],
                    commit_sha=row["commit_sha"],
                    commit_date=row["commit_date"],
                    release_version=row["release_version"],
                    semver_major=row["semver_major"],
                    semver_minor=row["semver_minor"],
                    pr_number=row["pr_number"],
                    pr_title=row["pr_title"],
                    author=row["author"],
                    is_deprecated=row["is_deprecated"],
                    breaking_change=row["breaking_change"],
                    content_hash=row["content_hash"],
                )
                if row["language"] in ("markdown", "mdx"):
                    doc_chunk = DocChunk(
                        content=row["content"],
                        header_path=tuple(row["header_path"]) if row["header_path"] else (),
                        start_line=row["start_line"],
                        end_line=row["end_line"],
                        metadata=meta,
                    )
                    result.append(doc_chunk)
                else:
                    code_chunk = CodeChunk(
                        content=row["content"],
                        raw_code=row["raw_code"] or row["content"],
                        start_line=row["start_line"],
                        end_line=row["end_line"],
                        metadata=meta,
                    )
                    result.append(code_chunk)
            return result
        except Exception as e:
            raise StorageError(
                f"Error al leer chunks por commit en Delta Lake: {e}",
                {"commit_sha": commit_sha, "table_uri": self._table_uri},
            ) from e
