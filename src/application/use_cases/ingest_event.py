import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from config.settings import Settings, get_settings
from src.application.splitters.ast_python import ASTPythonSplitter
from src.application.splitters.canonical_resolver import CanonicalImportResolver
from src.application.splitters.diff_splitter import GitDiffSplitter
from src.application.splitters.mdx_parser import MDXParser
from src.application.splitters.source_chunker import SourceFileChunker
from src.domain.models.chunk import CodeChunk, DocChunk
from src.domain.models.events import CommitFileChange
from src.domain.models.state import SyncCheckpoint
from src.domain.ports.embedder import EmbeddingEngine
from src.domain.ports.event_queue import EventQueue
from src.domain.ports.object_store import SilverStorage
from src.domain.ports.source_host import SourceCodeHost
from src.domain.ports.state_store import StateStore
from src.domain.ports.vector_store import VectorRepository
from src.infrastructure.github.shallow_git import ShallowGitEngine
from src.infrastructure.security.secret_scrubber import SecretScrubber

logger = logging.getLogger(__name__)

# Estados de fichero del webhook / API REST de GitHub -> estados del dominio
_FILE_STATUS_MAP: dict[str, str] = {
    "added": "ADDED",
    "copied": "ADDED",
    "removed": "REMOVED",
    "deleted": "REMOVED",
    "renamed": "RENAMED",
}


class IngestEventUseCase:
    """Caso de uso para procesamiento reactivo en background de eventos de GitHub encolados."""

    def __init__(
        self,
        silver_storage: SilverStorage,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        event_queue: EventQueue | None = None,
        shallow_git: ShallowGitEngine | None = None,
        ast_splitter: ASTPythonSplitter | None = None,
        mdx_parser: MDXParser | None = None,
        diff_splitter: GitDiffSplitter | None = None,
        resolver: CanonicalImportResolver | None = None,
        scrubber: SecretScrubber | None = None,
        state_store: StateStore | None = None,
        settings: Settings | None = None,
        source_host: SourceCodeHost | None = None,
    ) -> None:
        self._silver_storage = silver_storage
        self._vector_repo = vector_repo
        self._embedding_engine = embedding_engine
        self._event_queue = event_queue
        self._state_store = state_store
        # Sin source_host solo se indexan los hunks de diff presentes en el payload
        self._source_host = source_host
        self._shallow_git = shallow_git or ShallowGitEngine()
        self._scrubber = scrubber or SecretScrubber()
        self._resolver = resolver or CanonicalImportResolver()
        self._ast_splitter = ast_splitter or ASTPythonSplitter(
            resolver=self._resolver, scrubber=self._scrubber
        )
        self._mdx_parser = mdx_parser or MDXParser(scrubber=self._scrubber)
        self._diff_splitter = diff_splitter or GitDiffSplitter(
            resolver=self._resolver, scrubber=self._scrubber
        )
        self._chunker = SourceFileChunker(self._ast_splitter, self._mdx_parser, self._resolver)
        self._settings = settings or get_settings()

    def _repo_coordinates(self, event_data: dict[str, Any]) -> tuple[str, str]:
        """Extrae owner/repo del payload del webhook, con fallback a la configuración."""
        payload = event_data.get("payload")
        repository = event_data.get("repository") or (
            payload.get("repository") if isinstance(payload, dict) else None
        )
        full_name = (repository or {}).get("full_name") or ""
        if "/" in full_name:
            owner, repo = full_name.split("/", 1)
            return owner, repo
        return self._settings.github_repo_owner, self._settings.github_repo_name

    @staticmethod
    def _files_from_payload(raw_files: list[dict[str, Any]]) -> list[CommitFileChange]:
        files: list[CommitFileChange] = []
        for f in raw_files:
            raw_status = str(f.get("status", "modified")).lower()
            files.append(
                CommitFileChange(
                    file_path=f.get("filename") or f.get("path") or "",
                    status=_FILE_STATUS_MAP.get(raw_status, "MODIFIED"),  # type: ignore[arg-type]
                    patch=f.get("patch") or None,
                    previous_filename=f.get("previous_filename"),
                )
            )
        return files

    @staticmethod
    def _is_breaking(pr_data: dict[str, Any]) -> bool:
        title = str(pr_data.get("title") or "").lower()
        body = str(pr_data.get("body") or "").lower()
        labels = [
            str(lbl.get("name", "")).lower()
            for lbl in pr_data.get("labels") or []
            if isinstance(lbl, dict)
        ]
        return (
            "breaking change" in title
            or "breaking change" in body
            or any("breaking" in lbl for lbl in labels)
        )

    async def _ack(self, stream_name: str, group_name: str, message_id: str) -> None:
        if self._event_queue:
            await self._event_queue.ack_event(stream_name, group_name, message_id)

    async def process_stream_message(
        self,
        stream_name: str,
        group_name: str,
        message_id: str,
        fields: dict[str, Any],
        target_prefixes: tuple[str, ...] = ("libs/core/", "docs/"),
    ) -> bool:
        """Procesa un mensaje individual extraído de Redis Streams y ejecuta ACK al finalizar."""
        payload_str = fields.get("payload", "{}")
        try:
            event_data = json.loads(payload_str)
        except (TypeError, json.JSONDecodeError) as e:
            logger.error("JSON malformado en mensaje de stream %s: %s", message_id, e)
            await self._ack(stream_name, group_name, message_id)
            return False

        if not isinstance(event_data, dict):
            logger.error("Mensaje %s sin objeto JSON válido", message_id)
            await self._ack(stream_name, group_name, message_id)
            return False

        # El evento encolado envuelve el payload del webhook en "payload"
        inner = event_data.get("payload")
        inner = inner if isinstance(inner, dict) else {}
        action = event_data.get("action") or inner.get("action")
        pr_data = event_data.get("pull_request") or inner.get("pull_request")

        # Solo procesar PRs que han sido cerrados y fusionados (merged == True)
        if not pr_data or action != "closed" or not pr_data.get("merged", False):
            logger.info("Ignorando evento no relevante (no es PR merged): action=%s", action)
            await self._ack(stream_name, group_name, message_id)
            return True

        collection_name = self._settings.qdrant_collection_active
        await self._vector_repo.ensure_collection(collection_name)

        owner, repo = self._repo_coordinates(event_data)
        pr_number = pr_data.get("number", 0)
        pr_title = pr_data.get("title", "")
        pr_author = (pr_data.get("user") or {}).get("login", "unknown")
        real_merge_sha: str | None = pr_data.get("merge_commit_sha")
        merge_sha = real_merge_sha or ("0" * 40)
        is_breaking = self._is_breaking(pr_data)
        merged_at_raw = pr_data.get("merged_at")
        merged_at = (
            datetime.fromisoformat(merged_at_raw.replace("Z", "+00:00"))
            if merged_at_raw
            else datetime.now(UTC)
        )

        # El webhook `pull_request` no incluye la lista de ficheros: se consulta a la API
        raw_files = pr_data.get("files")
        if raw_files is not None:
            files = self._files_from_payload(raw_files)
        elif self._source_host is not None and pr_number:
            files = await self._source_host.fetch_pr_files(owner, repo, pr_number)
        else:
            logger.warning("PR #%s sin lista de ficheros ni SourceCodeHost configurado", pr_number)
            files = []

        chunks_to_persist: list[CodeChunk | DocChunk] = []

        from src.infrastructure.telemetry.metrics import get_metrics

        metrics = get_metrics()

        with metrics.measure_time("pipeline_processing_time_seconds", labels={"phase": "chunk"}):
            for file_change in files:
                file_path = file_change.file_path

                if self._shallow_git.is_excluded_file(file_path):
                    continue

                if not any(file_path.startswith(prefix) for prefix in target_prefixes):
                    continue

                # 1. Tombstoning de archivos eliminados
                if file_change.status == "REMOVED":
                    logger.info("Aplicando tombstoning para archivo eliminado: %s", file_path)
                    await self._vector_repo.delete_by_file_path(collection_name, file_path)
                    continue

                # Un renombrado deja huérfanos los vectores de la ruta anterior
                if file_change.status == "RENAMED" and file_change.previous_filename:
                    await self._vector_repo.delete_by_file_path(
                        collection_name, file_change.previous_filename
                    )

                # 2. Idempotencia a nivel de archivo: borrar vectores previos antes de reinsertar
                await self._vector_repo.delete_by_file_path(collection_name, file_path)

                # 3. Particionado estructural sobre el contenido completo en el commit de merge
                if (
                    self._source_host is not None
                    and real_merge_sha
                    and self._chunker.supports(file_path)
                ):
                    content = await self._source_host.fetch_file_content(
                        owner, repo, file_path, real_merge_sha
                    )
                    if content and content.strip():
                        chunks_to_persist.extend(
                            self._chunker.chunk(
                                file_path=file_path,
                                content=content,
                                commit_sha=merge_sha,
                                commit_date=merged_at,
                                pr_number=pr_number,
                                pr_title=pr_title,
                                author=pr_author,
                                breaking_change=is_breaking,
                            )
                        )

                # 4. Particionado de Diff Semántico (Hunks)
                if file_change.patch:
                    chunks_to_persist.extend(
                        self._diff_splitter.parse_patch_hunks(
                            file_path=file_path,
                            patch_content=file_change.patch,
                            commit_sha=merge_sha,
                            commit_date=merged_at,
                            release_version="main",
                            pr_number=pr_number,
                            pr_title=pr_title,
                            author=pr_author,
                        )
                    )

        if chunks_to_persist:
            # 5. Guardar en Capa Silver (Delta Lake)
            await self._silver_storage.append_chunks(chunks_to_persist)

            # 6. Generar embeddings (fuera del event loop) y hacer Upsert en Gold (Qdrant)
            with metrics.measure_time(
                "pipeline_processing_time_seconds", labels={"phase": "embed"}
            ):
                texts = [c.content for c in chunks_to_persist]
                dense_vectors = await asyncio.to_thread(self._embedding_engine.embed_dense, texts)
                sparse_vectors = await asyncio.to_thread(self._embedding_engine.embed_sparse, texts)

            with metrics.measure_time(
                "pipeline_processing_time_seconds", labels={"phase": "upsert"}
            ):
                inserted = await self._vector_repo.upsert_chunks(
                    collection_name=collection_name,
                    chunks=chunks_to_persist,
                    dense_vectors=dense_vectors,
                    sparse_vectors=sparse_vectors,
                )

            metrics.inc_counter(
                "indexed_chunks_total", float(inserted), labels={"collection": collection_name}
            )
            logger.info(
                "PR #%d procesado con éxito: %d chunks indexados en Gold",
                pr_number,
                len(chunks_to_persist),
            )

        if self._state_store:
            await self._state_store.record_processed_pr(pr_number, merge_sha)
            await self._state_store.save_checkpoint(
                SyncCheckpoint(
                    checkpoint_id="stream_consumer",
                    last_commit_sha=merge_sha,
                    cursor=message_id,
                    last_synced_at=datetime.now(UTC),
                    total_processed_items=await self._state_store.get_processed_pr_count(),
                    metadata={"stream_name": stream_name, "pr_number": pr_number},
                )
            )

        # 7. Confirmar mensaje en el Consumer Group de Redis Streams (XACK)
        await self._ack(stream_name, group_name, message_id)

        return True
