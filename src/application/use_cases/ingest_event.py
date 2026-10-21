import json
import logging
from datetime import UTC, datetime
from typing import Any

from config.settings import Settings, get_settings
from src.application.splitters.ast_python import ASTPythonSplitter
from src.application.splitters.canonical_resolver import CanonicalImportResolver
from src.application.splitters.mdx_parser import MDXParser
from src.domain.models.chunk import CodeChunk, DocChunk
from src.domain.ports.embedder import EmbeddingEngine
from src.domain.ports.event_queue import EventQueue
from src.domain.ports.object_store import SilverStorage
from src.domain.ports.vector_store import VectorRepository
from src.infrastructure.github.shallow_git import ShallowGitEngine
from src.infrastructure.security.secret_scrubber import SecretScrubber

logger = logging.getLogger(__name__)


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
        resolver: CanonicalImportResolver | None = None,
        scrubber: SecretScrubber | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._silver_storage = silver_storage
        self._vector_repo = vector_repo
        self._embedding_engine = embedding_engine
        self._event_queue = event_queue
        self._shallow_git = shallow_git or ShallowGitEngine()
        self._scrubber = scrubber or SecretScrubber()
        self._resolver = resolver or CanonicalImportResolver()
        self._ast_splitter = ast_splitter or ASTPythonSplitter(
            resolver=self._resolver, scrubber=self._scrubber
        )
        self._mdx_parser = mdx_parser or MDXParser(scrubber=self._scrubber)
        self._settings = settings or get_settings()

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
        except Exception as e:
            logger.error("JSON malformado en mensaje de stream %s: %s", message_id, e)
            if self._event_queue:
                await self._event_queue.ack_event(stream_name, group_name, message_id)
            return False

        # El evento encolado puede tener estructura interna:
        action = event_data.get("action")
        pr_data = event_data.get("pull_request") or event_data.get("payload", {}).get(
            "pull_request"
        )

        # Solo procesar PRs que han sido cerrados y fusionados (merged == True)
        if not pr_data or action != "closed" or not pr_data.get("merged", False):
            logger.info("Ignorando evento no relevante (no es PR merged): action=%s", action)
            if self._event_queue:
                await self._event_queue.ack_event(stream_name, group_name, message_id)
            return True

        collection_name = self._settings.qdrant_collection_active
        await self._vector_repo.ensure_collection(collection_name)

        pr_number = pr_data.get("number", 0)
        pr_title = pr_data.get("title", "")
        pr_author = (pr_data.get("user") or {}).get("login", "unknown")
        merge_sha = pr_data.get("merge_commit_sha") or ("0" * 40)
        merged_at_raw = pr_data.get("merged_at")
        merged_at = (
            datetime.fromisoformat(merged_at_raw.replace("Z", "+00:00"))
            if merged_at_raw
            else datetime.now(UTC)
        )

        files_list = pr_data.get("files") or []
        chunks_to_persist: list[CodeChunk | DocChunk] = []

        from src.infrastructure.telemetry.metrics import get_metrics

        metrics = get_metrics()

        with metrics.measure_time("pipeline_processing_time_seconds", labels={"phase": "chunk"}):
            for f in files_list:
                file_path = f.get("filename") or f.get("path") or ""
                file_status = f.get("status", "modified").upper()
                patch_content = f.get("patch", "")

                if self._shallow_git.is_excluded_file(file_path):
                    continue

                if not any(file_path.startswith(prefix) for prefix in target_prefixes):
                    continue

                # 1. Tombstoning de archivos eliminados (REMOVED / DELETED)
                if file_status in ("REMOVED", "DELETED"):
                    logger.info("Aplicando tombstoning para archivo eliminado: %s", file_path)
                    await self._vector_repo.delete_by_file_path(collection_name, file_path)
                    continue

                # 2. Idempotencia a nivel de archivo: borrar vectores previos antes de reinsertar
                await self._vector_repo.delete_by_file_path(collection_name, file_path)

                if not patch_content:
                    continue

                # 3. Particionado sintáctico según tipo
                if file_path.endswith(".py"):
                    if file_path.endswith("__init__.py"):
                        self._resolver.register_init_file(file_path, patch_content)

                    py_chunks = self._ast_splitter.split_file(
                        file_path=file_path,
                        source_code=patch_content,
                        commit_sha=merge_sha,
                        commit_date=merged_at,
                        release_version="main",
                        pr_number=pr_number,
                        pr_title=pr_title,
                        author=pr_author,
                    )
                    chunks_to_persist.extend(py_chunks)

                elif file_path.endswith((".md", ".mdx")):
                    doc_chunks = self._mdx_parser.parse_document(
                        file_path=file_path,
                        text_content=patch_content,
                        commit_sha=merge_sha,
                        commit_date=merged_at,
                        release_version="main",
                        pr_number=pr_number,
                        pr_title=pr_title,
                        author=pr_author,
                    )
                    chunks_to_persist.extend(doc_chunks)

        if chunks_to_persist:
            # 4. Guardar en Capa Silver (Delta Lake)
            await self._silver_storage.append_chunks(chunks_to_persist)

            # 5. Generar embeddings y hacer Upsert en Gold (Qdrant)
            with metrics.measure_time(
                "pipeline_processing_time_seconds", labels={"phase": "embed"}
            ):
                texts = [c.content for c in chunks_to_persist]
                dense_vectors = self._embedding_engine.embed_dense(texts)
                sparse_vectors = self._embedding_engine.embed_sparse(texts)

            with metrics.measure_time(
                "pipeline_processing_time_seconds", labels={"phase": "upsert"}
            ):
                inserted = await self._vector_repo.upsert_chunks(
                    collection_name=collection_name,
                    chunks=chunks_to_persist,
                    dense_vectors=dense_vectors,
                    sparse_vectors=sparse_vectors,
                )

            metrics.set_gauge(
                "active_vectors_count", float(inserted), labels={"collection": collection_name}
            )
            logger.info(
                "PR #%d procesado con éxito: %d chunks indexados en Gold",
                pr_number,
                len(chunks_to_persist),
            )

        # 6. Confirmar mensaje en el Consumer Group de Redis Streams (XACK)
        if self._event_queue:
            await self._event_queue.ack_event(stream_name, group_name, message_id)

        return True
