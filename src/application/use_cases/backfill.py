import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from config.settings import Settings, get_settings
from src.application.splitters.ast_python import ASTPythonSplitter
from src.application.splitters.canonical_resolver import CanonicalImportResolver
from src.application.splitters.mdx_parser import MDXParser
from src.domain.models.chunk import CodeChunk, DocChunk
from src.domain.models.state import SyncCheckpoint
from src.domain.ports.embedder import EmbeddingEngine
from src.domain.ports.object_store import BronzeStorage, SilverStorage
from src.domain.ports.state_store import StateStore
from src.domain.ports.vector_store import VectorRepository
from src.infrastructure.github.graphql_client import GitHubGraphQLClient
from src.infrastructure.github.shallow_git import ShallowGitEngine
from src.infrastructure.security.secret_scrubber import SecretScrubber

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillRequest:
    """Parámetros de entrada para el caso de uso de backfilling histórico."""

    repo_owner: str | None = None
    repo_name: str | None = None
    target_prefixes: tuple[str, ...] = ("libs/core/", "docs/")
    max_prs: int = 50
    batch_size: int = 64
    collection_name: str | None = None


@dataclass(frozen=True)
class BackfillSummary:
    """Métricas y resultado consolidado de la ejecución del backfilling."""

    prs_processed: int
    files_processed: int
    chunks_created: int
    chunks_indexed: int
    chunks_skipped_by_hash: int
    duration_seconds: float


class BackfillUseCase:
    """Orquesta la sincronización y vectorización histórica de código y docs hacia Bronze, Silver y Gold."""

    def __init__(
        self,
        github_client: GitHubGraphQLClient,
        bronze_storage: BronzeStorage,
        silver_storage: SilverStorage,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        shallow_git: ShallowGitEngine | None = None,
        ast_splitter: ASTPythonSplitter | None = None,
        mdx_parser: MDXParser | None = None,
        resolver: CanonicalImportResolver | None = None,
        scrubber: SecretScrubber | None = None,
        state_store: StateStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._github_client = github_client
        self._bronze_storage = bronze_storage
        self._silver_storage = silver_storage
        self._vector_repo = vector_repo
        self._embedding_engine = embedding_engine
        self._state_store = state_store
        self._shallow_git = shallow_git or ShallowGitEngine()
        self._scrubber = scrubber or SecretScrubber()
        self._resolver = resolver or CanonicalImportResolver()
        self._ast_splitter = ast_splitter or ASTPythonSplitter(
            resolver=self._resolver, scrubber=self._scrubber
        )
        self._mdx_parser = mdx_parser or MDXParser(scrubber=self._scrubber)
        self._settings = settings or get_settings()

    async def execute(self, request: BackfillRequest) -> BackfillSummary:
        """Ejecuta el pipeline completo de backfilling batch."""
        start_time = time.monotonic()
        target_collection = request.collection_name or self._settings.qdrant_collection_active

        # Asegurar que la colección existe en Gold
        await self._vector_repo.ensure_collection(target_collection)

        prs_processed = 0
        files_processed = 0
        chunks_created = 0
        chunks_indexed = 0
        chunks_skipped_by_hash = 0

        cursor: str | None = None
        if self._state_store:
            saved_cp = await self._state_store.get_checkpoint("backfill_cursor")
            if saved_cp and saved_cp.cursor:
                cursor = saved_cp.cursor
                logger.info("Reanudando backfill desde cursor guardado: %s", cursor)

        has_next = True
        seen_hashes: set[str] = set()

        while has_next and prs_processed < request.max_prs:
            fetch_size = min(20, request.max_prs - prs_processed)
            prs, cursor, has_next = await self._github_client.fetch_merged_prs(
                owner=request.repo_owner,
                repo=request.repo_name,
                page_size=fetch_size,
                cursor=cursor,
            )

            if not prs:
                break

            for pr in prs:
                if self._state_store and await self._state_store.is_pr_processed(pr.pr_number):
                    logger.info("PR #%d ya procesado según StateStore. Omitiendo.", pr.pr_number)
                    continue

                # 1. Guardar raw event en Bronze (inmutable)
                raw_payload = json.dumps(pr.model_dump(), default=str)
                await self._bronze_storage.save_raw_event(
                    event_id=f"pr_{pr.pr_number}_{pr.merge_commit_sha[:8]}",
                    year=pr.merged_at.year,
                    month=pr.merged_at.month,
                    day=pr.merged_at.day,
                    raw_payload=raw_payload,
                )

                pr_chunks: list[CodeChunk | DocChunk] = []

                # 2. Procesar archivos modificados
                for file_change in pr.files_changed:
                    path = file_change.file_path
                    if self._shallow_git.is_excluded_file(path):
                        continue

                    # Filtrar carpetas objetivo
                    if not any(path.startswith(prefix) for prefix in request.target_prefixes):
                        continue

                    patch_content = file_change.patch or ""
                    if not patch_content:
                        continue

                    files_processed += 1

                    if path.endswith(".py"):
                        # Si es un __init__.py, registrarlo en el resolver
                        if path.endswith("__init__.py"):
                            self._resolver.register_init_file(path, patch_content)

                        parsed_chunks = self._ast_splitter.split_file(
                            file_path=path,
                            source_code=patch_content,
                            commit_sha=pr.merge_commit_sha,
                            commit_date=pr.merged_at,
                            release_version="main",
                            pr_number=pr.pr_number,
                            pr_title=pr.title,
                            author=pr.author,
                            breaking_change=pr.is_breaking_change,
                        )
                        pr_chunks.extend(parsed_chunks)

                    elif path.endswith((".md", ".mdx")):
                        doc_chunks = self._mdx_parser.parse_document(
                            file_path=path,
                            text_content=patch_content,
                            commit_sha=pr.merge_commit_sha,
                            commit_date=pr.merged_at,
                            release_version="main",
                            pr_number=pr.pr_number,
                            pr_title=pr.title,
                            author=pr.author,
                        )
                        pr_chunks.extend(doc_chunks)

                chunks_created += len(pr_chunks)

                if pr_chunks:
                    # 3. Persistir en Silver (Delta Lake)
                    await self._silver_storage.append_chunks(pr_chunks)

                    # 4. Filtrar por content_hash para evitar recalcular embeddings (60-80% ahorro)
                    to_embed: list[CodeChunk | DocChunk] = []
                    for ch in pr_chunks:
                        h = ch.metadata.content_hash
                        if h in seen_hashes:
                            chunks_skipped_by_hash += 1
                        else:
                            seen_hashes.add(h)
                            to_embed.append(ch)

                    # 5. Generar embeddings y hacer Upsert por lotes en Gold (Qdrant)
                    for i in range(0, len(to_embed), request.batch_size):
                        batch = to_embed[i : i + request.batch_size]
                        texts = [c.content for c in batch]

                        dense_vectors = self._embedding_engine.embed_dense(texts)
                        sparse_vectors = self._embedding_engine.embed_sparse(texts)

                        inserted = await self._vector_repo.upsert_chunks(
                            collection_name=target_collection,
                            chunks=batch,
                            dense_vectors=dense_vectors,
                            sparse_vectors=sparse_vectors,
                        )
                        chunks_indexed += inserted
                if self._state_store:
                    await self._state_store.record_processed_pr(pr.pr_number, pr.merge_commit_sha)

                prs_processed += 1

            if self._state_store and prs:
                last_pr = prs[-1]
                await self._state_store.save_checkpoint(
                    SyncCheckpoint(
                        checkpoint_id="backfill_cursor",
                        last_commit_sha=last_pr.merge_commit_sha,
                        cursor=cursor,
                        last_synced_at=datetime.now(UTC),
                        total_processed_items=prs_processed,
                        metadata={"collection": target_collection},
                    )
                )

        duration = time.monotonic() - start_time
        return BackfillSummary(
            prs_processed=prs_processed,
            files_processed=files_processed,
            chunks_created=chunks_created,
            chunks_indexed=chunks_indexed,
            chunks_skipped_by_hash=chunks_skipped_by_hash,
            duration_seconds=duration,
        )
