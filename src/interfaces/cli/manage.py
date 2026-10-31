import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from config.settings import get_settings
from src.application.use_cases.backfill import BackfillRequest, BackfillUseCase
from src.application.use_cases.ingest_event import IngestEventUseCase
from src.infrastructure.embeddings.bge_m3_local import FastembedEmbeddingAdapter
from src.infrastructure.github.graphql_client import GitHubGraphQLClient
from src.infrastructure.queue.redis_streams import RedisStreamsAdapter
from src.infrastructure.state.redis_state import RedisStateStore
from src.infrastructure.storage.delta_silver import DeltaSilverStorage
from src.infrastructure.storage.minio_bronze import MinioBronzeStorage
from src.infrastructure.vector.qdrant_adapter import QdrantVectorAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("radar-cli")


async def run_backfill(args: argparse.Namespace) -> None:
    """Ejecuta el backfill histórico desde la línea de comandos."""
    settings = get_settings()
    logger.info("Iniciando backfill para %s/%s...", args.owner, args.repo)

    github_client = GitHubGraphQLClient(settings=settings)
    bronze_storage = MinioBronzeStorage(settings=settings)
    silver_storage = DeltaSilverStorage(settings=settings)
    vector_repo = QdrantVectorAdapter(settings=settings)
    embedding_engine = FastembedEmbeddingAdapter(settings=settings)
    state_store = RedisStateStore()

    use_case = BackfillUseCase(
        github_client=github_client,
        bronze_storage=bronze_storage,
        silver_storage=silver_storage,
        vector_repo=vector_repo,
        embedding_engine=embedding_engine,
        state_store=state_store,
        settings=settings,
    )

    request = BackfillRequest(
        repo_owner=args.owner,
        repo_name=args.repo,
        max_prs=args.max_prs,
        batch_size=args.batch_size,
        collection_name=args.collection,
    )

    summary = await use_case.execute(request)
    logger.info("=== Backfill Completado con Éxito ===")
    logger.info("PRs procesados: %d", summary.prs_processed)
    logger.info("Archivos analizados: %d", summary.files_processed)
    logger.info("Chunks generados: %d", summary.chunks_created)
    logger.info("Chunks indexados en Qdrant: %d", summary.chunks_indexed)
    logger.info("Chunks omitidos por Content Hash: %d", summary.chunks_skipped_by_hash)
    logger.info("Tiempo total: %.2f segundos", summary.duration_seconds)


def run_serve(args: argparse.Namespace) -> None:
    """Inicia el servidor API FastAPI."""
    import uvicorn

    logger.info("Iniciando API FastAPI en %s:%d (reload=%s)...", args.host, args.port, args.reload)
    uvicorn.run("src.interfaces.api.app:app", host=args.host, port=args.port, reload=args.reload)


async def run_worker(args: argparse.Namespace) -> None:
    """Bucle continuo de consumo de eventos de GitHub desde Redis Streams."""
    settings = get_settings()
    logger.info("Iniciando background worker consumidor de eventos...")
    queue = RedisStreamsAdapter(settings=settings)
    silver = DeltaSilverStorage(settings=settings)
    vector = QdrantVectorAdapter(settings=settings)
    embedder = FastembedEmbeddingAdapter(settings=settings)

    stream_name = settings.redis_stream_key
    group_name = settings.redis_consumer_group
    await queue.ensure_consumer_group(stream_name=stream_name, group_name=group_name)

    use_case = IngestEventUseCase(
        silver_storage=silver,
        vector_repo=vector,
        embedding_engine=embedder,
        event_queue=queue,
        state_store=RedisStateStore(),
        settings=settings,
    )

    logger.info(
        "Worker '%s' escuchando en stream '%s' (grupo '%s')...",
        args.consumer_name,
        stream_name,
        group_name,
    )

    while True:
        try:
            messages = await queue.read_events(
                stream_name=stream_name,
                group_name=group_name,
                consumer_name=args.consumer_name,
                count=args.batch_size,
                block_ms=args.block_ms,
            )
            for msg_id, fields in messages:
                logger.info("Procesando mensaje %s...", msg_id)
                success = await use_case.process_stream_message(
                    stream_name=stream_name,
                    group_name=group_name,
                    message_id=msg_id,
                    fields=fields,
                )
                logger.info("Mensaje %s procesado (éxito: %s)", msg_id, success)
            if args.once:
                break
        except asyncio.CancelledError:
            logger.info("Worker detenido por señal de cancelación.")
            break
        except Exception as e:
            logger.error("Error en ciclo de worker: %s", e, exc_info=True)
            if args.once:
                raise
            await asyncio.sleep(2)


async def run_reindex(args: argparse.Namespace) -> None:
    """Ejecuta una reindexación Blue/Green creando una colección staging."""
    settings = get_settings()
    tag = args.version_tag or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    staging_collection = f"{settings.qdrant_collection_prefix}_{tag}"
    vector_repo = QdrantVectorAdapter(settings=settings)

    logger.info("Creando colección staging: %s", staging_collection)
    await vector_repo.ensure_collection(staging_collection)

    if args.swap_alias:
        logger.info(
            "Conmutando alias '%s' hacia '%s'...",
            settings.qdrant_collection_active,
            staging_collection,
        )
        await vector_repo.switch_alias(
            alias_name=settings.qdrant_collection_active,
            target_collection=staging_collection,
            old_collection=None,
        )
        logger.info("Conmutación Blue/Green de alias completada.")


async def run_eval(args: argparse.Namespace) -> None:
    """Ejecuta el benchmark de evaluación RAG contra el dataset indicado."""
    from tests.evaluation.run_rag_eval import run_evaluation

    dataset_path = Path(args.dataset)
    logger.info("Iniciando evaluación RAG con dataset: %s", dataset_path)
    report = await run_evaluation(dataset_path)
    logger.info("Resultado de evaluación: %s", report)


async def run_status(_args: argparse.Namespace) -> None:
    """Muestra el estado de sincronización y métricas del High-Water Mark."""
    settings = get_settings()
    state_store = RedisStateStore()
    cp_backfill = await state_store.get_checkpoint("backfill_cursor")
    cp_stream = await state_store.get_checkpoint("stream_consumer")
    total_prs = await state_store.get_processed_pr_count()

    print("=== Estado de LangChain Radar ===")
    print(f"Entorno: {settings.app_env}")
    print(f"Colección activa Qdrant: {settings.qdrant_collection_active}")
    print(f"Stream Redis: {settings.redis_stream_key}")
    print(f"Total PRs procesados: {total_prs}")
    if cp_backfill:
        print(
            f"Backfill - Último commit: {cp_backfill.last_commit_sha} | "
            f"Sincronizado: {cp_backfill.last_synced_at.isoformat()}"
        )
    if cp_stream:
        print(f"Stream - Último commit: {cp_stream.last_commit_sha} | Mensaje: {cp_stream.cursor}")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="LangChain Radar - CLI de Gestión, Ingesta y Operaciones",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: backfill
    backfill_parser = subparsers.add_parser("backfill", help="Ejecutar backfill histórico")
    backfill_parser.add_argument(
        "--owner", default=settings.github_repo_owner, help="Propietario del repositorio"
    )
    backfill_parser.add_argument(
        "--repo", default=settings.github_repo_name, help="Nombre del repositorio"
    )
    backfill_parser.add_argument(
        "--max-prs", type=int, default=20, help="Número máximo de PRs a sincronizar"
    )
    backfill_parser.add_argument(
        "--batch-size", type=int, default=64, help="Tamaño de lote para vectorización"
    )
    backfill_parser.add_argument(
        "--collection",
        default=settings.qdrant_collection_active,
        help="Colección destino en Qdrant",
    )

    # Subcomando: serve
    serve_parser = subparsers.add_parser("serve", help="Iniciar servidor API REST (FastAPI)")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Dirección IP de bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Puerto HTTP")
    serve_parser.add_argument(
        "--reload", action="store_true", help="Recarga en caliente para desarrollo"
    )

    # Subcomando: worker
    worker_parser = subparsers.add_parser(
        "worker", help="Iniciar worker consumidor de Redis Streams"
    )
    worker_parser.add_argument(
        "--consumer-name", default="worker-1", help="Nombre identificador del consumidor"
    )
    worker_parser.add_argument(
        "--batch-size", type=int, default=10, help="Máximo de mensajes por lote"
    )
    worker_parser.add_argument(
        "--block-ms", type=int, default=2000, help="Tiempo de espera en milisegundos (XREADGROUP)"
    )
    worker_parser.add_argument(
        "--once", action="store_true", help="Procesar un solo lote y salir (para testing)"
    )

    # Subcomando: reindex
    reindex_parser = subparsers.add_parser(
        "reindex", help="Crear colección staging y conmutar alias Blue/Green"
    )
    reindex_parser.add_argument(
        "--version-tag", default=None, help="Tag de versión para la colección staging"
    )
    reindex_parser.add_argument(
        "--no-swap",
        dest="swap_alias",
        action="store_false",
        help="No conmutar el alias tras crear la colección",
    )

    # Subcomando: eval
    eval_parser = subparsers.add_parser("eval", help="Ejecutar benchmark de evaluación RAG")
    eval_parser.add_argument(
        "--dataset",
        default="tests/evaluation/golden_dataset.json",
        help="Ruta al fichero JSON con preguntas y ground truth",
    )

    # Subcomando: status
    subparsers.add_parser("status", help="Consultar estado de sincronización y checkpoints")

    args = parser.parse_args()

    if args.command == "backfill":
        asyncio.run(run_backfill(args))
    elif args.command == "serve":
        run_serve(args)
    elif args.command == "worker":
        asyncio.run(run_worker(args))
    elif args.command == "reindex":
        asyncio.run(run_reindex(args))
    elif args.command == "eval":
        asyncio.run(run_eval(args))
    elif args.command == "status":
        asyncio.run(run_status(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
