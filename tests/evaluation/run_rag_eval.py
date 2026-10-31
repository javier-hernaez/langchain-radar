import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from src.application.use_cases.search_code import SearchCodeUseCase
from src.domain.models.search import SearchFilters, SearchQuery
from src.infrastructure.embeddings.bge_m3_local import FastembedEmbeddingAdapter
from src.infrastructure.embeddings.bge_reranker_local import FastembedRerankerAdapter
from src.infrastructure.vector.qdrant_adapter import QdrantVectorAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-eval")


def evaluate_syntactic_validity(import_statement: str) -> bool:
    """Valida sintácticamente la sentencia de importación con compile() de Python."""
    try:
        compile(import_statement, "<eval_snippet>", "exec")
        return True
    except SyntaxError:
        return False


async def run_evaluation(dataset_path: Path) -> dict[str, Any]:
    """Ejecuta el benchmark contra el dataset dorado de evaluación."""
    with open(dataset_path, encoding="utf-8") as f:
        cases: list[dict[str, Any]] = json.load(f)

    logger.info("Cargados %d casos de prueba desde %s", len(cases), dataset_path)

    vector_repo = QdrantVectorAdapter()
    embedding_engine = FastembedEmbeddingAdapter(lazy_load=True)
    reranker_engine = FastembedRerankerAdapter(lazy_load=True)

    use_case = SearchCodeUseCase(
        vector_repo=vector_repo,
        embedding_engine=embedding_engine,
        reranker_engine=reranker_engine,
    )

    hits_at_5 = 0
    mrr_sum = 0.0
    syntactic_valid_count = 0
    total = len(cases)

    for item in cases:
        q_id = item["id"]
        question = item["question"]
        expected_symbol = item["expected_symbol"]
        expected_import = item["expected_canonical_import"]

        # Validar sintaxis del ground truth esperado
        if evaluate_syntactic_validity(expected_import):
            syntactic_valid_count += 1

        query = SearchQuery(
            query=question,
            limit=5,
            filters=SearchFilters(),
            search_mode="hybrid",
            use_reranker=True,
            include_parent_context=True,
        )

        try:
            response = await use_case.execute(query)
            results = response.results
        except Exception as e:
            logger.warning("Fallo en ejecución de búsqueda para %s: %s", q_id, e)
            results = ()

        # Calcular Hit Rate @ 5 y Reciprocal Rank
        found_rank: int | None = None
        for rank, res in enumerate(results, start=1):
            if expected_symbol in res.entity_name or expected_import in res.canonical_import:
                found_rank = rank
                break

        if found_rank is not None:
            hits_at_5 += 1
            mrr_sum += 1.0 / found_rank
            logger.info(
                "Caso %s: ACERTO en rango %d (MRR: %.2f)", q_id, found_rank, 1.0 / found_rank
            )
        else:
            logger.info("Caso %s: No encontrado en Top 5", q_id)

    hit_rate_5 = hits_at_5 / total if total > 0 else 0.0
    mrr = mrr_sum / total if total > 0 else 0.0
    syntactic_validity = (syntactic_valid_count / total) * 100.0 if total > 0 else 0.0

    report = {
        "total_cases": total,
        "hit_rate_at_5": round(hit_rate_5, 4),
        "mean_reciprocal_rank": round(mrr, 4),
        "syntactic_validity_percent": round(syntactic_validity, 2),
    }

    logger.info("=== Informe Final de Evaluación RAG ===")
    logger.info("Total de Casos Evaluados: %d", report["total_cases"])
    logger.info("Hit Rate @ 5: %.2f%%", report["hit_rate_at_5"] * 100.0)
    logger.info("MRR (Mean Reciprocal Rank): %.4f", report["mean_reciprocal_rank"])
    logger.info("Validez Sintáctica: %.2f%%", report["syntactic_validity_percent"])

    return report


if __name__ == "__main__":
    dataset_file = Path(__file__).parent / "golden_dataset.json"
    asyncio.run(run_evaluation(dataset_file))
