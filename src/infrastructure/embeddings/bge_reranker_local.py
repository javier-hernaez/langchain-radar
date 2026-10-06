from fastembed.rerank.cross_encoder import TextCrossEncoder

from config.settings import Settings, get_settings
from src.domain.models.search import SearchResultItem
from src.domain.ports.reranker import RerankerEngine


class FastembedRerankerAdapter(RerankerEngine):
    """Adaptador de Cross-Encoder local en CPU para re-ranking semántico profundo."""

    def __init__(
        self,
        model_name: str | None = None,
        settings: Settings | None = None,
        lazy_load: bool = True,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_name = model_name or "BAAI/bge-reranker-base"
        self._model: TextCrossEncoder | None = None

        if not lazy_load:
            self._init_model()

    def _init_model(self) -> None:
        if self._model is None:
            self._model = TextCrossEncoder(model_name=self._model_name)

    def rerank(
        self, query: str, candidates: list[SearchResultItem], top_k: int = 5
    ) -> list[SearchResultItem]:
        """Calcula scores de atención cruzada (Query, Chunk) y devuelve el top_k ordenado."""
        if not candidates:
            return []

        if len(candidates) <= 1:
            return candidates[:top_k]

        self._init_model()
        assert self._model is not None

        docs = [item.content for item in candidates]
        scores_iter = self._model.rerank(query=query, documents=docs, batch_size=32)
        scores = [float(s) for s in scores_iter]

        scored_items: list[tuple[float, SearchResultItem]] = []
        for score, item in zip(scores, candidates, strict=False):
            updated_item = item.model_copy(update={"rerank_score": score})
            scored_items.append((score, updated_item))

        # Ordenar de mayor a menor puntuación
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:top_k]]
