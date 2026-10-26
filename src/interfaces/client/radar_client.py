from typing import Any, Literal

import httpx

from src.domain.models.search import SearchFilters, SearchQuery, SearchResponse


class AsyncRadarClient:
    """Cliente asíncrono para interactuar con la API REST de LangChain Radar."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        admin_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.admin_token = admin_token
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "AsyncRadarClient":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Cierra el cliente HTTP si está abierto."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: SearchFilters | None = None,
        search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
        use_reranker: bool = True,
        include_parent_context: bool = True,
        include_diffs: bool = False,
    ) -> SearchResponse:
        """Ejecuta una búsqueda semántica híbrida con re-ranking."""
        payload = SearchQuery(
            query=query,
            limit=limit,
            filters=filters or SearchFilters(),
            search_mode=search_mode,
            use_reranker=use_reranker,
            include_parent_context=include_parent_context,
            include_diffs=include_diffs,
        ).model_dump(mode="json")

        client = self._get_client()
        resp = await client.post("/api/v1/search", json=payload)
        resp.raise_for_status()
        return SearchResponse.model_validate(resp.json())

    async def get_health(self) -> dict[str, Any]:
        """Consulta el estado de salud de la API y sus componentes."""
        client = self._get_client()
        resp = await client.get("/api/v1/health")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def get_sync_status(self) -> dict[str, Any]:
        """Consulta el estado de sincronización y colección activa."""
        client = self._get_client()
        resp = await client.get("/api/v1/status/sync")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def trigger_reindex(
        self, version_tag: str, admin_token: str | None = None
    ) -> dict[str, Any]:
        """Dispara una re-indexación Blue/Green en Qdrant."""
        token = admin_token or self.admin_token
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        client = self._get_client()
        resp = await client.post(
            "/api/v1/admin/reindex",
            json={"new_version_tag": version_tag},
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]


class RadarClient:
    """Cliente síncrono para interactuar con la API REST de LangChain Radar."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        admin_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.admin_token = admin_token
        self._client: httpx.Client | None = None

    def __enter__(self) -> "RadarClient":
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def close(self) -> None:
        """Cierra el cliente HTTP si está abierto."""
        if self._client:
            self._client.close()
            self._client = None

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: SearchFilters | None = None,
        search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
        use_reranker: bool = True,
        include_parent_context: bool = True,
        include_diffs: bool = False,
    ) -> SearchResponse:
        """Ejecuta una búsqueda semántica híbrida con re-ranking de forma síncrona."""
        payload = SearchQuery(
            query=query,
            limit=limit,
            filters=filters or SearchFilters(),
            search_mode=search_mode,
            use_reranker=use_reranker,
            include_parent_context=include_parent_context,
            include_diffs=include_diffs,
        ).model_dump(mode="json")

        client = self._get_client()
        resp = client.post("/api/v1/search", json=payload)
        resp.raise_for_status()
        return SearchResponse.model_validate(resp.json())

    def get_health(self) -> dict[str, Any]:
        """Consulta el estado de salud de la API."""
        client = self._get_client()
        resp = client.get("/api/v1/health")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def get_sync_status(self) -> dict[str, Any]:
        """Consulta el estado de sincronización y colección activa."""
        client = self._get_client()
        resp = client.get("/api/v1/status/sync")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def trigger_reindex(self, version_tag: str, admin_token: str | None = None) -> dict[str, Any]:
        """Dispara una re-indexación Blue/Green en Qdrant."""
        token = admin_token or self.admin_token
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        client = self._get_client()
        resp = client.post(
            "/api/v1/admin/reindex",
            json={"new_version_tag": version_tag},
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]
