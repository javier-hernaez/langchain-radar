import asyncio
import logging
import urllib.parse
from typing import Any

import httpx

from config.settings import Settings, get_settings
from src.domain.exceptions import RateLimitExceededError, StorageError
from src.domain.models.events import CommitFileChange
from src.domain.ports.source_host import SourceCodeHost

logger = logging.getLogger(__name__)

# Estados de la API REST de GitHub -> estados del dominio
_STATUS_MAP: dict[str, str] = {
    "added": "ADDED",
    "removed": "REMOVED",
    "modified": "MODIFIED",
    "renamed": "RENAMED",
    "copied": "ADDED",
    "changed": "MODIFIED",
    "unchanged": "MODIFIED",
}

# GitHub limita el listado de ficheros de un PR a 3000 entradas (30 páginas de 100)
_MAX_FILE_PAGES = 30
_MAX_BACKOFF_SECONDS = 60.0


class GitHubRestClient(SourceCodeHost):
    """Cliente REST de GitHub para ficheros de PRs y contenidos completos por revisión.

    La API GraphQL no expone el patch de los ficheros de un PR y el webhook
    `pull_request` no incluye la lista de ficheros: ambos se obtienen aquí.
    """

    def __init__(
        self,
        token: str | None = None,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.github.com",
        max_retries: int = 3,
    ) -> None:
        self._settings = settings or get_settings()
        self._token = token if token is not None else self._settings.github_token
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def aclose(self) -> None:
        """Cierra el cliente HTTP si fue creado internamente."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _headers(self, accept: str) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "User-Agent": "LangChain-Radar-Ingestion/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    @staticmethod
    def _is_rate_limited(response: httpx.Response) -> bool:
        if response.status_code == 429:
            return True
        return response.status_code == 403 and (
            response.headers.get("x-ratelimit-remaining") == "0"
            or "rate limit" in response.text.lower()
        )

    async def _get(
        self, path: str, params: dict[str, Any] | None = None, accept: str = "application/json"
    ) -> httpx.Response | None:
        """GET con reintentos ante rate limit y errores de red. Devuelve None ante 404."""
        client = self._get_client()
        url = f"{self._base_url}{path}"
        backoff = 1.0

        for attempt in range(self._max_retries + 1):
            try:
                response = await client.get(url, params=params, headers=self._headers(accept))
            except httpx.RequestError as e:
                if attempt >= self._max_retries:
                    raise StorageError(f"Error de conexión con GitHub REST API: {e}") from e
                await asyncio.sleep(backoff)
                backoff *= 2.0
                continue

            from src.infrastructure.telemetry.metrics import get_metrics

            get_metrics().inc_counter(
                "github_api_requests_total",
                labels={"status": str(response.status_code), "endpoint": "rest"},
            )

            if response.status_code == 404:
                return None

            if self._is_rate_limited(response):
                if attempt >= self._max_retries:
                    raise RateLimitExceededError(
                        f"Rate limit de GitHub REST API agotado (HTTP {response.status_code})"
                    )
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                await asyncio.sleep(min(delay, _MAX_BACKOFF_SECONDS))
                backoff *= 2.0
                continue

            if response.status_code != 200:
                raise StorageError(
                    f"Error HTTP en GitHub REST API ({path}): "
                    f"{response.status_code} - {response.text[:500]}"
                )
            return response

        raise StorageError(f"Reintentos agotados contra GitHub REST API ({path})")

    async def fetch_pr_files(self, owner: str, repo: str, pr_number: int) -> list[CommitFileChange]:
        """Lista paginada de ficheros modificados de un PR, con su patch unificado."""
        files: list[CommitFileChange] = []
        for page in range(1, _MAX_FILE_PAGES + 1):
            response = await self._get(
                f"/repos/{owner}/{repo}/pulls/{pr_number}/files",
                params={"per_page": 100, "page": page},
            )
            if response is None:
                break
            items: list[dict[str, Any]] = response.json()
            for item in items:
                files.append(
                    CommitFileChange(
                        file_path=item.get("filename", ""),
                        status=_STATUS_MAP.get(item.get("status", ""), "MODIFIED"),  # type: ignore[arg-type]
                        additions=item.get("additions", 0),
                        deletions=item.get("deletions", 0),
                        changes=item.get("changes", 0),
                        patch=item.get("patch"),
                        previous_filename=item.get("previous_filename"),
                    )
                )
            if len(items) < 100:
                break
        return files

    async def fetch_file_content(self, owner: str, repo: str, path: str, ref: str) -> str | None:
        """Contenido completo de un fichero en la revisión `ref` (None si no existe)."""
        quoted_path = urllib.parse.quote(path, safe="/")
        response = await self._get(
            f"/repos/{owner}/{repo}/contents/{quoted_path}",
            params={"ref": ref},
            accept="application/vnd.github.raw+json",
        )
        if response is None:
            return None
        return response.content.decode("utf-8", errors="replace")
