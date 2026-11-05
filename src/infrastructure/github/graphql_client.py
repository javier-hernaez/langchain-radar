import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from config.settings import Settings, get_settings
from src.domain.exceptions import RateLimitExceededError, StorageError
from src.domain.models.events import CommitFileChange, PRMergedPayload

logger = logging.getLogger(__name__)

MERGED_PRS_QUERY = """
query GetMergedPRs($owner: String!, $name: String!, $after: String, $pageSize: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      states: MERGED,
      first: $pageSize,
      after: $after,
      orderBy: {field: CREATED_AT, direction: DESC}
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        body
        author {
          login
        }
        mergedAt
        baseRefName
        headRefName
        mergeCommit {
          oid
        }
        labels(first: 20) {
          nodes {
            name
          }
        }
        files(first: 100) {
          totalCount
          pageInfo {
            hasNextPage
          }
          nodes {
            path
            changeType
            additions
            deletions
          }
        }
      }
    }
  }
  rateLimit {
    cost
    remaining
    resetAt
  }
}
"""


class GitHubGraphQLClient:
    """Cliente para la API GraphQL de GitHub con paginación y control de Rate Limits."""

    def __init__(
        self,
        token: str | None = None,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        endpoint: str = "https://api.github.com/graphql",
    ) -> None:
        self._settings = settings or get_settings()
        self._token = token if token is not None else self._settings.github_token
        self._endpoint = endpoint
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        # Un único cliente reutilizado: crear uno por petición filtraba conexiones
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def aclose(self) -> None:
        """Cierra el cliente HTTP si fue creado internamente."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def fetch_merged_prs(
        self,
        owner: str | None = None,
        repo: str | None = None,
        page_size: int = 20,
        cursor: str | None = None,
        max_retries: int = 3,
    ) -> tuple[list[PRMergedPayload], str | None, bool]:
        """Obtiene una página de Pull Requests fusionados (MERGED).

        Retorna:
            (lista_de_prs, next_cursor, has_next_page)
        """
        repo_owner = owner or self._settings.github_repo_owner
        repo_name = repo or self._settings.github_repo_name

        variables = {
            "owner": repo_owner,
            "name": repo_name,
            "after": cursor,
            "pageSize": min(page_size, 50),
        }

        async_client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self._token}",
            "User-Agent": "LangChain-Radar-Ingestion/1.0",
        }

        retry_count = 0
        backoff = 1.0

        while True:
            try:
                response = await async_client.post(
                    self._endpoint,
                    json={"query": MERGED_PRS_QUERY, "variables": variables},
                    headers=headers,
                )
                from src.infrastructure.telemetry.metrics import get_metrics

                get_metrics().inc_counter(
                    "github_api_requests_total",
                    labels={"status": str(response.status_code), "endpoint": "graphql"},
                )

                rate_limited = response.status_code == 429 or (
                    response.status_code == 403
                    and response.headers.get("x-ratelimit-remaining") == "0"
                )
                if rate_limited:
                    if retry_count >= max_retries:
                        raise RateLimitExceededError("Rate limit de GitHub API agotado (HTTP 429)")
                    retry_count += 1
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue

                if response.status_code != 200:
                    raise StorageError(
                        f"Error HTTP en GitHub GraphQL API: {response.status_code} - {response.text}"
                    )

                data: dict[str, Any] = response.json()
                if "errors" in data:
                    errors = data["errors"]
                    raise StorageError(f"Errores en GraphQL API: {errors}")

                return self._parse_response(data)

            except httpx.RequestError as e:
                if retry_count >= max_retries:
                    raise StorageError(f"Error de conexión con GitHub GraphQL API: {e}") from e
                retry_count += 1
                await asyncio.sleep(backoff)
                backoff *= 2.0

    def _parse_response(
        self, data: dict[str, Any]
    ) -> tuple[list[PRMergedPayload], str | None, bool]:
        """Parsea la respuesta de GraphQL a entidades inmutables PRMergedPayload."""
        repo_data = data.get("data", {}).get("repository", {})
        pr_container = repo_data.get("pullRequests", {})
        page_info = pr_container.get("pageInfo", {})
        nodes = pr_container.get("nodes", [])

        has_next_page: bool = page_info.get("hasNextPage", False)
        end_cursor: str | None = page_info.get("endCursor")

        results: list[PRMergedPayload] = []
        for node in nodes:
            if not node:
                continue

            # Parsear archivos modificados
            file_changes: list[CommitFileChange] = []
            files_node = node.get("files", {})
            file_items = files_node.get("nodes", [])

            for f in file_items:
                raw_change = f.get("changeType", "MODIFIED").upper()
                # Enum PatchStatus de GitHub GraphQL
                status_map = {
                    "ADDED": "ADDED",
                    "COPIED": "ADDED",
                    "MODIFIED": "MODIFIED",
                    "CHANGED": "MODIFIED",
                    "DELETED": "REMOVED",
                    "REMOVED": "REMOVED",
                    "RENAMED": "RENAMED",
                }
                status_val = status_map.get(raw_change, "MODIFIED")

                file_change = CommitFileChange(
                    file_path=f.get("path", ""),
                    status=status_val,  # type: ignore[arg-type]
                    additions=f.get("additions", 0),
                    deletions=f.get("deletions", 0),
                    changes=f.get("additions", 0) + f.get("deletions", 0),
                    # GraphQL no expone el patch; el contenido se obtiene vía SourceCodeHost
                    patch=None,
                )
                file_changes.append(file_change)

            # Etiquetas
            label_nodes = node.get("labels", {}).get("nodes", [])
            labels = tuple(lbl.get("name", "") for lbl in label_nodes if lbl.get("name"))

            # Detección de Breaking Change
            title = node.get("title", "")
            body = node.get("body", "") or ""
            is_breaking = (
                "breaking change" in title.lower()
                or "breaking change" in body.lower()
                or any("breaking" in lbl.lower() for lbl in labels)
            )

            # Merge commit SHA
            merge_commit = node.get("mergeCommit") or {}
            merge_sha = merge_commit.get("oid") or ("0" * 40)

            # Timestamp
            merged_at_str = node.get("mergedAt")
            if merged_at_str:
                merged_at = datetime.fromisoformat(merged_at_str.replace("Z", "+00:00"))
            else:
                merged_at = datetime.now(UTC)

            author_obj = node.get("author") or {}
            author = author_obj.get("login") or "ghost"

            pr_payload = PRMergedPayload(
                pr_number=node.get("number", 0),
                title=title,
                body=body,
                author=author,
                base_branch=node.get("baseRefName", "main"),
                head_branch=node.get("headRefName", "branch"),
                merged_at=merged_at,
                merge_commit_sha=merge_sha,
                is_breaking_change=is_breaking,
                labels=labels,
                files_changed=tuple(file_changes),
            )
            results.append(pr_payload)

        return results, end_cursor, has_next_page
