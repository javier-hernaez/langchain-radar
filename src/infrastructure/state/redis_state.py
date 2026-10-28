import json
import logging
from typing import Any

from src.domain.exceptions import StorageError
from src.domain.models.state import SyncCheckpoint
from src.domain.ports.state_store import StateStore

logger = logging.getLogger(__name__)


class RedisStateStore(StateStore):
    """Adaptador de infraestructura para persistencia del High-Water Mark con Redis."""

    def __init__(self, client: Any = None, prefix: str = "radar:state") -> None:
        self._client = client
        self._prefix = prefix
        # Fallback en memoria si no hay cliente Redis configurado
        self._memory_checkpoints: dict[str, SyncCheckpoint] = {}
        self._memory_prs: dict[int, str] = {}

    def _checkpoint_key(self, checkpoint_id: str) -> str:
        return f"{self._prefix}:checkpoint:{checkpoint_id}"

    @property
    def _prs_set_key(self) -> str:
        return f"{self._prefix}:processed_prs"

    async def get_checkpoint(self, checkpoint_id: str) -> SyncCheckpoint | None:
        """Recupera el último checkpoint registrado."""
        if self._client is None:
            return self._memory_checkpoints.get(checkpoint_id)

        try:
            key = self._checkpoint_key(checkpoint_id)
            data = await self._client.get(key)
            if not data:
                return None
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            raw_dict = json.loads(data)
            return SyncCheckpoint.model_validate(raw_dict)
        except Exception as e:
            raise StorageError(
                f"Error al leer checkpoint '{checkpoint_id}' de Redis: {e}",
                {"checkpoint_id": checkpoint_id},
            ) from e

    async def save_checkpoint(self, checkpoint: SyncCheckpoint) -> None:
        """Persiste atómicamente un nuevo checkpoint de sincronización."""
        if self._client is None:
            self._memory_checkpoints[checkpoint.checkpoint_id] = checkpoint
            return

        try:
            key = self._checkpoint_key(checkpoint.checkpoint_id)
            payload = checkpoint.model_dump_json()
            await self._client.set(key, payload)
        except Exception as e:
            raise StorageError(
                f"Error al guardar checkpoint '{checkpoint.checkpoint_id}' en Redis: {e}",
                {"checkpoint_id": checkpoint.checkpoint_id},
            ) from e

    async def record_processed_pr(self, pr_number: int, commit_sha: str) -> None:
        """Registra un PR fusionado como procesado de forma duradera."""
        if self._client is None:
            self._memory_prs[pr_number] = commit_sha
            return

        try:
            pipe = self._client.pipeline()
            pipe.sadd(self._prs_set_key, str(pr_number))
            pipe.hset(f"{self._prefix}:pr_commits", str(pr_number), commit_sha)
            await pipe.execute()
        except Exception as e:
            raise StorageError(
                f"Error al registrar PR #{pr_number} en Redis: {e}",
                {"pr_number": pr_number, "commit_sha": commit_sha},
            ) from e

    async def is_pr_processed(self, pr_number: int) -> bool:
        """Verifica si un PR ya fue completamente procesado."""
        if self._client is None:
            return pr_number in self._memory_prs

        try:
            result = await self._client.sismember(self._prs_set_key, str(pr_number))
            return bool(result)
        except Exception as e:
            raise StorageError(
                f"Error al verificar estado de PR #{pr_number} en Redis: {e}",
                {"pr_number": pr_number},
            ) from e

    async def get_processed_pr_count(self) -> int:
        """Retorna el número total de PRs procesados registrados."""
        if self._client is None:
            return len(self._memory_prs)

        try:
            count = await self._client.scard(self._prs_set_key)
            return int(count)
        except Exception as e:
            raise StorageError(
                f"Error al obtener conteo de PRs procesados de Redis: {e}",
                {},
            ) from e
