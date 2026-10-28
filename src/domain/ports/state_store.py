from typing import Protocol

from src.domain.models.state import SyncCheckpoint


class StateStore(Protocol):
    """Puerto de dominio para la persistencia del estado High-Water Mark e idempotencia."""

    async def get_checkpoint(self, checkpoint_id: str) -> SyncCheckpoint | None:
        """Recupera el último checkpoint registrado para un stream o job."""
        ...

    async def save_checkpoint(self, checkpoint: SyncCheckpoint) -> None:
        """Persiste atómicamente un nuevo checkpoint de sincronización."""
        ...

    async def record_processed_pr(self, pr_number: int, commit_sha: str) -> None:
        """Registra un PR fusionado como procesado de forma duradera."""
        ...

    async def is_pr_processed(self, pr_number: int) -> bool:
        """Verifica si un PR ya fue completamente procesado."""
        ...

    async def get_processed_pr_count(self) -> int:
        """Retorna el número total de PRs procesados registrados."""
        ...
