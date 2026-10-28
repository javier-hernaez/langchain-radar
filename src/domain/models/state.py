from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SyncCheckpoint(BaseModel):
    """Modelo inmutable de checkpoint para la persistencia del High-Water Mark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint_id: str = Field(
        description="Identificador del checkpoint (ej. 'backfill_cursor', 'stream_consumer')"
    )
    last_commit_sha: str = Field(
        min_length=7,
        max_length=40,
        description="SHA del último commit procesado y persistido",
    )
    cursor: str | None = Field(
        default=None,
        description="Cursor de paginación de GraphQL o ID de stream de Redis",
    )
    last_synced_at: datetime = Field(description="Timestamp UTC de la última actualización exitosa")
    total_processed_items: int = Field(
        default=0,
        ge=0,
        description="Conteo acumulado de elementos procesados hasta este checkpoint",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadatos complementarios del estado de sincronización",
    )
