from typing import Protocol

from src.domain.models.chunk import CodeChunk, DocChunk


class BronzeStorage(Protocol):
    """Puerto de dominio para la capa Bronze: persistencia cruda e inmutable de eventos."""

    async def save_raw_event(
        self, event_id: str, year: int, month: int, day: int, raw_payload: str
    ) -> str:
        """Guarda un payload crudo JSON en Bronze y devuelve la URI S3."""
        ...

    async def get_raw_event(self, s3_uri: str) -> str:
        """Recupera el payload JSON crudo a partir de su URI S3."""
        ...


class SilverStorage(Protocol):
    """Puerto de dominio para la capa Silver: Delta Lake transaccional con esquema validado."""

    async def append_chunks(self, chunks: list[CodeChunk | DocChunk]) -> str:
        """Escribe un lote de chunks normalizados en la tabla Delta y devuelve la ruta/versión."""
        ...

    async def get_chunks_by_commit(self, commit_sha: str) -> list[CodeChunk | DocChunk]:
        """Recupera los chunks asociados a un commit específico desde Silver."""
        ...
