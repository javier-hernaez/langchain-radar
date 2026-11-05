from typing import Protocol

from src.domain.models.events import CommitFileChange


class SourceCodeHost(Protocol):
    """Puerto de dominio para obtener ficheros y contenidos de un repositorio remoto (GitHub)."""

    async def fetch_pr_files(self, owner: str, repo: str, pr_number: int) -> list[CommitFileChange]:
        """Lista los ficheros modificados de un PR, incluyendo su patch unificado si existe."""
        ...

    async def fetch_file_content(self, owner: str, repo: str, path: str, ref: str) -> str | None:
        """Devuelve el contenido completo de un fichero en una revisión, o None si no existe."""
        ...
