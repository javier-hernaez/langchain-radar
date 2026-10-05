from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CommitFileChange(BaseModel):
    """Representa el cambio de un archivo individual dentro de un PR o Commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str = Field(description="Ruta relativa del archivo dentro del repositorio")
    status: Literal["ADDED", "MODIFIED", "REMOVED", "RENAMED"] = Field(
        description="Estado del cambio del archivo"
    )
    additions: int = Field(default=0, ge=0)
    deletions: int = Field(default=0, ge=0)
    changes: int = Field(default=0, ge=0)
    patch: str | None = Field(default=None, description="Diff del archivo si está disponible")
    previous_filename: str | None = Field(
        default=None, description="Nombre anterior si fue renombrado"
    )


class PRMergedPayload(BaseModel):
    """Entidad inmutable que representa un Pull Request fusionado con éxito."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pr_number: int = Field(gt=0, description="Número del Pull Request")
    title: str = Field(description="Título descriptivo del Pull Request")
    body: str | None = Field(default="", description="Cuerpo explicativo del PR")
    author: str = Field(description="Usuario de GitHub autor del cambio")
    base_branch: str = Field(default="main", description="Rama destino")
    head_branch: str = Field(description="Rama origen")
    merged_at: datetime = Field(description="Timestamp ISO 8601 de la fusión")
    merge_commit_sha: str = Field(
        min_length=7, max_length=40, description="SHA del commit de merge"
    )
    is_breaking_change: bool = Field(
        default=False, description="Indica si se detectó un cambio incompatible (Breaking Change)"
    )
    labels: tuple[str, ...] = Field(default=(), description="Etiquetas asignadas al PR")
    files_changed: tuple[CommitFileChange, ...] = Field(
        default=(), description="Lista inmutable de archivos modificados"
    )


class GitHubWebhookEvent(BaseModel):
    """Evento crudo recibido a través de la API de Webhooks de GitHub."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(description="Identificador único del delivery (X-GitHub-Delivery)")
    event_type: str = Field(description="Tipo de evento (X-GitHub-Event), ej. pull_request")
    action: str = Field(description="Acción del evento, ej. closed")
    repository_full_name: str = Field(
        description="Nombre completo del repo, ej. langchain-ai/langchain"
    )
    received_at: datetime = Field(description="Timestamp de recepción en la plataforma")
    raw_payload: str = Field(description="Payload JSON crudo para auditoría e ingesta Bronze")
    pr_data: PRMergedPayload | None = Field(
        default=None, description="Payload estructurado de PR si aplica y fue fusionado"
    )
