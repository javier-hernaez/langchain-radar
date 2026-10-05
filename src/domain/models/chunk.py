from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Contrato estricto e inmutable de metadatos para Silver Delta Lake y Gold Qdrant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: UUID = Field(description="Identificador único del chunk")
    parent_chunk_id: UUID | None = Field(
        default=None, description="ID del chunk padre si es un método o sección secundaria"
    )
    file_path: str = Field(description="Ruta relativa del archivo en el repositorio")
    canonical_import: str = Field(description="Sintaxis canónica de importación idiomática pública")
    module_name: str = Field(description="Nombre completo del módulo Python o sección doc")
    language: Literal["python", "markdown", "mdx", "diff"] = Field(
        default="python", description="Lenguaje o tipo de contenido"
    )
    entity_type: Literal[
        "class", "method", "function", "module_var", "doc_section", "diff_hunk"
    ] = Field(description="Tipo de entidad estructural parseada")
    entity_name: str = Field(
        description="Nombre identificador calificado (ej. BaseChatModel.bind_tools)"
    )
    is_parent_class: bool = Field(
        default=False, description="True si representa el esqueleto principal de una clase"
    )
    commit_sha: str = Field(min_length=7, max_length=40, description="SHA del commit")
    commit_date: datetime = Field(description="Timestamp del commit")
    release_version: str = Field(default="dev", description="Versión de release estimada o tag")
    semver_major: int = Field(default=0, ge=0)
    semver_minor: int = Field(default=0, ge=0)
    pr_number: int | None = Field(default=None, description="Número de PR asociado si aplica")
    pr_title: str | None = Field(default=None, description="Título del PR")
    author: str = Field(description="Autor del cambio")
    is_deprecated: bool = Field(default=False, description="Indica si la entidad está obsoleta")
    breaking_change: bool = Field(
        default=False, description="Indica si introduce cambios incompatibles"
    )
    content_hash: str = Field(
        min_length=64, max_length=64, description="Hash SHA-256 del contenido para deduplicación"
    )


class CodeChunk(BaseModel):
    """Entidad de chunk de código con su contenido contextual y metadatos."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str = Field(
        description="Código fuente del fragmento con cabeceras contextuales inyectadas"
    )
    raw_code: str = Field(description="Código exacto sin inyecciones para visualización pura")
    start_line: int = Field(ge=1, description="Línea inicial en el archivo de origen")
    end_line: int = Field(ge=1, description="Línea final en el archivo de origen")
    metadata: ChunkMetadata = Field(description="Metadatos tipados del chunk")


class DocChunk(BaseModel):
    """Entidad de chunk de documentación Markdown o MDX."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str = Field(description="Contenido del fragmento documental")
    header_path: tuple[str, ...] = Field(
        default=(), description="Jerarquía de cabeceras precedentes, ej. ('Overview', 'Quickstart')"
    )
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    metadata: ChunkMetadata = Field(description="Metadatos tipados del chunk")
