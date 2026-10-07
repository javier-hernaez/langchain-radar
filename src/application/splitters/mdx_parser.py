import hashlib
import re
import uuid
from datetime import UTC, datetime
from typing import Literal

from src.domain.models.chunk import ChunkMetadata, DocChunk
from src.infrastructure.security.secret_scrubber import SecretScrubber


class MDXParser:
    """Parser jerárquico de Markdown y MDX con preservación atómica de bloques de código cercados."""

    HEADER_REGEX = re.compile(r"^(#{1,6})\s+(.+)$")
    CODE_FENCE_REGEX = re.compile(r"^```")

    def __init__(self, scrubber: SecretScrubber | None = None) -> None:
        self._scrubber = scrubber or SecretScrubber()

    def parse_document(
        self,
        file_path: str,
        text_content: str,
        commit_sha: str,
        commit_date: datetime | None = None,
        release_version: str = "dev",
        semver_major: int = 0,
        semver_minor: int = 0,
        pr_number: int | None = None,
        pr_title: str | None = None,
        author: str = "docs-team",
    ) -> list[DocChunk]:
        """Segmenta un documento .md o .mdx por encabezados respetando la atomicidad de los bloques de código."""
        if not text_content.strip():
            return []

        # 1. Sanitizar secretos pre-particionado
        scrub_res = self._scrubber.scrub(text_content)
        clean_text = scrub_res.sanitized_text

        lines = clean_text.splitlines()
        commit_ts = commit_date or datetime.now(UTC)
        lang: Literal["markdown", "mdx"] = (
            "mdx" if file_path.lower().endswith(".mdx") else "markdown"
        )

        sections: list[tuple[int, int, tuple[str, ...], list[str]]] = []
        current_header_stack: list[tuple[int, str]] = []  # (level, title)
        current_lines: list[str] = []
        section_start_line = 1
        in_code_block = False

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Rastrear bloques de código cercados para nunca cortarlos
            if self.CODE_FENCE_REGEX.match(stripped):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            # Si estamos dentro de un bloque de código, acumular sin evaluar cabeceras
            if in_code_block:
                current_lines.append(line)
                continue

            # Detectar encabezados Markdown
            header_match = self.HEADER_REGEX.match(stripped)
            if header_match:
                # Si teníamos contenido acumulado previo, cerrarlo como sección
                if current_lines and any(ln.strip() for ln in current_lines):
                    header_path = tuple(t for _, t in current_header_stack)
                    sections.append((section_start_line, line_num - 1, header_path, current_lines))
                    current_lines = []

                header_hashes, header_title = header_match.groups()
                level = len(header_hashes)
                title = header_title.strip()

                # Actualizar pila jerárquica de cabeceras
                while current_header_stack and current_header_stack[-1][0] >= level:
                    current_header_stack.pop()
                current_header_stack.append((level, title))

                section_start_line = line_num
                current_lines.append(line)
            else:
                current_lines.append(line)

        # Cerrar última sección si tiene contenido
        if current_lines and any(ln.strip() for ln in current_lines):
            header_path = tuple(t for _, t in current_header_stack)
            sections.append((section_start_line, len(lines), header_path, current_lines))

        chunks: list[DocChunk] = []
        for start_ln, end_ln, h_path, sec_lines in sections:
            content_str = "\n".join(sec_lines).strip()
            if not content_str:
                continue

            entity_name = h_path[-1] if h_path else file_path
            module_name = (
                file_path.replace("\\", "/")
                .removesuffix(".md")
                .removesuffix(".mdx")
                .replace("/", ".")
            )

            meta = ChunkMetadata(
                chunk_id=uuid.uuid4(),
                parent_chunk_id=None,
                file_path=file_path,
                canonical_import=file_path,
                module_name=module_name,
                language=lang,
                entity_type="doc_section",
                entity_name=entity_name,
                is_parent_class=False,
                commit_sha=commit_sha,
                commit_date=commit_ts,
                release_version=release_version,
                semver_major=semver_major,
                semver_minor=semver_minor,
                pr_number=pr_number,
                pr_title=pr_title,
                author=author,
                is_deprecated=False,
                breaking_change=False,
                content_hash=hashlib.sha256(content_str.encode("utf-8")).hexdigest(),
            )

            chunk = DocChunk(
                content=content_str,
                header_path=h_path,
                start_line=start_ln,
                end_line=end_ln,
                metadata=meta,
            )
            chunks.append(chunk)

        return chunks
