import hashlib
import logging
import re
import uuid
from datetime import datetime

from src.application.splitters.canonical_resolver import CanonicalImportResolver
from src.domain.models.chunk import ChunkMetadata, CodeChunk
from src.infrastructure.security.secret_scrubber import SecretScrubber

logger = logging.getLogger(__name__)

HUNK_HEADER_REGEX = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: ?(.*))?$")
BREAKING_PATTERN_REGEX = re.compile(
    r"(?i)\b(breaking[\s_-]?change|feat!|fix!|refactor!|deprecated)\b"
)


class GitDiffSplitter:
    """Segmentador semántico de diffs de Git a nivel de Hunk (@@ -x,y +a,b @@)."""

    def __init__(
        self,
        resolver: CanonicalImportResolver | None = None,
        scrubber: SecretScrubber | None = None,
    ) -> None:
        self._resolver = resolver or CanonicalImportResolver()
        self._scrubber = scrubber or SecretScrubber()

    def parse_patch_hunks(
        self,
        file_path: str,
        patch_content: str,
        commit_sha: str,
        commit_date: datetime,
        release_version: str = "dev",
        semver_major: int = 0,
        semver_minor: int = 0,
        pr_number: int | None = None,
        pr_title: str | None = None,
        author: str = "unknown",
    ) -> list[CodeChunk]:
        """Fragmenta un patch unificado en chunks atómicos correspondientes a cada Hunk."""
        if not patch_content or not patch_content.strip():
            return []

        # 1. Sanitizar secretos pre-ingesta
        clean_patch = self._scrubber.scrub(patch_content).sanitized_text
        lines = clean_patch.splitlines()

        hunks: list[tuple[int, int, str, list[str]]] = []
        current_header = ""
        current_start_line = 1
        current_end_line = 1
        current_lines: list[str] = []

        for line in lines:
            match = HUNK_HEADER_REGEX.match(line)
            if match:
                if current_lines:
                    hunks.append(
                        (
                            current_start_line,
                            current_end_line,
                            current_header,
                            current_lines,
                        )
                    )
                new_start = int(match.group(3))
                new_count = int(match.group(4) or "1")
                current_start_line = new_start
                current_end_line = new_start + max(1, new_count) - 1
                current_header = match.group(5).strip() if match.group(5) else ""
                current_lines = [line]
            else:
                if current_lines:
                    current_lines.append(line)

        if current_lines:
            hunks.append((current_start_line, current_end_line, current_header, current_lines))

        # Si no se detectó ningún encabezado @@, tratar el parche completo como un único hunk
        if not hunks:
            hunks.append((1, len(lines), "patch", lines))

        is_breaking = False
        if pr_title and BREAKING_PATTERN_REGEX.search(pr_title):
            is_breaking = True

        chunks: list[CodeChunk] = []

        for start_line, end_line, context_header, hunk_lines in hunks:
            raw_hunk_text = "\n".join(hunk_lines)

            # Si el hunk contiene palabras clave de breaking change en comentarios o código
            hunk_breaking = is_breaking or bool(BREAKING_PATTERN_REGEX.search(raw_hunk_text))

            # Resolver ruta canónica e importación si es un archivo Python
            canonical_import = ""
            module_name = file_path.replace("/", ".").removesuffix(".py")
            if file_path.endswith(".py"):
                # Intentar extraer entidad del contexto del hunk (ej: def foo(), class Bar:)
                match_entity = re.search(r"\b(def|class)\s+([A-Za-z0-9_]+)", context_header)
                if match_entity:
                    entity_name = f"{file_path}#{match_entity.group(2)}"
                    canonical_import = self._resolver.resolve(
                        file_path=file_path, entity_name=match_entity.group(2)
                    )
                else:
                    entity_name = f"{file_path}#{start_line}-{end_line}"
            else:
                entity_name = f"{file_path}#{start_line}-{end_line}"

            content_banner = (
                f"# [Git Diff Hunk - {file_path}]\n"
                f"# PR #{pr_number or 'N/A'}: {pr_title or 'Commit Diff'}\n"
                f"# Contexto: {context_header or 'Global'}\n"
                f"# Líneas: {start_line}-{end_line}\n\n"
                f"{raw_hunk_text}"
            )

            content_hash = hashlib.sha256(raw_hunk_text.encode("utf-8")).hexdigest()

            metadata = ChunkMetadata(
                chunk_id=uuid.uuid4(),
                parent_chunk_id=None,
                file_path=file_path,
                canonical_import=canonical_import,
                module_name=module_name,
                language="diff",
                entity_type="diff_hunk",
                entity_name=entity_name,
                is_parent_class=False,
                commit_sha=commit_sha,
                commit_date=commit_date,
                release_version=release_version,
                semver_major=semver_major,
                semver_minor=semver_minor,
                pr_number=pr_number,
                pr_title=pr_title,
                author=author,
                is_deprecated=False,
                breaking_change=hunk_breaking,
                content_hash=content_hash,
            )

            chunks.append(
                CodeChunk(
                    content=content_banner,
                    raw_code=raw_hunk_text,
                    start_line=start_line,
                    end_line=end_line,
                    metadata=metadata,
                )
            )

        return chunks
