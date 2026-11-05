import logging
from datetime import datetime

from src.application.splitters.ast_python import ASTPythonSplitter
from src.application.splitters.canonical_resolver import CanonicalImportResolver
from src.application.splitters.mdx_parser import MDXParser
from src.domain.exceptions import ParsingError
from src.domain.models.chunk import CodeChunk, DocChunk

logger = logging.getLogger(__name__)


class SourceFileChunker:
    """Despacha el particionado estructural (AST Python / Markdown) del contenido completo de un fichero."""

    SUPPORTED_SUFFIXES: tuple[str, ...] = (".py", ".md", ".mdx")

    def __init__(
        self,
        ast_splitter: ASTPythonSplitter,
        mdx_parser: MDXParser,
        resolver: CanonicalImportResolver,
    ) -> None:
        self._ast_splitter = ast_splitter
        self._mdx_parser = mdx_parser
        self._resolver = resolver

    @classmethod
    def supports(cls, file_path: str) -> bool:
        return file_path.lower().endswith(cls.SUPPORTED_SUFFIXES)

    def chunk(
        self,
        file_path: str,
        content: str,
        commit_sha: str,
        commit_date: datetime,
        pr_number: int | None,
        pr_title: str | None,
        author: str,
        breaking_change: bool = False,
        release_version: str = "main",
    ) -> list[CodeChunk | DocChunk]:
        """Genera chunks del fichero; un error de sintaxis descarta solo ese fichero."""
        try:
            if file_path.endswith(".py"):
                if file_path.endswith("__init__.py"):
                    self._resolver.register_init_file(file_path, content)
                return list(
                    self._ast_splitter.split_file(
                        file_path=file_path,
                        source_code=content,
                        commit_sha=commit_sha,
                        commit_date=commit_date,
                        release_version=release_version,
                        pr_number=pr_number,
                        pr_title=pr_title,
                        author=author,
                        breaking_change=breaking_change,
                    )
                )
            if file_path.lower().endswith((".md", ".mdx")):
                return list(
                    self._mdx_parser.parse_document(
                        file_path=file_path,
                        text_content=content,
                        commit_sha=commit_sha,
                        commit_date=commit_date,
                        release_version=release_version,
                        pr_number=pr_number,
                        pr_title=pr_title,
                        author=author,
                    )
                )
        except ParsingError as e:
            logger.warning("Fichero omitido por error de parseo (%s): %s", file_path, e.message)
        return []
