import ast
import hashlib
import logging
import uuid
from datetime import UTC, datetime

from src.application.splitters.canonical_resolver import CanonicalImportResolver
from src.domain.exceptions import ParsingError
from src.domain.models.chunk import ChunkMetadata, CodeChunk
from src.infrastructure.security.secret_scrubber import SecretScrubber

logger = logging.getLogger(__name__)


class ASTPythonSplitter:
    """Particionador semántico jerárquico Padre-Hijo de código Python mediante AST."""

    def __init__(
        self,
        resolver: CanonicalImportResolver | None = None,
        scrubber: SecretScrubber | None = None,
    ) -> None:
        self._resolver = resolver or CanonicalImportResolver()
        self._scrubber = scrubber or SecretScrubber()

    def split_file(
        self,
        file_path: str,
        source_code: str,
        commit_sha: str,
        commit_date: datetime | None = None,
        release_version: str = "dev",
        semver_major: int = 0,
        semver_minor: int = 0,
        pr_number: int | None = None,
        pr_title: str | None = None,
        author: str = "unknown",
        breaking_change: bool = False,
    ) -> list[CodeChunk]:
        """Parsea el código fuente y genera chunks jerárquicos (Clases padre, métodos hijos y funciones)."""
        if not source_code.strip():
            return []

        # 1. Sanitizar secretos pre-particionado
        scrub_res = self._scrubber.scrub(source_code)
        clean_code = scrub_res.sanitized_text

        # 2. Parsear el árbol AST
        try:
            tree = ast.parse(clean_code, filename=file_path)
        except SyntaxError as e:
            raise ParsingError(
                f"Error sintáctico al analizar {file_path}: {e}",
                {"file_path": file_path, "lineno": e.lineno or 0},
            ) from e

        code_lines = clean_code.splitlines()
        chunks: list[CodeChunk] = []
        commit_ts = commit_date or datetime.now(UTC)
        module_name = self._resolver._derive_module_name(file_path)

        for node in tree.body:
            # Procesar Clases y sus métodos
            if isinstance(node, ast.ClassDef):
                class_chunks = self._process_class_node(
                    node=node,
                    code_lines=code_lines,
                    file_path=file_path,
                    module_name=module_name,
                    commit_sha=commit_sha,
                    commit_ts=commit_ts,
                    release_version=release_version,
                    semver_major=semver_major,
                    semver_minor=semver_minor,
                    pr_number=pr_number,
                    pr_title=pr_title,
                    author=author,
                    breaking_change=breaking_change,
                )
                chunks.extend(class_chunks)

            # Procesar Funciones sueltas a nivel de módulo
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_chunk = self._process_standalone_function(
                    node=node,
                    code_lines=code_lines,
                    file_path=file_path,
                    module_name=module_name,
                    commit_sha=commit_sha,
                    commit_ts=commit_ts,
                    release_version=release_version,
                    semver_major=semver_major,
                    semver_minor=semver_minor,
                    pr_number=pr_number,
                    pr_title=pr_title,
                    author=author,
                    breaking_change=breaking_change,
                )
                chunks.append(func_chunk)

        return chunks

    def _process_class_node(
        self,
        node: ast.ClassDef,
        code_lines: list[str],
        file_path: str,
        module_name: str,
        commit_sha: str,
        commit_ts: datetime,
        release_version: str,
        semver_major: int,
        semver_minor: int,
        pr_number: int | None,
        pr_title: str | None,
        author: str,
        breaking_change: bool,
    ) -> list[CodeChunk]:
        """Genera el chunk Padre (esqueleto) y chunks Hijos (métodos) para una clase."""
        parent_chunk_id = uuid.uuid4()
        canonical_import = self._resolver.resolve(file_path, node.name)

        # 1. Extraer docstring de clase y métodos públicos para el esqueleto
        docstring = ast.get_docstring(node) or ""
        method_signatures: list[str] = []
        child_method_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_method_nodes.append(item)
                # Extraer firma de la primera línea de la función
                sig_line = code_lines[item.lineno - 1].strip()
                method_signatures.append(f"    {sig_line} ...")

        # Construir cabecera de la clase
        class_def_line = code_lines[node.lineno - 1].strip()
        doc_part = f'\n    """{docstring}"""\n' if docstring else "\n"
        skeleton_methods = "\n".join(method_signatures) if method_signatures else "    pass"
        skeleton_content = (
            f"# [Contexto Canónico de Clase]\n"
            f"# Modulo: {module_name}\n"
            f"# Import: {canonical_import}\n\n"
            f"{class_def_line}"
            f"{doc_part}"
            f"{skeleton_methods}"
        )

        parent_meta = ChunkMetadata(
            chunk_id=parent_chunk_id,
            parent_chunk_id=None,
            file_path=file_path,
            canonical_import=canonical_import,
            module_name=module_name,
            language="python",
            entity_type="class",
            entity_name=node.name,
            is_parent_class=True,
            commit_sha=commit_sha,
            commit_date=commit_ts,
            release_version=release_version,
            semver_major=semver_major,
            semver_minor=semver_minor,
            pr_number=pr_number,
            pr_title=pr_title,
            author=author,
            is_deprecated=self._is_deprecated(node),
            breaking_change=breaking_change,
            content_hash=hashlib.sha256(skeleton_content.encode("utf-8")).hexdigest(),
        )

        parent_chunk = CodeChunk(
            content=skeleton_content,
            raw_code=self._slice_lines(code_lines, node.lineno, node.end_lineno or node.lineno),
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            metadata=parent_meta,
        )

        chunks: list[CodeChunk] = [parent_chunk]

        # 2. Generar Chunks Hijos para cada método
        for method_node in child_method_nodes:
            method_name = f"{node.name}.{method_node.name}"
            raw_method_code = self._slice_lines(
                code_lines, method_node.lineno, method_node.end_lineno or method_node.lineno
            )

            # Inyectar contexto canónico en la cabecera del chunk del método
            injected_header = (
                f"# [Contexto Canónico Inyectado]\n"
                f"# Modulo: {module_name}\n"
                f"# Import: {canonical_import}\n"
                f"# Clase: {class_def_line}\n\n"
            )
            child_content = injected_header + raw_method_code

            child_meta = ChunkMetadata(
                chunk_id=uuid.uuid4(),
                parent_chunk_id=parent_chunk_id,
                file_path=file_path,
                canonical_import=canonical_import,
                module_name=module_name,
                language="python",
                entity_type="method",
                entity_name=method_name,
                is_parent_class=False,
                commit_sha=commit_sha,
                commit_date=commit_ts,
                release_version=release_version,
                semver_major=semver_major,
                semver_minor=semver_minor,
                pr_number=pr_number,
                pr_title=pr_title,
                author=author,
                is_deprecated=self._is_deprecated(method_node),
                breaking_change=breaking_change,
                content_hash=hashlib.sha256(child_content.encode("utf-8")).hexdigest(),
            )

            child_chunk = CodeChunk(
                content=child_content,
                raw_code=raw_method_code,
                start_line=method_node.lineno,
                end_line=method_node.end_lineno or method_node.lineno,
                metadata=child_meta,
            )
            chunks.append(child_chunk)

        return chunks

    def _process_standalone_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        code_lines: list[str],
        file_path: str,
        module_name: str,
        commit_sha: str,
        commit_ts: datetime,
        release_version: str,
        semver_major: int,
        semver_minor: int,
        pr_number: int | None,
        pr_title: str | None,
        author: str,
        breaking_change: bool,
    ) -> CodeChunk:
        """Genera un chunk para una función de nivel de módulo."""
        canonical_import = self._resolver.resolve(file_path, node.name)
        raw_code = self._slice_lines(code_lines, node.lineno, node.end_lineno or node.lineno)
        injected_header = (
            f"# [Contexto Canónico Inyectado]\n"
            f"# Modulo: {module_name}\n"
            f"# Import: {canonical_import}\n\n"
        )
        content = injected_header + raw_code

        meta = ChunkMetadata(
            chunk_id=uuid.uuid4(),
            parent_chunk_id=None,
            file_path=file_path,
            canonical_import=canonical_import,
            module_name=module_name,
            language="python",
            entity_type="function",
            entity_name=node.name,
            is_parent_class=False,
            commit_sha=commit_sha,
            commit_date=commit_ts,
            release_version=release_version,
            semver_major=semver_major,
            semver_minor=semver_minor,
            pr_number=pr_number,
            pr_title=pr_title,
            author=author,
            is_deprecated=self._is_deprecated(node),
            breaking_change=breaking_change,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

        return CodeChunk(
            content=content,
            raw_code=raw_code,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            metadata=meta,
        )

    def _slice_lines(self, lines: list[str], start_line: int, end_line: int) -> str:
        """Extrae el bloque exacto de líneas basado en índices 1-based."""
        sliced = lines[max(0, start_line - 1) : end_line]
        return "\n".join(sliced)

    def _is_deprecated(self, node: ast.AST) -> bool:
        """Detecta si la entidad tiene decoradores o advertencias de obsolescencia."""
        if hasattr(node, "decorator_list"):
            for dec in getattr(node, "decorator_list", []):
                dec_name = ""
                if isinstance(dec, ast.Name):
                    dec_name = dec.id
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    dec_name = dec.func.id
                elif isinstance(dec, ast.Attribute):
                    dec_name = dec.attr
                if "deprecated" in dec_name.lower():
                    return True
        return False
