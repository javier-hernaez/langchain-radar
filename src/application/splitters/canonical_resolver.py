import ast
import logging
import re
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)


class CanonicalImportResolver:
    """Resuelve la ruta canónica idiomática de importación pública para cualquier entidad Python."""

    def __init__(self) -> None:
        # Mapeo: (modulo_normalizado, symbol_name) -> canonical_import_statement
        self._symbol_reexports: dict[tuple[str, str], str] = {}
        # Mapeo: paquete_directorio -> set de símbolos públicos expuestos en __all__
        self._package_exports: dict[str, dict[str, str]] = {}

    def register_init_file(self, file_path: str, content: str) -> None:
        """Analiza un archivo __init__.py extrayendo re-exportaciones y la lista __all__."""
        norm_path = PurePosixPath(file_path.replace("\\", "/"))
        package_dir = str(norm_path.parent)

        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as e:
            logger.warning(
                "Fallo sintáctico al parsear %s para resolver importaciones: %e", file_path, e
            )
            return

        all_symbols: set[str] = set()
        for node in tree.body:
            # Buscar __all__ = ["A", "B"]
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "__all__"
                        and isinstance(node.value, (ast.List, ast.Tuple))
                    ):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                all_symbols.add(elt.value)

        # Buscar sentencias de re-exportación: from .submodule import Symbol / from pkg import Symbol
        reexports: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                for alias in node.names:
                    symbol = alias.asname or alias.name
                    if not all_symbols or symbol in all_symbols:
                        reexports[symbol] = module_name

        self._package_exports[package_dir] = reexports

    def resolve(self, file_path: str, entity_name: str) -> str:
        """Determina la importación canónica pública para un archivo y nombre de entidad dados.

        Ejemplo:
          file_path: libs/core/langchain_core/language_models/chat_models.py
          entity_name: BaseChatModel
          -> 'from langchain_core.language_models import BaseChatModel' (si está reexportado)
          o fallback: 'from langchain_core.language_models.chat_models import BaseChatModel'
        """
        # Extraer nombre base de la entidad (ej. 'BaseChatModel.bind_tools' -> 'BaseChatModel')
        root_symbol = entity_name.split(".")[0]
        norm_path = PurePosixPath(file_path.replace("\\", "/"))

        # 1. Buscar en directores ancestros si algún __init__.py exporta root_symbol
        parent = norm_path.parent
        while str(parent) not in (".", "/"):
            parent_str = str(parent)
            if parent_str in self._package_exports:
                exports = self._package_exports[parent_str]
                if root_symbol in exports:
                    # Derivar nombre de módulo canónico
                    canonical_module = self._derive_module_name(parent_str)
                    return f"from {canonical_module} import {root_symbol}"
            parent = parent.parent

        # 2. Fallback idiomático directo derivado de la ruta física
        file_module = self._derive_module_name(str(norm_path.with_suffix("")))
        return f"from {file_module} import {root_symbol}"

    def _derive_module_name(self, raw_path: str) -> str:
        """Convierte una ruta de archivo (ej. libs/core/langchain_core/tools.py) en módulo Python idiomático."""
        clean = raw_path.replace("\\", "/").strip("/")

        # Eliminar prefijos de monorepo conocidos de LangChain
        # ej. libs/core/, libs/community/, libs/langchain/, libs/partners/openai/
        patterns_to_strip = [
            r"^libs/core/",
            r"^libs/community/",
            r"^libs/langchain/",
            r"^libs/partners/[^/]+/",
            r"^libs/standard-tests/",
            r"^libs/experimental/",
            r"^src/",
        ]
        for pattern in patterns_to_strip:
            clean = re.sub(pattern, "", clean)

        # Si termina en __init__, remover
        if clean.endswith("/__init__") or clean.endswith("__init__"):
            clean = clean.removesuffix("/__init__").removesuffix("__init__")

        # Reemplazar barras por puntos
        module = clean.replace("/", ".")
        return module or "langchain"
