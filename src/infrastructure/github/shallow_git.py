import asyncio
import logging
import shutil
import tempfile
from pathlib import PurePosixPath

from src.domain.exceptions import StorageError

logger = logging.getLogger(__name__)

# Archivos y extensiones excluidos rigurosamente según PRD 6.1.3 y 12.1.7
EXCLUDED_EXTENSIONS: set[str] = {
    ".lock",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".min.js",
    ".min.css",
    ".map",
    ".wasm",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".pyc",
    ".parquet",
}

EXCLUDED_FILENAMES: set[str] = {
    "poetry.lock",
    "uv.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Pipfile.lock",
}

EXCLUDED_DIR_PATTERNS: tuple[str, ...] = (
    "tests/fixtures/",
    "tests/data/",
    "node_modules/",
    ".git/",
    "dist/",
    "build/",
)


class ShallowGitEngine:
    """Motor de Fallback de Git Superficial para diffs masivos en monorepos."""

    MAX_FILES_THRESHOLD = 300
    MAX_LINES_THRESHOLD = 3000

    def is_excluded_file(self, file_path: str) -> bool:
        """Determina si un archivo debe ser descartado según las reglas de exclusión estricta."""
        norm_path = PurePosixPath(file_path.replace("\\", "/"))
        filename = norm_path.name.lower()

        if filename in EXCLUDED_FILENAMES:
            return True

        if (
            any(filename.endswith(ext) for ext in EXCLUDED_EXTENSIONS)
            or norm_path.suffix.lower() in EXCLUDED_EXTENSIONS
        ):
            return True

        path_str = str(norm_path).lower()
        return any(excluded_dir.lower() in path_str for excluded_dir in EXCLUDED_DIR_PATTERNS)

    def should_use_shallow_fallback(
        self, total_files: int, total_lines: int, is_truncated: bool = False
    ) -> bool:
        """Evalúa si un PR excede los límites de la API y requiere el fallback superficial."""
        return (
            is_truncated
            or total_files > self.MAX_FILES_THRESHOLD
            or total_lines > self.MAX_LINES_THRESHOLD
        )

    async def fetch_commit_files_shallow(
        self,
        repo_url: str,
        commit_sha: str,
        target_prefixes: tuple[str, ...] = ("libs/core/", "docs/"),
    ) -> list[tuple[str, str]]:
        """Ejecuta un fetch efímero superficial (--depth 2 --filter=blob:none) para extraer archivos.

        Retorna:
            Lista de tuplas (file_path, content_text)
        """
        temp_dir = tempfile.mkdtemp(prefix="radar_shallow_")
        try:
            # 1. Clon superficial con filtro de blobs para no descargar el historial completo
            clone_cmd = [
                "git",
                "clone",
                "--bare",
                "--depth",
                "2",
                "--filter=blob:none",
                "--no-checkout",
                repo_url,
                temp_dir,
            ]
            proc = await asyncio.create_subprocess_exec(
                *clone_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise StorageError(
                    f"Fallo al ejecutar git clone superficial: {stderr.decode('utf-8', errors='replace')}"
                )

            # 2. Listar archivos del commit
            list_cmd = ["git", "--git-dir", temp_dir, "ls-tree", "-r", "--name-only", commit_sha]
            proc_ls = await asyncio.create_subprocess_exec(
                *list_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_ls, stderr_ls = await proc_ls.communicate()
            if proc_ls.returncode != 0:
                raise StorageError(
                    f"Fallo al listar archivos del commit {commit_sha}: {stderr_ls.decode('utf-8', errors='replace')}"
                )

            file_paths = stdout_ls.decode("utf-8").splitlines()
            result: list[tuple[str, str]] = []

            for path in file_paths:
                path_clean = path.strip()
                if not path_clean:
                    continue

                if self.is_excluded_file(path_clean):
                    continue

                # Filtrar solo prefijos de interés (ej. libs/core/ o docs/)
                if target_prefixes and not any(
                    path_clean.startswith(pref) for pref in target_prefixes
                ):
                    continue

                # 3. Leer contenido del archivo directamente desde el árbol git
                cat_cmd = ["git", "--git-dir", temp_dir, "show", f"{commit_sha}:{path_clean}"]
                proc_cat = await asyncio.create_subprocess_exec(
                    *cat_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_cat, _ = await proc_cat.communicate()
                if proc_cat.returncode == 0:
                    text_content = stdout_cat.decode("utf-8", errors="replace")
                    result.append((path_clean, text_content))

            return result

        finally:
            # Purgar directorio efímero (regla de workers stateless PRD 12.1.1)
            shutil.rmtree(temp_dir, ignore_errors=True)
