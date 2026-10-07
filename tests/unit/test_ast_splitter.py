from datetime import UTC, datetime

from src.application.splitters.ast_python import ASTPythonSplitter


def test_ast_splitter_hierarchical_parent_child() -> None:
    code = '''
class BaseChatModel:
    """Clase base para todos los modelos de chat."""

    def invoke(self, input_data: str) -> str:
        """Invoca el modelo con una entrada de texto."""
        return input_data.upper()

    def bind_tools(self, tools: list) -> "BaseChatModel":
        """Vincula herramientas estructuradas al modelo."""
        return self


def helper_function(x: int) -> int:
    """Función de utilidad independiente."""
    return x * 2
'''
    splitter = ASTPythonSplitter()
    chunks = splitter.split_file(
        file_path="libs/core/langchain_core/chat_models.py",
        source_code=code,
        commit_sha="abcdef1234567890abcdef1234567890abcdef12",
        commit_date=datetime.now(UTC),
        release_version="v0.3.1",
        author="hwchase17",
    )

    # Debe haber:
    # 1. Chunk Padre (BaseChatModel esqueleto)
    # 2. Chunk Hijo (BaseChatModel.invoke)
    # 3. Chunk Hijo (BaseChatModel.bind_tools)
    # 4. Chunk Función suelta (helper_function)
    assert len(chunks) == 4

    parent_chunk = chunks[0]
    assert parent_chunk.metadata.entity_name == "BaseChatModel"
    assert parent_chunk.metadata.is_parent_class is True
    assert parent_chunk.metadata.parent_chunk_id is None
    assert "class BaseChatModel:" in parent_chunk.content
    assert "invoke" in parent_chunk.content
    assert "bind_tools" in parent_chunk.content

    child_invoke = [c for c in chunks if c.metadata.entity_name == "BaseChatModel.invoke"][0]
    assert child_invoke.metadata.is_parent_class is False
    assert child_invoke.metadata.parent_chunk_id == parent_chunk.metadata.chunk_id
    assert "# [Contexto Canónico Inyectado]" in child_invoke.content
    assert "def invoke" in child_invoke.content

    child_tools = [c for c in chunks if c.metadata.entity_name == "BaseChatModel.bind_tools"][0]
    assert child_tools.metadata.parent_chunk_id == parent_chunk.metadata.chunk_id
    assert "def bind_tools" in child_tools.content

    standalone = [c for c in chunks if c.metadata.entity_name == "helper_function"][0]
    assert standalone.metadata.entity_type == "function"
    assert standalone.metadata.parent_chunk_id is None


def test_ast_splitter_redacts_secrets() -> None:
    code = """
def run_llm():
    api_key = "sk-proj-1234567890abcdef1234567890abcdef12345678"
    return api_key
"""
    splitter = ASTPythonSplitter()
    chunks = splitter.split_file(
        file_path="src/llm.py",
        source_code=code,
        commit_sha="1111222233334444555566667777888899990000",
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert "sk-proj-" not in chunk.content
    assert "[REDACTED_SECRET]" in chunk.content


def test_ast_splitter_detects_deprecated() -> None:
    code = """
@deprecated("Use new_function instead")
def old_function():
    pass
"""
    splitter = ASTPythonSplitter()
    chunks = splitter.split_file(
        file_path="src/legacy.py",
        source_code=code,
        commit_sha="1111222233334444555566667777888899990000",
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.is_deprecated is True
