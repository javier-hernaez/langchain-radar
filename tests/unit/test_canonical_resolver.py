from src.application.splitters.canonical_resolver import CanonicalImportResolver


def test_canonical_resolver_with_registered_init() -> None:
    resolver = CanonicalImportResolver()

    init_code = (
        "from .chat_models import BaseChatModel\n"
        "from .llms import BaseLLM\n"
        '__all__ = ["BaseChatModel", "BaseLLM"]\n'
    )
    resolver.register_init_file("libs/core/langchain_core/language_models/__init__.py", init_code)

    resolved = resolver.resolve(
        file_path="libs/core/langchain_core/language_models/chat_models.py",
        entity_name="BaseChatModel.bind_tools",
    )
    assert resolved == "from langchain_core.language_models import BaseChatModel"


def test_canonical_resolver_fallback_monorepo_path() -> None:
    resolver = CanonicalImportResolver()

    # Sin __init__ registrado, debe usar la ruta de módulo idiomática eliminando prefijos libs/core/
    resolved = resolver.resolve(
        file_path="libs/core/langchain_core/tools/base.py",
        entity_name="BaseTool",
    )
    assert resolved == "from langchain_core.tools.base import BaseTool"


def test_canonical_resolver_strips_partners_prefix() -> None:
    resolver = CanonicalImportResolver()

    resolved = resolver.resolve(
        file_path="libs/partners/openai/langchain_openai/chat_models/base.py",
        entity_name="ChatOpenAI",
    )
    assert resolved == "from langchain_openai.chat_models.base import ChatOpenAI"
