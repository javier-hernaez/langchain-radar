from src.application.splitters.mdx_parser import MDXParser


def test_mdx_parser_header_hierarchy_and_code_preservation() -> None:
    doc = """# LangChain Radar

Esta es la introducción.

## Instalación

Para instalar el paquete ejecuta:

```python
pip install langchain-radar
import langchain_radar
print("installed")
```

## Uso Básico

A continuación describimos el uso básico.

### Instanciación de Clases

Aquí explicamos cómo usar LCEL:

```python
chain = prompt | model | parser
chain.invoke("hello")
```
"""
    parser = MDXParser()
    chunks = parser.parse_document(
        file_path="docs/quickstart.md",
        text_content=doc,
        commit_sha="1111222233334444555566667777888899990000",
    )

    assert len(chunks) == 4

    # Sección 1: Introducción (# LangChain Radar)
    sec_1 = chunks[0]
    assert sec_1.header_path == ("LangChain Radar",)
    assert "Esta es la introducción." in sec_1.content

    # Sección 2: Instalación (## Instalación)
    sec_2 = chunks[1]
    assert sec_2.header_path == ("LangChain Radar", "Instalación")
    assert "```python" in sec_2.content
    assert "pip install langchain-radar" in sec_2.content

    # Sección 3: Uso Básico (## Uso Básico)
    sec_3 = chunks[2]
    assert sec_3.header_path == ("LangChain Radar", "Uso Básico")
    assert "A continuación describimos el uso básico." in sec_3.content

    # Sección 4: Subsección (### Instanciación de Clases)
    sec_4 = chunks[3]
    assert sec_4.header_path == ("LangChain Radar", "Uso Básico", "Instanciación de Clases")
    assert "chain = prompt | model | parser" in sec_4.content


def test_mdx_parser_redacts_secrets_in_docs() -> None:
    doc = """# Quickstart

Guarda tu clave de API:

```python
api_key = "sk-ant-1234567890abcdef1234567890abcdef12345678"
```
"""
    parser = MDXParser()
    chunks = parser.parse_document(
        file_path="docs/keys.mdx",
        text_content=doc,
        commit_sha="1111222233334444555566667777888899990000",
    )

    assert len(chunks) == 1
    assert "sk-ant-" not in chunks[0].content
    assert "[REDACTED_SECRET]" in chunks[0].content
    assert chunks[0].metadata.language == "mdx"
