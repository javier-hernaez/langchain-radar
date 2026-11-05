# LangChain Radar 📡

> Pipeline RAG de Ingesta Continua, Desacoplada y de Alta Precisión para el Ecosistema LangChain.

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy.readthedocs.io/)
[![Architecture: Hexagonal](https://img.shields.io/badge/architecture-hexagonal-green)](https://en.wikipedia.org/wiki/Hexagonal_architecture_(software))

---

## 1. Visión General

**LangChain Radar** elimina el problema de **Knowledge Cutoff** en modelos de lenguaje comerciales e internos al ingerir, normalizar, segmentar sintácticamente y vectorizar en tiempo real cada cambio, PR fusionado (`status: MERGED`), commit y documentación técnica de LangChain.

### Características Clave
* **Arquitectura Medallion en Data Lakehouse:**
  * **Bronze:** JSON crudo e inmutable almacenado con particionamiento temporal en MinIO S3.
  * **Silver:** Tablas **Delta Lake** (`delta-rs` / PyArrow) con transaccionalidad ACID y validación de esquemas.
  * **Gold:** **Qdrant Vector DB** con índices híbridos (Dense + Sparse Lexical) y alias atómicos Blue/Green.
* **Chunking Jerárquico Padre-Hijo por AST:** Fragmenta clases preservando la relación entre el esqueleto de clase y sus métodos hijos mediante enlaces relacionales `parent_chunk_id`.
* **Canonical Import Resolver:** Mapea automáticamente la estructura física de archivos a rutas idiomáticas públicas (`from langchain_core.language_models import BaseChatModel`).
* **Seguridad Pre-Ingesta (Secret Scrubber):** Sanitización estricta basada en firmas regex de Gitleaks para prevenir la indexación de credenciales.
* **Inferencia Local 100% CPU:** Embeddings densos (`bge-small-en-v1.5` por defecto), dispersos (`Qdrant/bm25`) y Cross-Encoder `bge-reranker-base` vía FastEmbed / ONNX Runtime, sin GPUs. Configurables con `EMBEDDING_MODEL_NAME`, `SPARSE_MODEL_NAME` y `RERANKER_MODEL_NAME`.
* **Ingesta Reactiva en Tiempo Real:** Receptor de Webhooks en FastAPI con verificación HMAC SHA-256 en $<20\text{ ms}$, deduplicación en Redis con `SETNX` (TTL 24h) y encolado en `Redis Streams`. El worker obtiene los ficheros del PR y su contenido completo en el commit de merge vía la API REST de GitHub (requiere `GITHUB_TOKEN`), reclama mensajes huérfanos con `XAUTOCLAIM` y deriva los fallos a una Dead-Letter Queue (`<stream>_dlq`).
* **Arranque seguro en producción:** con `APP_ENV=production` la aplicación se niega a arrancar si `SECRET_KEY` es el valor por defecto (o tiene menos de 32 caracteres) o si falta `GITHUB_WEBHOOK_SECRET`.

---

## 2. Estructura del Proyecto (Arquitectura Hexagonal)

```text
langchain-radar/
├── config/                  # Configuración centralizada tipada (.env)
├── docker/                  # docker-compose.yml (MinIO, Qdrant, Redis)
├── src/
│   ├── domain/              # Modelos inmutables Pydantic v2 y Ports/Interfaces
│   │   ├── models/          # CodeChunk, DocChunk, PRMergedPayload, SearchQuery
│   │   └── ports/           # VectorRepository, BronzeStorage, SilverStorage, etc.
│   ├── application/         # Casos de uso y lógica de procesamiento
│   │   ├── splitters/       # ASTPythonSplitter, CanonicalResolver, MDXParser
│   │   └── use_cases/       # BackfillUseCase, IngestEventUseCase, SearchCodeUseCase
│   ├── infrastructure/      # Adaptadores concretos
│   │   ├── storage/         # MinioBronzeStorage, DeltaSilverStorage
│   │   ├── vector/          # QdrantVectorAdapter (Híbrido + Blue/Green)
│   │   ├── embeddings/      # Fastembed (BGE-M3 denso/sparse, Cross-Encoder)
│   │   ├── queue/           # RedisStreamsAdapter
│   │   ├── security/        # SecretScrubber
│   │   └── github/          # GitHubGraphQLClient, GitHubRestClient, ShallowGitEngine, WebhookValidator
│   └── interfaces/          # Puntos de entrada (FastAPI y CLI)
│       ├── api/             # Endpoints /api/v1/search, /webhook, /health, /reindex
│       └── cli/             # manage.py
└── tests/                   # Suite completa con cobertura >= 85%
```

---

## 3. Puesta en Marcha Rápida (Quickstart)

### Requisitos Previos
* Python 3.12+
* Docker Desktop (opcional para servicios en producción; las pruebas locales usan clientes herméticos en memoria)

### Instalación y Entorno
```bash
# 1. Crear entorno e instalar dependencias con uv
uv venv
uv pip install -e ".[dev,embeddings]"

# 2. Configurar variables de entorno
cp .env.example .env
```

### Levantar Servicios Locales con Docker
```bash
docker compose -f docker/docker-compose.yml up -d
```
* **Qdrant:** `http://localhost:6333`
* **MinIO Console:** `http://localhost:9001` (user: `minioadmin` / pass: `minioadmin`)
* **Redis:** `localhost:6379`

---

## 4. Ejecución del Servidor y Comandos CLI

### Iniciar Servidor FastAPI
```bash
uv run uvicorn src.interfaces.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Documentación OpenAPI interactiva disponible en: `http://localhost:8000/docs`.

### Comandos de Operación CLI
```bash
# Iniciar servidor API
uv run python -m src.interfaces.cli.manage serve --port 8000

# Iniciar worker de ingesta en segundo plano (Redis Streams)
uv run python -m src.interfaces.cli.manage worker

# Ejecutar sincronización histórica (Backfill)
uv run python -m src.interfaces.cli.manage backfill --max-prs 50 --batch-size 64

# Desencadenar reindexación Blue/Green
uv run python -m src.interfaces.cli.manage reindex

# Ejecutar benchmark de evaluación RAG (Recall@5, MRR, NDCG)
uv run python -m src.interfaces.cli.manage eval
```

---

## 5. Endpoints de la API

| Método | Endpoint | Descripción |
| :---: | :--- | :--- |
| `POST` | `/api/v1/search` | Búsqueda semántica híbrida (First-Stage) con Cross-Encoder local (Second-Stage) y enriquecimiento de contexto padre. |
| `POST` | `/api/v1/webhook` | Receptor reactivo de GitHub Webhooks con HMAC SHA-256 y deduplicación atómica (<20ms). |
| `GET` | `/api/v1/health` | Liveness y readiness probes de componentes. |
| `GET` | `/api/v1/status/sync` | Estado de la sincronización y colección activa. |
| `POST` | `/api/v1/admin/reindex` | Migración Blue/Green zero-downtime con cambio atómico de alias en Qdrant. |
| `GET` | `/metrics` | Métricas Prometheus de latencia, peticiones y conteo de vectores activos. |

---

## 6. SDK de Cliente en Python

LangChain Radar incluye un cliente oficial sincrónico y asincrónico para integración directa en aplicaciones:

```python
from src.interfaces.client import RadarClient, AsyncRadarClient

# Uso asincrónico
async with AsyncRadarClient(base_url="http://localhost:8000") as client:
    health = await client.get_health()
    results = await client.search("how to bind tools to chat model", limit=5)
    for hit in results.results:
        print(f"[{hit.rerank_score:.3f}] {hit.canonical_import}: {hit.content[:80]}")

# Uso sincrónico
with RadarClient(base_url="http://localhost:8000") as client:
    status = client.get_sync_status()
    print(status)
```

---

## 7. Pruebas y Control de Calidad

```bash
# Formato y linter
uv run ruff check .
uv run ruff format --check .

# Tipado estático estricto
uv run mypy src tests

# Suite de pruebas unitarias y cobertura
uv run pytest --cov=src --cov-report=term-missing

# Benchmark de evaluación continua RAG
uv run python tests/evaluation/run_rag_eval.py
```
