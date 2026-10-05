from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración centralizada y tipada de la aplicación."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Entorno
    app_env: Literal["development", "testing", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    # Servidor FastAPI
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT", ge=1024, le=65535)
    secret_key: str = Field(default="dev-secret-key-change-in-production", alias="SECRET_KEY")

    # Ingesta GitHub
    github_token: str = Field(default="", alias="GITHUB_TOKEN")
    github_webhook_secret: str = Field(default="", alias="GITHUB_WEBHOOK_SECRET")
    github_repo_owner: str = Field(default="langchain-ai", alias="GITHUB_REPO_OWNER")
    github_repo_name: str = Field(default="langchain", alias="GITHUB_REPO_NAME")

    # Redis (Deduplicación & Streams)
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT", ge=1, le=65535)
    redis_db: int = Field(default=0, alias="REDIS_DB", ge=0, le=15)
    redis_stream_key: str = Field(default="stream:github_events", alias="REDIS_STREAM_KEY")
    redis_consumer_group: str = Field(default="radar_workers", alias="REDIS_CONSUMER_GROUP")
    redis_dedup_ttl_seconds: int = Field(default=86400, alias="REDIS_DEDUP_TTL_SECONDS", gt=0)

    # MinIO Lakehouse (Bronze & Silver)
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")
    minio_bronze_bucket: str = Field(default="bronze", alias="MINIO_BRONZE_BUCKET")
    minio_silver_bucket: str = Field(default="silver", alias="MINIO_SILVER_BUCKET")

    # Qdrant Vector DB (Gold)
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT", ge=1, le=65535)
    qdrant_grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT", ge=1, le=65535)
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection_active: str = Field(
        default="langchain_radar_active", alias="QDRANT_COLLECTION_ACTIVE"
    )
    qdrant_collection_prefix: str = Field(
        default="langchain_radar", alias="QDRANT_COLLECTION_PREFIX"
    )

    # Embeddings y Reranker
    embedding_model_name: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL_NAME")
    reranker_model_name: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL_NAME")
    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE", gt=0, le=512)
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve una instancia cacheada singleton de la configuración."""
    return Settings()
