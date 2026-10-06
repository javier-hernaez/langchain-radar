import io
import urllib.parse

from minio import Minio
from minio.error import S3Error

from config.settings import Settings, get_settings
from src.domain.exceptions import StorageError
from src.domain.ports.object_store import BronzeStorage


class MinioBronzeStorage(BronzeStorage):
    """Adaptador de infraestructura para la capa Bronze utilizando MinIO / S3."""

    def __init__(self, client: Minio | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._bucket = self._settings.minio_bronze_bucket
        if client:
            self._client = client
        else:
            self._client = Minio(
                endpoint=self._settings.minio_endpoint,
                access_key=self._settings.minio_access_key,
                secret_key=self._settings.minio_secret_key,
                secure=self._settings.minio_secure,
            )

    def ensure_bucket(self) -> None:
        """Crea el bucket bronze si no existe."""
        try:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
        except Exception as e:
            raise StorageError(
                f"Fallo al verificar o crear el bucket '{self._bucket}': {e}",
                {"bucket": self._bucket},
            ) from e

    async def save_raw_event(
        self, event_id: str, year: int, month: int, day: int, raw_payload: str
    ) -> str:
        """Persiste el payload JSON inmutable en la partición temporal de Bronze."""
        object_name = (
            f"github/{self._settings.github_repo_name}/"
            f"year={year}/month={month:02d}/day={day:02d}/{event_id}.json"
        )
        data_bytes = raw_payload.encode("utf-8")
        stream = io.BytesIO(data_bytes)
        try:
            self._client.put_object(
                bucket_name=self._bucket,
                object_name=object_name,
                data=stream,
                length=len(data_bytes),
                content_type="application/json",
            )
            return f"s3://{self._bucket}/{object_name}"
        except Exception as e:
            raise StorageError(
                f"Error al guardar evento crudo en MinIO: {e}",
                {"bucket": self._bucket, "object_name": object_name},
            ) from e

    async def get_raw_event(self, s3_uri: str) -> str:
        """Recupera el contenido JSON crudo a partir de su URI s3://bucket/key."""
        parsed = urllib.parse.urlparse(s3_uri)
        if parsed.scheme != "s3":
            raise StorageError(f"Esquema de URI no soportado (esperado s3://): {s3_uri}")

        bucket_name = parsed.netloc
        object_name = parsed.path.lstrip("/")
        try:
            response = self._client.get_object(bucket_name, object_name)
            try:
                content = response.read().decode("utf-8")
                return content
            finally:
                response.close()
                response.release_conn()
        except S3Error as e:
            raise StorageError(
                f"Objeto no encontrado o error S3 en {s3_uri}: {e}",
                {"s3_uri": s3_uri},
            ) from e
        except Exception as e:
            raise StorageError(
                f"Error inesperado al leer objeto de MinIO {s3_uri}: {e}",
                {"s3_uri": s3_uri},
            ) from e
