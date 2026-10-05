class DomainError(Exception):
    """Excepción base para todos los errores del dominio en LangChain Radar."""

    def __init__(
        self, message: str, details: dict[str, str | int | float | bool] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class WebhookValidationError(DomainError):
    """Lanzada cuando un webhook entrante falla en la verificación HMAC SHA-256 o en formato."""


class SecurityViolationError(DomainError):
    """Lanzada ante la detección de credenciales o secretos prohibidos en el código/metadata."""


class ParsingError(DomainError):
    """Lanzada cuando falla el particionado sintáctico AST o Markdown."""


class CanonicalResolutionError(DomainError):
    """Lanzada cuando no se puede resolver la ruta canónica de importación pública."""


class StorageError(DomainError):
    """Lanzada ante fallos en operaciones de persistencia (MinIO, Delta Lake)."""


class VectorStoreError(DomainError):
    """Lanzada ante fallos en Qdrant (upsert, delete, alias swap, search)."""


class RateLimitExceededError(DomainError):
    """Lanzada cuando se superan las cuotas de API externa de GitHub."""


class EventQueueError(DomainError):
    """Lanzada ante fallos de encolado o consumo en Redis Streams."""
