import hashlib
import hmac
import json
import logging
from typing import Any

from config.settings import Settings, get_settings
from src.domain.exceptions import WebhookValidationError

logger = logging.getLogger(__name__)


class WebhookValidator:
    """Validador criptográfico de firmas HMAC-SHA256 para GitHub Webhooks."""

    def __init__(self, secret: str | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._secret = secret if secret is not None else self._settings.github_webhook_secret

    def verify_signature(self, payload_bytes: bytes, signature_header: str | None) -> bool:
        """Verifica la firma HMAC-SHA256 provista en la cabecera X-Hub-Signature-256."""
        if not self._secret:
            # Si no hay secreto configurado en dev, registrar advertencia
            logger.warning(
                "GITHUB_WEBHOOK_SECRET no configurado; omitiendo validación criptográfica."
            )
            return True

        if not signature_header:
            return False

        if not signature_header.startswith("sha256="):
            return False

        expected_signature = signature_header[7:]
        mac = hmac.new(self._secret.encode("utf-8"), msg=payload_bytes, digestmod=hashlib.sha256)
        computed_signature = mac.hexdigest()

        return hmac.compare_digest(computed_signature, expected_signature)

    def validate_and_parse(
        self, payload_bytes: bytes, signature_header: str | None
    ) -> dict[str, Any]:
        """Verifica la firma y retorna el JSON parseado. Lanza WebhookValidationError si es inválido."""
        if not self.verify_signature(payload_bytes, signature_header):
            raise WebhookValidationError(
                "Firma HMAC-SHA256 inválida o ausente en la cabecera X-Hub-Signature-256"
            )

        try:
            data = json.loads(payload_bytes.decode("utf-8"))
            if not isinstance(data, dict):
                raise WebhookValidationError("El payload JSON debe ser un objeto/diccionario")
            return data
        except UnicodeDecodeError as e:
            raise WebhookValidationError(f"Error de decodificación UTF-8 en el webhook: {e}") from e
        except json.JSONDecodeError as e:
            raise WebhookValidationError(f"Payload JSON malformado: {e}") from e
