import json
import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Response, status

from config.settings import get_settings
from src.domain.ports.event_queue import EventQueue
from src.domain.ports.object_store import BronzeStorage
from src.infrastructure.github.webhook_validator import WebhookValidator
from src.infrastructure.queue.redis_streams import RedisStreamsAdapter
from src.infrastructure.storage.minio_bronze import MinioBronzeStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Webhooks"])

# Instancias compartidas por inyección o defaults
_validator: WebhookValidator | None = None
_event_queue: EventQueue | None = None
_bronze_storage: BronzeStorage | None = None


def get_validator() -> WebhookValidator:
    global _validator
    if _validator is None:
        _validator = WebhookValidator()
    return _validator


def get_queue() -> EventQueue:
    global _event_queue
    if _event_queue is None:
        _event_queue = RedisStreamsAdapter()
    return _event_queue


def get_bronze() -> BronzeStorage:
    global _bronze_storage
    if _bronze_storage is None:
        _bronze_storage = MinioBronzeStorage()
    return _bronze_storage


def set_dependencies(
    validator: WebhookValidator | None = None,
    queue: EventQueue | None = None,
    bronze: BronzeStorage | None = None,
) -> None:
    """Permite inyectar dependencias mockeadas en tests herméticos."""
    global _validator, _event_queue, _bronze_storage
    _validator = validator
    _event_queue = queue
    _bronze_storage = bronze


@router.post(
    "/webhook",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Receptor seguro de GitHub Webhooks",
    description="Valida la firma HMAC-SHA256, deduplica en Redis y encola el evento en <20ms",
)
async def handle_github_webhook(
    request: Request,
    response: Response,
    x_github_delivery: Annotated[str | None, Header()] = None,
    x_github_event: Annotated[str | None, Header()] = None,
    x_hub_signature_256: Annotated[str | None, Header()] = None,
) -> dict[str, str]:
    """Endpoint reactivo ultrarrápido (<20ms) para ingesta de webhooks."""
    raw_body = await request.body()
    validator = get_validator()
    queue = get_queue()
    bronze = get_bronze()
    settings = get_settings()

    # 1. Manejo de ping inicial de GitHub
    if x_github_event == "ping":
        response.status_code = status.HTTP_200_OK
        return {"status": "pong"}

    # 2. Validación de firma criptográfica
    if not validator.verify_signature(raw_body, x_hub_signature_256):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firma criptográfica HMAC-SHA256 inválida o ausente",
        )

    # 3. Decodificar payload antes de marcar el evento, para no deduplicar basura
    try:
        payload_data = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cuerpo de la petición JSON malformado: {e}",
        ) from e
    if not isinstance(payload_data, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El payload JSON debe ser un objeto",
        )

    # Si no hay delivery id provisto, generar uno
    event_id = x_github_delivery or f"evt_{datetime.now(UTC).timestamp()}"

    # 4. Deduplicación atómica en Redis (SETNX 24h)
    is_duplicate = await queue.is_duplicate_or_mark(
        event_id=event_id, ttl_seconds=settings.redis_dedup_ttl_seconds
    )
    if is_duplicate:
        # Evento retransmitido por GitHub: responder 200 sin reprocesar
        response.status_code = status.HTTP_200_OK
        return {"status": "already_processed", "event_id": event_id}

    now = datetime.now(UTC)
    try:
        # 5. Persistencia raw inmutable en Bronze Lakehouse
        await bronze.save_raw_event(
            event_id=event_id,
            year=now.year,
            month=now.month,
            day=now.day,
            raw_payload=raw_body.decode("utf-8"),
        )

        # 6. Encolado reactivo en Redis Streams para procesamiento desacoplado
        queue_payload = json.dumps(
            {
                "event_id": event_id,
                "event_type": x_github_event or "unknown",
                "action": payload_data.get("action", ""),
                "received_at": now.isoformat(),
                "payload": payload_data,
            }
        )
        await queue.enqueue_event(
            stream_name=settings.redis_stream_key,
            event_id=event_id,
            payload_json=queue_payload,
        )
    except Exception:
        # Liberar la marca: si no, el reintento de GitHub se descartaría como duplicado
        logger.exception("Fallo al persistir/encolar el evento %s; se libera la marca", event_id)
        await queue.unmark_event(event_id)
        raise

    response.status_code = status.HTTP_202_ACCEPTED
    return {"status": "accepted", "event_id": event_id}
