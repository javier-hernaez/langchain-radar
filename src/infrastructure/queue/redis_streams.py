import logging
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from config.settings import Settings, get_settings
from src.domain.exceptions import EventQueueError
from src.domain.ports.event_queue import EventQueue

logger = logging.getLogger(__name__)


class RedisStreamsAdapter(EventQueue):
    """Adaptador de infraestructura para Redis Streams, deduplicación atómica y Consumer Groups."""

    def __init__(
        self,
        client: aioredis.Redis | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if client:
            self._client = client
        else:
            self._client = aioredis.Redis(
                host=self._settings.redis_host,
                port=self._settings.redis_port,
                db=self._settings.redis_db,
                decode_responses=True,
            )

    async def is_duplicate_or_mark(self, event_id: str, ttl_seconds: int = 86400) -> bool:
        """Comprueba atómicamente si el evento ya fue recibido mediante SETNX.

        Retorna:
            True si es un duplicado (la clave ya existía).
            False si es un evento nuevo (se insertó con TTL de 24h).
        """
        key = f"event:{event_id}"
        try:
            # SETNX atómico con expiración
            was_set = await self._client.set(key, "1", ex=ttl_seconds, nx=True)
            # Si was_set es None o False, la clave ya existía -> es duplicado
            return not bool(was_set)
        except Exception as e:
            raise EventQueueError(f"Error al verificar deduplicación en Redis: {e}") from e

    async def unmark_event(self, event_id: str) -> None:
        """Elimina la clave de deduplicación para que un reintento del emisor sea aceptado."""
        try:
            await self._client.delete(f"event:{event_id}")
        except Exception as e:
            raise EventQueueError(f"Error al liberar deduplicación en Redis: {e}") from e

    async def enqueue_event(self, stream_name: str, event_id: str, payload_json: str) -> str:
        """Encola el evento en Redis Streams y devuelve el Message ID asignado."""
        try:
            fields = {"event_id": event_id, "payload": payload_json}
            res = await self._client.xadd(stream_name, fields)  # type: ignore[arg-type]
            return str(res)
        except Exception as e:
            raise EventQueueError(
                f"Error al encolar evento '{event_id}' en stream '{stream_name}': {e}"
            ) from e

    async def ensure_consumer_group(
        self, stream_name: str, group_name: str, start_id: str = "$"
    ) -> None:
        """Crea el Consumer Group en el Stream si no existe (idempotente)."""
        try:
            await self._client.xgroup_create(
                name=stream_name, groupname=group_name, id=start_id, mkstream=True
            )
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                # El grupo ya existe, operación idempotente
                pass
            else:
                raise EventQueueError(f"Error al crear consumer group en Redis: {e}") from e
        except Exception as e:
            raise EventQueueError(f"Error inesperado con consumer group en Redis: {e}") from e

    async def read_events(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 2000,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Lee mensajes asignados al consumer group.

        Retorna:
            Lista de tuplas (message_id, fields_dict)
        """
        try:
            streams_dict = {stream_name: ">"}
            raw_response: Any = await self._client.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams=streams_dict,  # type: ignore[arg-type]
                count=count,
                block=block_ms,
            )
            if not raw_response or not isinstance(raw_response, list):
                return []

            results: list[tuple[str, dict[str, Any]]] = []
            for stream_entry in raw_response:
                messages = stream_entry[1] if len(stream_entry) > 1 else []
                for msg in messages:
                    msg_id = str(msg[0])
                    fields = dict(msg[1]) if len(msg) > 1 else {}
                    results.append((msg_id, fields))
            return results
        except Exception as e:
            raise EventQueueError(
                f"Error al leer eventos de Redis Streams ({stream_name}): {e}"
            ) from e

    async def claim_stale_events(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        min_idle_ms: int = 300_000,
        count: int = 10,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Reclama mensajes pendientes de consumidores caídos (XAUTOCLAIM).

        Sin esto, un mensaje entregado a un worker que muere antes del XACK
        queda en la Pending Entries List indefinidamente.
        """
        try:
            raw_response: Any = await self._client.xautoclaim(
                name=stream_name,
                groupname=group_name,
                consumername=consumer_name,
                min_idle_time=min_idle_ms,
                start_id="0-0",
                count=count,
            )
            if not raw_response or len(raw_response) < 2:
                return []
            results: list[tuple[str, dict[str, Any]]] = []
            for msg in raw_response[1] or []:
                if not msg or msg[1] is None:
                    continue
                results.append((str(msg[0]), dict(msg[1])))
            return results
        except Exception as e:
            raise EventQueueError(
                f"Error al reclamar mensajes pendientes de {stream_name}: {e}"
            ) from e

    async def ack_event(self, stream_name: str, group_name: str, message_id: str) -> None:
        """Confirma el procesamiento exitoso de un mensaje en el consumer group."""
        try:
            await self._client.xack(stream_name, group_name, message_id)
        except Exception as e:
            raise EventQueueError(
                f"Error al ejecutar XACK para mensaje {message_id} en {stream_name}: {e}"
            ) from e

    async def send_to_dlq(
        self,
        stream_name: str,
        event_id: str,
        error_message: str,
        payload_json: str,
    ) -> str:
        """Envía un evento fallido permanentemente a la Dead-Letter Queue (DLQ)."""
        dlq_stream = f"{stream_name}_dlq"
        try:
            fields = {
                "event_id": event_id,
                "error": error_message,
                "payload": payload_json,
            }
            res = await self._client.xadd(dlq_stream, fields)  # type: ignore[arg-type]
            msg_id = str(res)
            logger.error("Evento %s derivado a DLQ %s (id: %s)", event_id, dlq_stream, msg_id)
            return msg_id
        except Exception as e:
            raise EventQueueError(
                f"Fallo crítico al enviar evento a DLQ ({dlq_stream}): {e}"
            ) from e
