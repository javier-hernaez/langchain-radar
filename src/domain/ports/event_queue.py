from typing import Protocol


class EventQueue(Protocol):
    """Puerto de dominio para desacoplamiento reactivo, colas y deduplicación con Redis."""

    async def is_duplicate_or_mark(self, event_id: str, ttl_seconds: int = 86400) -> bool:
        """Comprueba atómicamente si el evento ya fue recibido. Devuelve True si es duplicado."""
        ...

    async def enqueue_event(self, stream_name: str, event_id: str, payload_json: str) -> str:
        """Encola el evento en Redis Streams con backpressure y devuelve el message ID."""
        ...

    async def ack_event(self, stream_name: str, group_name: str, message_id: str) -> None:
        """Confirma el procesamiento exitoso de un mensaje en el consumer group."""
        ...
