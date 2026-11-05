import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager


class PrometheusMetrics:
    """Colector de métricas en memoria conforme al formato de exposición de Prometheus."""

    _instance: "PrometheusMetrics | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._gauges: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Solo se acumulan (count, sum): guardar cada observación crecería sin límite
        self._summaries: dict[str, dict[tuple[tuple[str, str], ...], list[float]]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0.0])
        )
        self._descriptions: dict[str, str] = {
            "github_api_requests_total": "Total de peticiones realizadas a la API de GitHub",
            "pipeline_processing_time_seconds": "Tiempo de procesamiento por fase del pipeline",
            "indexed_chunks_total": "Total de chunks indexados en Qdrant por colección",
            "http_requests_total": "Total de peticiones HTTP por método, ruta y estado",
            "http_request_duration_seconds": "Latencia de las peticiones HTTP en segundos",
            "search_requests_total": "Total de consultas de búsqueda recibidas",
            "search_latency_seconds": "Latencia en segundos de las operaciones de búsqueda",
        }

    @classmethod
    def get_instance(cls) -> "PrometheusMetrics":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _labels_to_key(self, labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def inc_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        key = self._labels_to_key(labels)
        with self._lock:
            self._counters[name][key] += value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._labels_to_key(labels)
        with self._lock:
            self._gauges[name][key] = value

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        key = self._labels_to_key(labels)
        with self._lock:
            stats = self._summaries[name][key]
            stats[0] += 1
            stats[1] += value

    @contextmanager
    def measure_time(self, name: str, labels: dict[str, str] | None = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.observe_histogram(name, elapsed, labels)

    def generate_prometheus_text(self) -> str:
        """Genera el volcado en formato estándar Prometheus (text/plain; version=0.0.4)."""
        lines: list[str] = []

        with self._lock:
            # Counters
            for name, labeled_data in sorted(self._counters.items()):
                desc = self._descriptions.get(name, f"Counter {name}")
                lines.append(f"# HELP {name} {desc}")
                lines.append(f"# TYPE {name} counter")
                for labels_tuple, val in sorted(labeled_data.items()):
                    if labels_tuple:
                        lbl_str = ",".join(f'{k}="{v}"' for k, v in labels_tuple)
                        lines.append(f"{name}{{{lbl_str}}} {val}")
                    else:
                        lines.append(f"{name} {val}")

            # Gauges
            for name, labeled_data in sorted(self._gauges.items()):
                desc = self._descriptions.get(name, f"Gauge {name}")
                lines.append(f"# HELP {name} {desc}")
                lines.append(f"# TYPE {name} gauge")
                for labels_tuple, val in sorted(labeled_data.items()):
                    if labels_tuple:
                        lbl_str = ",".join(f'{k}="{v}"' for k, v in labels_tuple)
                        lines.append(f"{name}{{{lbl_str}}} {val}")
                    else:
                        lines.append(f"{name} {val}")

            # Duraciones: sin buckets, el tipo Prometheus válido es "summary" (_count y _sum)
            for name, summary_data in sorted(self._summaries.items()):
                desc = self._descriptions.get(name, f"Summary {name}")
                lines.append(f"# HELP {name} {desc}")
                lines.append(f"# TYPE {name} summary")
                for labels_tuple, (count_f, total_sum) in sorted(summary_data.items()):
                    count = int(count_f)
                    lbl_base = ",".join(f'{k}="{v}"' for k, v in labels_tuple)
                    lbl_count = f"{{{lbl_base}}}" if lbl_base else ""
                    lines.append(f"{name}_count{lbl_count} {count}")
                    lines.append(f"{name}_sum{lbl_count} {round(total_sum, 6)}")

        return "\n".join(lines) + "\n"


def get_metrics() -> PrometheusMetrics:
    """Devuelve la instancia singleton de PrometheusMetrics."""
    return PrometheusMetrics.get_instance()
