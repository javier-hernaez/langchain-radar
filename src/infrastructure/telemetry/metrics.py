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
        self._histograms: dict[str, dict[tuple[tuple[str, str], ...], list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._descriptions: dict[str, str] = {
            "github_api_requests_total": "Total de peticiones realizadas a la API de GitHub",
            "pipeline_processing_time_seconds": "Histograma de tiempo de procesamiento por fase",
            "active_vectors_count": "Conteo en tiempo real de vectores activos por colección",
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
            self._histograms[name][key].append(value)

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

            # Histograms (Count and Sum summary)
            for name, hist_data in sorted(self._histograms.items()):
                desc = self._descriptions.get(name, f"Histogram {name}")
                lines.append(f"# HELP {name} {desc}")
                lines.append(f"# TYPE {name} histogram")
                for labels_tuple, observations in sorted(hist_data.items()):
                    count = len(observations)
                    total_sum = sum(observations)
                    lbl_base = ",".join(f'{k}="{v}"' for k, v in labels_tuple)
                    lbl_count = f"{{{lbl_base}}}" if lbl_base else ""
                    lines.append(f"{name}_count{lbl_count} {count}")
                    lines.append(f"{name}_sum{lbl_count} {round(total_sum, 6)}")

        return "\n".join(lines) + "\n"


def get_metrics() -> PrometheusMetrics:
    """Devuelve la instancia singleton de PrometheusMetrics."""
    return PrometheusMetrics.get_instance()
