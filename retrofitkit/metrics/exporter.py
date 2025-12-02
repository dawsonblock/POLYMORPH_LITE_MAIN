from threading import Lock
from typing import Dict, Tuple, Any, Optional


class Metrics:
    """Thread-safe metrics collection with Prometheus format export."""
    
    _inst: Optional["Metrics"] = None
    __slots__ = ('_m', '_lock')
    
    def __init__(self):
        self._m: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], float] = {}
        self._lock = Lock()

    @classmethod
    def get(cls) -> "Metrics":
        """Get or create the singleton metrics instance."""
        if cls._inst is None:
            cls._inst = Metrics()
        return cls._inst

    @classmethod
    def start(cls) -> None:
        """Start metrics collection (noop for now)."""
        pass

    def set(self, name: str, value: float, labels: Optional[Dict[str, Any]] = None) -> None:
        """Set a metric value."""
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._m[key] = float(value)

    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, Any]] = None) -> None:
        """Increment a counter metric."""
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._m[key] = self._m.get(key, 0.0) + value

    def render_prom(self) -> str:
        """Render metrics in Prometheus format."""
        lines = []
        with self._lock:
            for (name, labels), val in self._m.items():
                if labels:
                    lab = ",".join(f'{k}="{v}"' for k, v in labels)
                    lines.append(f"{name}{{{lab}}} {val}")
                else:
                    lines.append(f"{name} {val}")
        return "\n".join(lines) + "\n" if lines else "\n"
