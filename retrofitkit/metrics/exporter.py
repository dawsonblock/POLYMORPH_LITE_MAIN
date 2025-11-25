from threading import Lock

class Metrics:
    _inst = None
    def __init__(self):
        self._m = {}
        self._lock = Lock()

    @classmethod
    def get(cls):
        if not cls._inst:
            cls._inst = Metrics()
        return cls._inst

    def set(self, name, value, labels=None):
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._m[key] = float(value)

    def render_prom(self) -> str:
        lines = []
        with self._lock:
            for (name, labels), val in self._m.items():
                if labels:
                    lab = ",".join([f"{k}=\"{v}\"" for k,v in labels])
                    lines.append(f"{name}{{{lab}}} {val}")
                else:
                    lines.append(f"{name} {val}")
        return "\n".join(lines) + "\n"
