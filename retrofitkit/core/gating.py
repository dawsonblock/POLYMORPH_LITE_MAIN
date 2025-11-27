from typing import Dict, Any
from collections import deque

class GatingEngine:
    def __init__(self, rules):
        self.rules = rules
        self.window = deque(maxlen=50)

    def update(self, spectrum: Dict[str, Any]) -> bool:
        # spectrum: {"t": seconds, "wavelengths": [...], "intensities": [...], "peak_nm": f, "peak_intensity": I}
        self.window.append(spectrum)
        for r in self.rules:
            if r["name"] == "peak_threshold":
                if r["direction"] == "above" and spectrum["peak_intensity"] >= r["threshold"]:
                    return True
                if r["direction"] == "below" and spectrum["peak_intensity"] <= r["threshold"]:
                    return True
            if r["name"] == "slope_stop":
                if len(self.window) >= 3:
                    y = [x["peak_intensity"] for x in list(self.window)[-5:]]
                    x = [x["t"] for x in list(self.window)[-5:]]
                    slope = (y[-1] - y[0]) / max(1e-6, (x[-1] - x[0]))
                    if slope <= r["slope_threshold"]:
                        return True
        return False
