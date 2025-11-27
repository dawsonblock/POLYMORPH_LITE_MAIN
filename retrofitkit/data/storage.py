import os
import json
import time
import csv
import uuid
import zipfile
import glob

class DataStore:
    def __init__(self, base_dir: str):
        self.base = base_dir
        os.makedirs(self.base, exist_ok=True)

    def start_run(self, recipe_name: str, operator: str, simulation: bool) -> str:
        rid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        run_dir = os.path.join(self.base, rid)
        os.makedirs(run_dir, exist_ok=True)
        meta = {"recipe": recipe_name, "operator": operator, "simulation": simulation, "t0": time.time(), "state": "RUNNING"}
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(run_dir, "spectra.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t", "peak_nm", "peak_intensity"])
        return rid

    def append_spectrum(self, rid: str, spec: dict):
        run_dir = os.path.join(self.base, rid)
        with open(os.path.join(run_dir, "spectra.csv"), "a", newline="") as f:
            w = csv.writer(f); w.writerow([spec["t"], spec["peak_nm"], spec["peak_intensity"]])

    def list_runs(self, limit=200):
        runs = []
        for d in sorted(glob.glob(os.path.join(self.base, "*")), reverse=True)[:limit]:
            mp = os.path.join(d, "meta.json")
            if os.path.exists(mp):
                with open(mp, "r", encoding="utf-8") as f:
                    m = json.load(f)
                m["id"] = os.path.basename(d)
                runs.append(m)
        return runs

    def package_run(self, rid: str) -> str:
        run_dir = os.path.join(self.base, rid)
        pkg = os.path.join(run_dir, f"{rid}_package.zip")
        with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ["meta.json", "spectra.csv"]:
                p = os.path.join(run_dir, name)
                if os.path.exists(p):
                    z.write(p, os.path.basename(p))
        return pkg
