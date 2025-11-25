#!/usr/bin/env python3
import os, sys, yaml, shutil
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: apply_overlay.py <overlay_name> <kit_dir>")
    print("Overlays:", ", ".join([p.name for p in Path('overlays').iterdir() if p.is_dir()]))
    sys.exit(1)

overlay, kit_dir = sys.argv[1], sys.argv[2]
ov_path = Path('overlays')/overlay/'config.yaml'
cfg_path = Path(kit_dir)/'config'/'config.yaml'

if not ov_path.exists():
    print("Overlay not found:", ov_path); sys.exit(2)
if not cfg_path.exists():
    print("Kit config.yaml not found in", cfg_path); sys.exit(3)

with open(cfg_path, 'r', encoding='utf-8') as f:
    base = yaml.safe_load(f) or {}
with open(ov_path, 'r', encoding='utf-8') as f:
    ov = yaml.safe_load(f) or {}

# Merge shallow keys
for k,v in ov.items():
    if isinstance(v, dict):
        base.setdefault(k,{})
        base[k].update(v)
    else:
        base[k] = v

with open(cfg_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(base, f, sort_keys=False)

print("Applied overlay:", overlay, "->", cfg_path)
