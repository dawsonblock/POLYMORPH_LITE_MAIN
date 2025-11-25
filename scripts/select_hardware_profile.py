#!/usr/bin/env python3
import yaml, os, sys
CFG = "config/config.yaml"
if len(sys.argv) != 2:
    print("Usage: select_hardware_profile.py <profile.yaml>"); sys.exit(1)
prof = sys.argv[1]
with open(prof, "r", encoding="utf-8") as f:
    p = yaml.safe_load(f)
with open(CFG, "r", encoding="utf-8") as f:
    c = yaml.safe_load(f)
c["daq"]["ni"].update(p)
with open(CFG, "w", encoding="utf-8") as f:
    yaml.safe_dump(c, f, sort_keys=False)
print("Applied profile to", CFG)
