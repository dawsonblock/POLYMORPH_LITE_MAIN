#!/usr/bin/env python3
import os, sys
tmpl = "docker/prometheus/prometheus.yml.tmpl"
out = "docker/prometheus/prometheus.yml"
target = os.environ.get("APP_TARGET") or (sys.argv[1] if len(sys.argv)>1 else "host.docker.internal:8000")
with open(tmpl,"r",encoding="utf-8") as f: txt = f.read()
txt = txt.replace("${APP_TARGET}", target)
with open(out,"w",encoding="utf-8") as f: f.write(txt)
print("Wrote", out, "with target", target)
