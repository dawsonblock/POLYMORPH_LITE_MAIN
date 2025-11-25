#!/usr/bin/env python3
import os, sys, yaml
print("== Polymorph-4 Hardware Wizard ==")
CFG = os.environ.get("P4_CONFIG", "config/config.yaml")
with open(CFG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# NI enum
ni_devices, ni_info = [], {}
try:
    import nidaqmx
    from nidaqmx.system import System
    sys_local = System.local()
    for d in sys_local.devices:
        ni_devices.append(d.name)
        ni_info[d.name] = {
            "ai": [f"{d.name}/{c.name}" for c in d.ai_physical_chans],
            "ao": [f"{d.name}/{c.name}" for c in d.ao_physical_chans],
            "di": [f"{d.name}/{c.name}" for c in d.di_lines],
            "do": [f"{d.name}/{c.name}" for c in d.do_lines],
        }
except Exception as e:
    print("(hint) NI enumeration not available:", e)

def choose(title, options, default_idx=0):
    if not options:
        return None
    print(f"\n{title}")
    for i,o in enumerate(options): print(f"  [{i}] {o}")
    s = input(f"Pick index [{default_idx}]: ").strip() or str(default_idx)
    try: i = int(s)
    except: i = default_idx
    return options[max(0, min(i, len(options)-1))]

dev = choose("NI Device", ni_devices, 0) if ni_devices else (input("NI device (Dev1): ").strip() or "Dev1")
ai_list = ni_info.get(dev,{}).get("ai", [f"{dev}/ai0"])
ao_list = ni_info.get(dev,{}).get("ao", [f"{dev}/ao0"])
di_list = ni_info.get(dev,{}).get("di", [f"{dev}/port0/line0", f"{dev}/port0/line1"])
do_list = ni_info.get(dev,{}).get("do", [f"{dev}/port0/line2"])

ai = (choose("AI channel", ai_list, 0) or f"{dev}/ai0").split("/",1)[-1]
ao = (choose("AO channel", ao_list, 0) or f"{dev}/ao0").split("/",1)[-1]
di_estop = (choose("DI for E-Stop", di_list, 0) or f"{dev}/port0/line0").split("/",1)[-1]
di_door  = (choose("DI for Door", di_list, 1 if len(di_list)>1 else 0) or f"{dev}/port0/line1").split("/",1)[-1]
do_wd    = (choose("DO for Watchdog", do_list, 0) or f"{dev}/port0/line2").split("/",1)[-1]

cfg["daq"]["backend"] = "ni"
cfg["daq"]["ni"]["device_name"] = dev
cfg["daq"]["ni"]["ai_voltage_channel"] = ai
cfg["daq"]["ni"]["ao_voltage_channel"] = ao
cfg["daq"]["ni"]["di_lines"] = [di_estop, di_door]
cfg["daq"]["ni"]["do_watchdog_line"] = do_wd

# Raman
vendors = ["ocean","horiba","andor","simulator"]
print("\nRaman providers:")
for i,v in enumerate(vendors): print(f"  [{i}] {v}")
sel = input("Provider [0]: ").strip() or "0"
try: sel = int(sel)
except: sel = 0
provider = vendors[max(0, min(sel, len(vendors)-1))]
cfg["raman"]["provider"] = provider

if provider == "ocean":
    try:
        from seabreeze.spectrometers import list_devices
        devs = list_devices()
        if devs:
            print("Ocean devices detected:")
            for i,d in enumerate(devs): print(f"  [{i}] {d}")
            odx = input("Ocean device index [0]: ").strip() or "0"
            cfg["raman"].setdefault("vendor", {})["ocean_index"] = int(odx)
    except Exception as e:
        print("(hint) Ocean enumeration not available:", e)

try:
    ei = int(input("\nIndex for E-Stop (0 first DI) [0]: ").strip() or "0")
    di = int(input("Index for Door (1 second DI) [1]: ").strip() or "1")
except:
    ei, di = 0, 1
cfg["safety"]["interlocks"]["estop_line"] = ei
cfg["safety"]["interlocks"]["door_line"] = di

with open(CFG, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("\nSaved ->", CFG)
print("Done.")
