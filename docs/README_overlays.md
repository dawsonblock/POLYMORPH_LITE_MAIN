# Polymorph-4 Config Overlays (v6)

Prewired configurations you can apply to any Polymorph-4 Retrofit Kit (v3+).

## Usage
```bash
unzip Polymorph4_Config_Overlays_v6.zip -d overlays_pack
cd overlays_pack
python apply_overlay.py NI_USB6343_Ocean0 /path/to/Polymorph4_Retrofit_Kit_vX
# or choose: NI_PCIE6363_Horiba | RedPitaya_Andor | NI_USB6343_Simulator
```
This patches your kit's `config/config.yaml` with production-ready DAQ, Raman, and safety settings.
