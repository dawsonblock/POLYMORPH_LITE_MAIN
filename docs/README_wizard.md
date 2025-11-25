# Polymorph-4 Hardware Wizard Add-on (v6)

This add-on pins your exact NI chassis/channels and Raman vendor into `config/config.yaml`.

## Install
1) Extract this ZIP into the root of your existing kit (v3/v4/v5).
2) Ensure hardware deps are installed:
   ```bash
   pip install -r requirements-hw.txt
   ```

## Run
```bash
python scripts/hardware_wizard.py
```
- Enumerates **NI** devices/channels (if `nidaqmx` present).
- Enumerates **Ocean** spectrometers (if `seabreeze` present).
- Prompts for **DI** interlock indices (E-stop/Door).
- Writes updated `config/config.yaml`.

> You can override path with `P4_CONFIG=path/to/config.yaml`.
