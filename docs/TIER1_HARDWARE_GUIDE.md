# Tier-1 Hardware Stack Guide
## NI DAQ + Ocean Optics Raman Spectrometer

> **Production-Ready Configuration for POLYMORPH_LITE v4.0**

---

## Overview

The Tier-1 hardware stack provides the recommended production configuration for POLYMORPH_LITE v4.0:

- **DAQ**: National Instruments USB-6343 (or equivalent multifunction I/O device)
- **Raman**: Ocean Optics USB2000+ (or equivalent spectrometer)

This combination provides reliable, high-precision measurements for polymorph discovery workflows.

---

## Hardware Requirements

### NI DAQ (USB-6343 or compatible)
- **Analog Input**: 8 channels, 16-bit, ±10V range
- **Analog Output**: 4 channels, 16-bit, ±10V range  
- **Digital I/O**: 32 lines (for safety interlocks)
- **USB Interface**: USB 2.0 or 3.0
- **Driver**: NI-DAQmx 21.x or later

### Ocean Optics Spectrometer (USB2000+ or compatible)
- **Wavelength Range**: 200-1100 nm
- **Resolution**: ~1.5 nm FWHM
- **Integration Time**: 1 ms - 65 seconds
- **USB Interface**: USB 2.0 or 3.0
- **Driver**: SeaBreeze (via python-seabreeze)

---

## Software Dependencies

### Python Packages

```bash
# Install NI DAQ support
pip install nidaqmx

# Install Ocean Optics support
pip install seabreeze

# Configure SeaBreeze backend
seabreeze_os_setup
```

### System Requirements

- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS 10.15+
- **Python**: 3.9 or later
- **RAM**: 8 GB minimum, 16 GB recommended
- **USB Ports**: 2x USB 2.0/3.0 ports

---

## Setup Instructions

### 1. Hardware Connection

1. Connect NI DAQ via USB
2. Connect Ocean Optics spectrometer via USB
3. Verify device LEDs indicate power/ready status

### 2. Driver Installation

**NI DAQ (Windows/Linux)**:
```bash
# Download and install NI-DAQmx from ni.com
# Verify installation
python -c "import nidaqmx; print(nidaqmx.system.System.local().driver_version)"
```

**Ocean Optics**:
```bash
# Install SeaBreeze
pip install seabreeze

# Run OS-specific setup
seabreeze_os_setup

# Verify detection
python -c "from seabreeze.spectrometers import list_devices; print(list_devices())"
```

### 3. Device Discovery

POLYMORPH_LITE v4.0 includes automatic device discovery:

```python
from retrofitkit.drivers.discovery import get_discovery_service

service = get_discovery_service()
results = service.discover_all()

# Get Tier-1 devices
tier1 = service.get_tier1_devices()
print(f"DAQ: {tier1['daq']}")
print(f"Raman: {tier1['raman']}")
```

### 4. Configuration

Create or update your config overlay:

```yaml
# config/overlays/tier1/config.yaml
system:
  name: "POLYMORPH_LITE v4.0 - Tier-1"
  environment: "production"

daq:
  backend: "ni"
  ni_device_name: "Dev1"  # Will be auto-detected
  ni_ao_channel: "ao0"
  ni_ai_channel: "ai0"
  
raman:
  provider: "ocean_optics"
  ocean_device_index: 0  # Will be auto-detected
  ocean_integration_time_us: 100000  # 100ms
```

---

## Calibration Procedures

### DAQ Calibration

1. **Voltage Offset Calibration**:
   ```python
   # Set AO to 0V and measure AI
   await daq.set_voltage(0.0)
   offset = await daq.read_ai(channel=0, samples=100)
   print(f"Offset: {offset:.6f} V")
   ```

2. **Voltage Gain Calibration**:
   ```python
   # Apply known voltage and verify readback
   await daq.set_voltage(5.0)
   readback = await daq.read_ai(channel=0, samples=100)
   gain = readback / 5.0
   print(f"Gain: {gain:.6f}")
   ```

### Raman Calibration

1. **Wavelength Calibration**:
   - Use mercury lamp or neon lamp with known emission lines
   - Verify peak positions match expected wavelengths
   - Acceptable tolerance: ±0.5 nm

2. **Intensity Calibration**:
   - Use NIST-traceable intensity standard
   - Capture spectrum and compare to reference
   - Document calibration factor

---

## Synchronized Workflow Execution

### Running the Tier-1 Sweep

```bash
# Execute via CLI
python -m retrofitkit.cli workflow run \
  --workflow workflows/tier1_daq_raman_sweep.yaml \
  --operator user@example.com

# Or via API
curl -X POST http://localhost:8000/api/workflows/execute \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "tier1_daq_raman_sweep",
    "operator_email": "user@example.com",
    "parameters": {}
  }'
```

### Workflow Steps

The Tier-1 workflow performs:

1. **Device Discovery**: Auto-detect and validate DAQ + Raman
2. **Initialization**: Configure both devices
3. **Voltage Sweep**: 
   - Set DAQ AO voltage
   - Read back AI voltage
   - Capture Raman spectrum
   - Send to AI for analysis
   - Repeat for voltage range
4. **Safety Shutdown**: Return DAQ to 0V
5. **Report Generation**: Create PDF/JSON report

---

## Troubleshooting

### Device Not Found

**NI DAQ**:
```bash
# List available devices
python -c "import nidaqmx; print([d.name for d in nidaqmx.system.System.local().devices])"

# Check USB connection
# Verify NI-DAQmx is installed
# Try different USB port
```

**Ocean Optics**:
```bash
# List available spectrometers
python -c "from seabreeze.spectrometers import list_devices; print(list_devices())"

# If empty:
# - Re-run seabreeze_os_setup
# - Check USB cable
# - Try different USB port
# - Verify device LED is on
```

### Communication Errors

1. **USB Bandwidth**: Connect devices to separate USB controllers
2. **Driver Conflicts**: Uninstall conflicting software
3. **Permissions**: Run as administrator (Windows) or with sudo (Linux)

### Data Quality Issues

1. **High Noise**:
   - Increase Raman integration time
   - Check for electromagnetic interference
   - Verify shielding and grounding

2. **Voltage Mismatch**:
   - Re-run DAQ calibration
   - Check wiring connections
   - Verify channel configuration

---

## Performance Benchmarks

### Expected Performance

- **Device Discovery**: < 2 seconds
- **DAQ AO/AI**: < 10 ms per operation
- **Raman Acquisition**: 100-500 ms (depends on integration time)
- **Full Sweep (11 points)**: ~10-15 minutes

### Optimization Tips

- Reduce Raman averages (trade SNR for speed)
- Decrease integration time for faster scans
- Use batch DAQ operations where possible

---

## Safety Considerations

### Voltage Limits

- **Maximum AO Voltage**: ±10V (hardware enforced)
- **Software Limits**: Configurable via `safety.yaml`
- **Emergency Stop**: Automatically sets AO to 0V

### Laser Safety (Raman)

- Ocean Optics spectrometers use Class I lasers (safe)
- Do not disable safety interlocks
- Follow manufacturer's safety guidelines

### Data Integrity

- All measurements logged with timestamps
- Audit trail maintained per 21 CFR 11
- Electronic signatures required for GxP workflows

---

## Maintenance

### Daily

- Verify device connectivity
- Check for error logs
- Validate calibration standards

### Weekly

- Run full calibration check
- Review audit logs
- Update firmware if available

### Monthly

- Deep clean spectrometer optics
- Verify mechanical connections
- Test emergency stop procedures

---

## Support

For technical support:
- **NI DAQ**: https://www.ni.com/support
- **Ocean Optics**: https://www.oceaninsight.com/support
- **POLYMORPH_LITE**: See `docs/SUPPORT.md`
