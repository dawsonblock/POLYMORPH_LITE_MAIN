# Hardware Integration Guide

## Overview

POLYMORPH-LITE uses a **simulation-first** approach to hardware integration. All drivers work in simulation mode by default, allowing development and testing without physical hardware. Real hardware integration is optional and requires vendor SDKs.

## Simulation Mode

### Default Behavior
- All drivers start in simulation mode
- No vendor SDKs required
- Generates realistic synthetic data
- Safe for development and testing

### Configuration
Set in `.env`:
```bash
SIMULATION_MODE=true  # Default
```

## Supported Hardware

### ✅ Fully Supported (with Real Hardware)

#### Ocean Optics Spectrometers
- **Driver**: `retrofitkit/drivers/raman/vendor_ocean_optics.py`
- **SDK**: SeaBreeze (Python package)
- **Installation**:
  ```bash
  pip install seabreeze
  ```
- **Models**: USB2000, USB4000, QE65000, Flame series
- **Status**: Production-ready

#### National Instruments DAQ
- **Driver**: `retrofitkit/drivers/daq/ni.py`
- **SDK**: NI-DAQmx
- **Installation**:
  ```bash
  pip install nidaqmx
  ```
- **Models**: USB-6001, USB-6008, PCIe-6321, cDAQ series
- **Status**: Production-ready

### ⚠️ Partially Supported

#### Red Pitaya DAQ
- **Driver**: `retrofitkit/drivers/daq/redpitaya.py`
- **SDK**: Red Pitaya SCPI API
- **Installation**: Requires Red Pitaya firmware
- **Status**: Simulation mode only (SDK integration pending)

#### Gamry Potentiostat
- **Driver**: `retrofitkit/drivers/daq/vendor_gamry.py`
- **SDK**: Gamry COM Interface
- **Installation**: Requires Gamry Framework
- **Status**: Simulation mode only (SDK integration pending)

### 🔧 Simulation Only

#### Horiba Raman
- **Driver**: `retrofitkit/drivers/raman/vendor_horiba.py`
- **SDK**: Horiba SDK (proprietary)
- **Status**: Simulation mode only
- **Note**: SDK integration requires vendor license

#### Andor Cameras
- **Driver**: `retrofitkit/drivers/raman/vendor_andor.py`
- **SDK**: Andor SDK3
- **Status**: Simulation mode only
- **Note**: SDK integration requires vendor license

## Integrating Real Hardware

### Step 1: Install Vendor SDK

Follow vendor-specific instructions for your operating system.

**Example (Ocean Optics)**:
```bash
# Install SeaBreeze
pip install seabreeze

# Configure udev rules (Linux only)
seabreeze_os_setup
```

### Step 2: Update Configuration

Edit `.env`:
```bash
SIMULATION_MODE=false
```

### Step 3: Run Integration Tests

```bash
# Test specific hardware
pytest tests/integration/test_ocean_optics.py -v

# Test all hardware
pytest tests/ -m hardware -v
```

### Step 4: Validate in Lab

Perform lab validation per 21 CFR Part 11 requirements:
1. Installation Qualification (IQ)
2. Operational Qualification (OQ)
3. Performance Qualification (PQ)

See `docs/validation/` for templates.

## Driver Architecture

### Base Classes

All drivers inherit from:
- `HardwareDriver` - Base hardware interface
- `RamanDriver` - Raman spectrometer interface
- `DAQDriver` - Data acquisition interface

### Simulation Fallback

Drivers use try/except blocks for graceful fallback:

```python
try:
    import vendor_sdk
    SIMULATION_MODE = False
except ImportError:
    SIMULATION_MODE = True
    logger.warning("Vendor SDK not found, using simulation mode")
```

### Adding New Hardware

1. Create driver in `retrofitkit/drivers/`
2. Inherit from appropriate base class
3. Implement required methods
4. Add simulation mode support
5. Write integration tests
6. Document in this guide

## Troubleshooting

### SDK Not Found

**Symptom**: Driver falls back to simulation mode

**Solution**:
1. Verify SDK installation
2. Check Python path
3. Review driver logs

### Hardware Not Detected

**Symptom**: Driver initializes but no devices found

**Solution**:
1. Check USB connections
2. Verify device power
3. Check OS permissions (udev rules on Linux)
4. Review vendor documentation

### Permission Errors

**Symptom**: Access denied to hardware

**Solution (Linux)**:
```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Safety Considerations

### Hardware Interlocks

All drivers support safety interlocks:
- Temperature limits
- Pressure limits
- Emergency stop
- Watchdog timers

Configure in `config/hardware/safety.yaml`

### Validation Requirements

For GMP/GLP compliance:
1. Document all hardware
2. Perform IQ/OQ/PQ
3. Maintain calibration records
4. Track audit trail

## Support

### Community
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share experiences

### Commercial
Contact for commercial support, custom drivers, and validation assistance.

## License

Hardware drivers are MIT licensed. Vendor SDKs have separate licenses - consult vendor documentation.
