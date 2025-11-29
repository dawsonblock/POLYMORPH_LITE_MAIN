# Hardware Driver Status

## Overview

This document provides the current implementation status of all hardware drivers in POLYMORPH-LITE, including production-ready drivers, partial implementations, and simulation-only drivers.

---

## Production-Ready Drivers ✅

### DAQ (Data Acquisition)

#### NI DAQ (`retrofitkit/drivers/daq/ni.py`)
**Status**: ✅ PRODUCTION-READY

**Features**:
- Full analog I/O support (AI/AO)
- Full digital I/O support (DI/DO)
- nidaqmx SDK integration
- Simulation fallback when SDK unavailable
- Watchdog timer support

**Supported Hardware**:
- NI USB-6343
- NI PCIe-6363
- NI PXI-6733
- All nidaqmx-compatible devices

**Tier-1 Recommendation**: ✅ **YES**

**Example**:
```python
from retrofitkit.drivers.daq.ni import NIDAQ

daq = NIDAQ(cfg=config)
await daq.connect()

# Analog operations
await daq.set_voltage(1.5)
voltage = await daq.read_ai()

# Digital operations
await daq.write_do(line=0, on=True)
state = await daq.read_di(line=0)
```

---

### Raman Spectroscopy

#### Ocean Optics (`retrofitkit/drivers/raman/vendor_ocean_optics.py`)
**Status**: ✅ PRODUCTION-READY

**Features**:
- seabreeze SDK integration
- All USB spectrometers supported
- Dark correction
- Nonlinearity correction
- Simulation fallback
- Returns unified `Spectrum` data model

**Supported Hardware**:
- USB2000
- HR4000
- QE65000
- All seabreeze-compatible models

**Tier-1 Recommendation**: ✅ **YES**

**Example**:
```python
from retrofitkit.drivers.raman.vendor_ocean_optics import OceanOpticsSpectrometer

raman = OceanOpticsSpectrometer(device_index=0, integration_time_ms=20.0)
await raman.connect()

spectrum = await raman.acquire_spectrum(integration_time_ms=50.0)
# spectrum.wavelengths, spectrum.intensities, spectrum.meta
```

---

## Partial/Development Drivers ⚠️

### DAQ (Data Acquisition)

#### Red Pitaya (`retrofitkit/drivers/daq/redpitaya.py`)
**Status**: ⚠️ **ANALOG OUTPUT ONLY**

**Supported Features**:
- ✅ Analog Output (AO): PRODUCTION-READY via SCPI
- ❌ Analog Input (AI): NOT IMPLEMENTED
- ❌ Digital Input (DI): NOT IMPLEMENTED
- ❌ Digital Output (DO): NOT IMPLEMENTED

**Use Case**: Analog output control only

**Deployment Recommendation**:
- Use Red Pitaya ONLY for AO control
- Use NI DAQ for all AI/DI/DO requirements

**Example**:
```python
from retrofitkit.drivers.daq.redpitaya import RedPitayaDAQ

daq = RedPitayaDAQ(config)
await daq.connect()

# This works ✅
await daq.write_ao(channel=0, value=1.5)

# These raise RuntimeError ❌
# await daq.read_ai(channel=0)
# await daq.read_di(line=0)
# await daq.write_do(line=0, state=True)
```

**To Enable Full Support**:
1. Implement SCPI commands for AI/DI/DO
2. See Red Pitaya SCPI documentation: https://redpitaya.readthedocs.io/
3. Update tests

---

### Raman Spectroscopy

#### Horiba (`retrofitkit/drivers/raman/vendor_horiba.py`)
**Status**: ⚠️ **SIMULATION ONLY**

**Features**:
- ✅ Simulation mode: Returns synthetic spectra
- ❌ Real SDK integration: NOT IMPLEMENTED

**Use Case**: Workflow testing and development only

**Deployment Recommendation**:
- Use Ocean Optics for production Raman spectroscopy
- Use Horiba driver for testing workflows without hardware

**Example**:
```python
from retrofitkit.drivers.raman.vendor_horiba import HoribaRaman

raman = HoribaRaman(config)

# Returns simulated spectrum
spectrum = await raman.acquire_spectrum(
    integration_time_ms=100.0,
    averages=1,
    center_wavelength_nm=532.0
)
```

**To Enable Real Hardware**:
1. Install `horiba_sdk` package
2. Implement `_acquire_real()` method
3. Add SDK initialization and configuration
4. Update tests

---

#### Andor (`retrofitkit/drivers/raman/vendor_andor.py`)
**Status**: ⚠️ **SIMULATION ONLY**

**Features**:
- ✅ Simulation mode: Clean synthetic spectra
- ❌ Real SDK integration: NOT IMPLEMENTED

**Use Case**: Workflow testing and development only

**Deployment Recommendation**:
- Use Ocean Optics for production Raman spectroscopy
- Use Andor driver for testing workflows without hardware

**Example**:
```python
from retrofitkit.drivers/raman.vendor_andor import AndorRaman

raman = AndorRaman(config)

# Returns simulated spectrum
spectrum = await raman.acquire_spectrum(
    integration_time_ms=100.0,
    averages=1,
    center_wavelength_nm=550.0
)
```

**To Enable Real Hardware**:
1. Install Andor SDK
2. Implement `_acquire_real()` method with SDK calls
3. Update tests

---

## Tier-1 Stack Recommendation

### Production Configuration ✅

For production deployments, use the Tier-1 hardware stack:

**DAQ**: NI USB-6343 or PCIe-6363
- Full AI/AO/DI/DO support
- Production-validated
- Excellent reliability

**Raman**: Ocean Optics (any seabreeze-compatible model)
- Production-validated
- Wide hardware support
- Excellent SDK

**Configuration Overlay**: `config/overlays/NI_USB6343_Ocean0/`

**Example Deployment**:
```bash
# Load Tier-1 configuration
export POLYMORPH_OVERLAY=NI_USB6343_Ocean0

# Start system
python main.py
```

---

## Driver Development Guidelines

### Adding a New Driver

1. **Inherit from appropriate base class**:
   - DAQ: `DAQDevice` or `ProductionHardwareDriver`
   - Raman: `SpectrometerDevice` or `ProductionHardwareDriver`

2. **Implement required methods**:
   - `connect()`: Initialize hardware
   - `disconnect()`: Clean shutdown
   - `health()`: Return status dict
   - Device-specific methods (e.g., `acquire_spectrum()`)

3. **Register with DeviceRegistry**:
   ```python
   from retrofitkit.core.registry import registry
   registry.register("driver_name", DriverClass)
   ```

4. **Add capabilities**:
   ```python
   capabilities = DeviceCapabilities(
       kind=DeviceKind.SPECTROMETER,
       vendor="Vendor Name",
       model="Model Name",
       actions=["acquire_spectrum"],
       features={"key": "value"}
   )
   ```

5. **Write tests**:
   - Unit tests for driver methods
   - Integration tests for end-to-end workflows
   - Mock hardware responses

6. **Document limitations**:
   - Add comprehensive module docstring
   - Explain what works and what doesn't
   - Provide clear error messages
   - Include usage examples

---

## Testing Hardware Drivers

### Unit Tests
```bash
# Test specific driver
pytest tests/test_ni_daq.py -v

# Test all DAQ drivers
pytest tests/ -k "daq" -v

# Test all Raman drivers
pytest tests/ -k "raman" -v
```

### Integration Tests
```bash
# Test Tier-1 stack
pytest tests/test_tier1_integration.py -v -m tier1

# Test with real hardware (requires hardware connected)
pytest tests/test_tier1_integration.py -v -m tier1 --hardware
```

---

## Troubleshooting

### NI DAQ Issues

**Problem**: `nidaqmx` import fails
**Solution**: Install NI-DAQmx driver and Python package:
```bash
# Install NI-DAQmx driver from ni.com
# Then install Python package
pip install nidaqmx
```

**Problem**: Device not found
**Solution**: Check device name in NI MAX (Measurement & Automation Explorer)

---

### Ocean Optics Issues

**Problem**: `seabreeze` import fails
**Solution**: Install seabreeze:
```bash
pip install seabreeze
```

**Problem**: No devices found
**Solution**: 
1. Check USB connection
2. Verify device permissions (Linux)
3. Check device index (try 0, 1, 2...)

---

### Red Pitaya Issues

**Problem**: Connection timeout
**Solution**:
1. Verify Red Pitaya IP address
2. Check SCPI server is running on Red Pitaya
3. Verify network connectivity: `ping <red_pitaya_ip>`

**Problem**: `RuntimeError` on `read_ai()`
**Solution**: This is expected - Red Pitaya driver only supports AO. Use NI DAQ for AI.

---

## Summary

| Driver | Status | Production Ready | Tier-1 |
|--------|--------|------------------|--------|
| NI DAQ | ✅ Complete | ✅ Yes | ✅ Yes |
| Ocean Optics | ✅ Complete | ✅ Yes | ✅ Yes |
| Red Pitaya | ⚠️ AO Only | ⚠️ Partial | ❌ No |
| Horiba | ⚠️ Simulation | ❌ No | ❌ No |
| Andor | ⚠️ Simulation | ❌ No | ❌ No |

**Tier-1 Stack**: NI DAQ + Ocean Optics ✅
