# Hardware Setup Guide

## 1. Ocean Optics Spectrometers

**Supported Devices:** USB2000+, HR4000, QE65000, and most Ocean Insight USB spectrometers.

### Installation
The Python driver (`seabreeze`) is already installed.

**Linux/macOS Users:**
You must run the permission setup script once:
```bash
seabreeze_os_setup
```
*Note: This may require sudo permissions.*

### Verification
Run the following to verify detection:
```bash
python3 -c "import seabreeze.spectrometers as sb; print(sb.list_devices())"
```

---

## 2. National Instruments DAQ (NI-DAQmx)

**Supported Devices:** USB-6000 series, PCIe DAQ cards, CompactDAQ.

### Step 1: Install Runtime Drivers (Required)
The Python library requires the underlying C drivers to communicate with hardware.

1. Go to [NI.com/drivers](https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html)
2. Download **NI-DAQmx Runtime** (latest version)
3. Install on your host machine (Windows/Linux/macOS)
4. Restart your computer if prompted

### Step 2: Python Library
The Python wrapper is already installed:
```bash
pip install nidaqmx
```

### Verification
Run the following to verify detection:
```bash
python3 -c "import nidaqmx.system; print(nidaqmx.system.System.local().devices)"
```

---

## 3. Horiba / Andor Cameras (Advanced)

These are proprietary drivers that cannot be downloaded automatically.

1. Locate the **SDK/DLL files** provided on the USB drive that came with your instrument.
2. Copy the DLLs to: `retrofitkit/drivers/lib/`
3. The system will automatically detect them on startup.
