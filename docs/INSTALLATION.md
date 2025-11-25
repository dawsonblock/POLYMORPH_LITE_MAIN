# Polymorph-4 Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 10.15+
- **Python**: 3.11 or later
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Internet connection for initial setup

### Hardware Requirements (Optional)
- **NI DAQ**: USB-6343, PCIe-6363, PXI-6733, or compatible
- **Raman Spectrometers**: Ocean Optics (USB), Horiba, Andor cameras
- **Safety I/O**: Digital input/output channels for interlocks

## Installation Methods

### Method 1: Automated Installation (Recommended)

#### Full Interactive Setup
```bash
# One command setup with guidance
python install.py --full-setup
```

#### Command Line Installation
```bash
# Install with hardware support
python install.py --hardware --admin-email your@email.com --admin-name "Your Name"

# Install without hardware (simulation only)
python install.py --admin-email your@email.com --admin-name "Your Name"
```

### Method 2: Manual Installation

#### Step 1: Install Python Dependencies
```bash
# Core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Hardware dependencies (optional)
pip install -r requirements-hw.txt
```

#### Step 2: Initialize System Database
```bash
python -m retrofitkit.cli init --admin-email your@email.com --admin-name "Your Name" --set-admin-password
```

#### Step 3: Configure Hardware
```bash
# Option A: Automatic detection
python scripts/hardware_wizard.py

# Option B: Apply preset configuration
python scripts/apply_overlay.py NI_USB6343_Ocean0 .

# Option C: Manual configuration (edit config/config.yaml)
```

### Method 3: Docker Installation

#### Development Setup
```bash
# Basic setup
docker compose up --build

# With observability stack
docker compose -f docker/docker-compose.yml -f docker/docker-compose.observability.yml up --build
```

#### Production Setup
```bash
# Configure environment
cp docker/.env.example docker/.env
# Edit docker/.env with your settings

# Start production stack
docker compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d
```

## Hardware Driver Installation

### National Instruments (NI-DAQmx)

#### Windows
1. Download NI-DAQmx Runtime from ni.com
2. Install the runtime environment
3. Install Python support: `pip install nidaqmx`

#### Linux
```bash
# Install NI Linux Device Drivers
sudo apt-get install ni-daqmx
pip install nidaqmx
```

### Ocean Optics (SeaBreeze)

#### Windows/Mac/Linux
```bash
pip install seabreeze
```

#### Linux Additional Setup
```bash
# Install libusb
sudo apt-get install libusb-1.0-0-dev

# Set USB permissions (create udev rule)
sudo tee /etc/udev/rules.d/10-oceanoptics.rules << EOF
SUBSYSTEM=="usb", ATTR{idVendor}=="2457", MODE="0666", GROUP="users"
EOF

sudo udevadm control --reload-rules
```

### Horiba SDK
Contact Horiba for SDK installation packages and license requirements.

### Andor SDK
Contact Andor for SDK installation packages and license requirements.

## Configuration

### Hardware Configuration Files

The system uses layered configuration:

1. **Base Config**: `config/config.yaml` - Main system settings
2. **Hardware Profiles**: `config/hardware_profiles/*.yaml` - Device-specific settings
3. **Overlays**: `config/overlays/*/config.yaml` - Complete hardware combinations

### Environment Variables

Optional environment variables:
```bash
# Configuration file override
export P4_CONFIG=/path/to/custom/config.yaml

# Data directory override  
export P4_DATA_DIR=/path/to/data

# Log level
export P4_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Security Configuration

#### RSA Key Generation
Keys are automatically generated during initialization, or manually:
```bash
python scripts/keygen.py
```

#### Password Policy
Configure in `config/config.yaml`:
```yaml
security:
  password_policy:
    min_length: 12
    require_upper: true
    require_digit: true
    require_symbol: true
```

## Verification

### Test Installation
```bash
# Check system status
python scripts/unified_cli.py system status

# List detected hardware
python scripts/unified_cli.py hardware list

# Run system tests
python -m pytest tests/
```

### Test Hardware Connection
```bash
# Start server
python scripts/unified_cli.py server

# Navigate to http://localhost:8000
# Go to Hardware section and test connections
```

## Post-Installation Setup

### 1. User Management
- Create additional users through the web interface
- Assign appropriate roles (Operator, Engineer, QA, Admin)
- Configure two-person signoff if required

### 2. Recipe Development
- Review example recipes in `recipes/` directory
- Create custom recipes for your processes
- Test recipes in simulation mode first

### 3. Safety Configuration
- Configure E-stop and door interlock connections
- Test safety systems before production use
- Review safety documentation in `docs/safety_wiring.md`

### 4. Backup Configuration
- Set up regular database backups
- Configure log rotation
- Document your configuration for disaster recovery

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Missing dependencies
pip install -r requirements.txt -r requirements-hw.txt

# Permission issues (Linux/Mac)
chmod +x install.py scripts/*.py
```

#### Database Issues
```bash
# Reinitialize database
rm -f data/audit.db
python -m retrofitkit.cli init --admin-email your@email.com --admin-name "Your Name"
```

#### Hardware Detection Issues
```bash
# Check hardware status
python scripts/unified_cli.py hardware list

# Windows: Ensure drivers are installed
# Linux: Check USB permissions and udev rules
# All: Verify device connections and power
```

#### Port Conflicts
```bash
# Check for processes using port 8000
lsof -i :8000  # Linux/Mac
netstat -an | findstr :8000  # Windows

# Use different port
python scripts/unified_cli.py server --port 8080
```

### Getting Help

1. **Check Logs**: `python scripts/unified_cli.py system logs`
2. **System Status**: `python scripts/unified_cli.py system status`
3. **Documentation**: Review files in `docs/` directory
4. **Hardware Wizard**: Re-run `python scripts/unified_cli.py hardware wizard`

## Uninstallation

### Remove Application
```bash
# Stop all services
docker compose down  # If using Docker

# Remove virtual environment
rm -rf .venv  # or conda remove --name polymorph4 --all

# Remove data (optional)
rm -rf data/ logs/
```

### Remove Hardware Drivers
Follow vendor-specific uninstallation procedures for NI-DAQmx, SeaBreeze, etc.