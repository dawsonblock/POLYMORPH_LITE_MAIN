# Polymorph-4 Unified Build Information

**Build Version**: Unified v1.0  
**Build Date**: 2025-09-08  
**Build Type**: Complete Unified Package

## Component Versions Included

| Component | Version | Size | Description |
|-----------|---------|------|-------------|
| **Retrofit Kit** | v4 | 36KB | Core system with all vendor support |
| **Config Overlays** | v6 | 2.3KB | Pre-built hardware configurations |
| **Hardware Wizard** | v6 | 1.9KB | Automatic hardware detection |
| **Unified CLI** | v1.0 | New | Integrated command interface |
| **Installation Scripts** | v1.0 | New | Automated setup and deployment |

## What's New in Unified Build

### 🔧 Unified Installation System
- **One-command setup**: `python install.py --full-setup`
- **Interactive wizards** for hardware and configuration
- **Automatic dependency management** including hardware drivers
- **Environment detection** and validation

### 🎯 Integrated CLI Tool  
- **Single interface** for all system functions (`scripts/unified_cli.py`)
- **Hardware management**: Detection, configuration, profiling
- **Config management**: Overlay application, listing, validation
- **System management**: Status, logs, initialization
- **Interactive quickstart** wizard

### 🚀 Enhanced Deployment
- **Multi-stage Docker builds** (development, production, hardware)
- **Production deployment scripts** with nginx reverse proxy
- **Observability integration** (Prometheus + Grafana)
- **Environment configuration** templates
- **SSL/TLS support** for production

### 📚 Comprehensive Documentation
- **Quickstart guide** for immediate setup
- **Detailed installation** instructions
- **Hardware configuration** guides
- **Deployment documentation** for all environments
- **Troubleshooting** and support information

## Hardware Support Matrix

| DAQ System | Raman Spectrometer | Configuration Overlay | Status |
|------------|-------------------|---------------------|--------|
| NI USB-6343 | Ocean Optics #0 | `NI_USB6343_Ocean0` | ✅ Tested |
| NI PCIe-6363 | Horiba | `NI_PCIE6363_Horiba` | ✅ Tested |
| Red Pitaya | Andor Camera | `RedPitaya_Andor` | ✅ Tested |
| NI USB-6343 | Simulator | `NI_USB6343_Simulator` | ✅ Tested |

### Supported Hardware
- **DAQ**: NI USB-6343, PCIe-6363, PXI-6733, Red Pitaya
- **Raman**: Ocean Optics (SeaBreeze), Horiba SDK, Andor SDK
- **Safety**: Digital I/O for E-stop, door interlocks, watchdog

## Software Dependencies

### Core Requirements
```
Python 3.11+
FastAPI 0.115.0
Uvicorn 0.30.6  
SQLAlchemy 2.0.32
NumPy 2.0.2
Pandas 2.2.2
```

### Hardware Drivers (Optional)
```
nidaqmx (NI-DAQmx support)
seabreeze (Ocean Optics support)
```

### Development Tools
```
typer (CLI framework)
rich (enhanced terminal output)
pytest (testing framework)
```

## Installation Options

### Quick Start
```bash
# Complete interactive setup
python install.py --full-setup

# Or use CLI wizard
python scripts/unified_cli.py quickstart
```

### Manual Setup
```bash
# Step by step
python install.py --hardware
python scripts/unified_cli.py hardware wizard  
python scripts/unified_cli.py config overlay NI_USB6343_Ocean0
python scripts/unified_cli.py server
```

### Docker Deployment
```bash
# Development
python scripts/deploy.py dev

# Production with observability
python scripts/deploy.py prod

# Hardware-enabled container
python scripts/deploy.py hardware
```

## File Structure

```
polymorph4_unified/
├── 📁 retrofitkit/          # Core application (from v4)
│   ├── api/                 # FastAPI server & web UI
│   ├── core/                # Recipe engine & orchestration  
│   ├── drivers/             # Hardware drivers (DAQ + Raman)
│   ├── safety/              # Safety interlocks & monitoring
│   ├── compliance/          # Audit trails & e-signatures
│   └── data/               # Database models & storage
├── 📁 config/               # Configuration management
│   ├── overlays/           # Pre-built hardware configs (from v6)
│   └── hardware_profiles/  # Device-specific profiles (from v4)
├── 📁 scripts/              # Utility scripts & unified CLI
│   ├── unified_cli.py      # 🆕 Integrated command interface
│   ├── hardware_wizard.py  # Hardware detection (from v6)
│   ├── apply_overlay.py    # Config overlay tool (from v6)
│   └── deploy.py           # 🆕 Production deployment
├── 📁 docker/               # Container orchestration
│   ├── docker-compose.yml         # Base services (from v4)
│   ├── docker-compose.prod.yml    # 🆕 Production setup
│   ├── docker-compose.observability.yml  # Monitoring (from v4)
│   ├── nginx/              # 🆕 Reverse proxy config
│   └── .env.example        # 🆕 Environment template
├── 📁 docs/                 # Documentation
│   ├── QUICKSTART.md       # 🆕 Quick setup guide
│   ├── INSTALLATION.md     # 🆕 Detailed install guide
│   ├── README_wizard.md    # Hardware wizard guide (from v6)
│   ├── README_overlays.md  # Config overlay guide (from v6)
│   └── validation/         # IQ/OQ/PQ templates (from v4)
├── 📄 install.py            # 🆕 Unified installation script
├── 📄 README.md            # 🆕 Comprehensive system overview
├── 📄 Dockerfile.multi     # 🆕 Multi-stage container builds
└── 📄 VERSION.md           # 🆕 This file
```

## Key Features

### 🔬 Scientific Capabilities
- **Real-time Raman spectroscopy** with gating conditions
- **Multi-vendor hardware** support (NI, Ocean, Horiba, Andor)
- **Recipe-based automation** with YAML workflows
- **Safety interlocks** and monitoring systems

### 🛡️ Compliance & Security
- **21 CFR Part 11 style** audit trails and e-signatures
- **Role-based access control** with password policies
- **Cryptographic integrity** with RSA key verification
- **Immutable records** in append-only database

### 🚀 Deployment & Operations
- **Docker containerization** for all environments
- **Observability stack** (Prometheus + Grafana)
- **Production-ready** reverse proxy and SSL
- **Automated deployment** and configuration

### 🔧 Integration & Extensibility
- **Plugin architecture** for new hardware vendors
- **RESTful API** with OpenAPI documentation
- **WebSocket support** for real-time updates
- **Modular design** for easy customization

## Upgrade Path

This unified build represents the culmination of all individual packages:
- **From v1-v3**: Full compatibility, additional features
- **From individual packages**: All features integrated  
- **Future updates**: Will maintain backward compatibility

## Support & Documentation

- **📚 Documentation**: Complete guides in `docs/` directory
- **🔧 CLI Help**: `python scripts/unified_cli.py --help`
- **📊 System Status**: `python scripts/unified_cli.py system status`
- **📝 Logs**: `python scripts/unified_cli.py system logs`

## License & Compliance Note

**License**: See `LICENSE` file  
**Compliance**: Includes 21 CFR Part 11-style controls. Certification and validation are the operator's responsibility.

---

**Ready to start?** Run `python scripts/unified_cli.py quickstart` for interactive setup!