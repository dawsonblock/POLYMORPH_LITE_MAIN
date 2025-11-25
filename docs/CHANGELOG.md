# Changelog

All notable changes to POLYMORPH-4 Lite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Web-based recipe editor (planned)
- Mobile dashboard support (planned)
- Machine learning integration (planned)
- Cloud deployment options (planned)

## [1.0.0] - 2025-09-08

### Added - Initial Unified Release
- **Complete unified build** integrating all POLYMORPH-4 components
- **Retrofit Kit v4** as the core system foundation
- **Hardware Wizard v6** for automatic device detection and configuration
- **Config Overlays v6** with pre-built hardware configurations
- **Unified CLI** (`scripts/unified_cli.py`) for integrated system management
- **One-command installation** (`install.py --full-setup`)
- **Production deployment** configurations with Docker
- **Comprehensive documentation** and setup guides

### Hardware Support
- **National Instruments**: USB-6343, PCIe-6363, PXI-6733 DAQ devices
- **Red Pitaya**: Network-connected SCPI/TCP devices
- **Ocean Optics**: USB spectrometers via SeaBreeze library
- **Horiba**: Spectrometers via vendor SDK
- **Andor**: Camera systems via vendor SDK
- **Built-in simulators** for all hardware types

### Core Features
- **Real-time Raman spectroscopy** with conditional gating
- **Recipe-based automation** with YAML workflow definitions
- **Safety interlocks** (E-stop, door monitoring, watchdog timers)
- **21 CFR Part 11 compliance** (audit trails, e-signatures, RBAC)
- **FastAPI web server** with dashboard and RESTful API
- **WebSocket support** for real-time updates

### Configuration Management
- **Hardware profiles** for easy device swapping (`config/hardware_profiles/`)
- **Configuration overlays** for complete hardware setups (`config/overlays/`)
- **Interactive hardware wizard** for guided setup
- **Environment validation** and automated configuration

### Deployment & Operations
- **Multi-stage Docker builds** (development, production, hardware)
- **Production reverse proxy** with nginx and SSL/TLS support
- **Observability stack** (Prometheus metrics + Grafana dashboards)
- **Health checks** and automatic service restarts
- **Container orchestration** with docker-compose

### CLI Tools & Utilities
- **Unified CLI** with subcommands for all system functions
- **Interactive quickstart** wizard for first-time setup
- **Hardware detection** and listing capabilities
- **System status** monitoring and log viewing
- **Deployment scripts** for different environments

### Documentation
- **Comprehensive README** with quick start and detailed guides
- **Installation guide** (`docs/INSTALLATION.md`)
- **Quickstart guide** (`docs/QUICKSTART.md`)
- **Hardware wizard documentation** (`docs/README_wizard.md`)
- **Configuration overlay guide** (`docs/README_overlays.md`)
- **Validation templates** (IQ/OQ/PQ) for compliance
- **Complete build information** (`VERSION.md`)

### Security & Compliance
- **RSA-based electronic signatures** with key management
- **Append-only audit database** with cryptographic integrity
- **Role-based access control** (Operator, Engineer, QA, Admin)
- **Password policy enforcement** and session management
- **Two-person signoff** workflows for critical operations

### Monitoring & Observability
- **Prometheus metrics** collection and export
- **Pre-built Grafana dashboards** for system visualization
- **Real-time performance** monitoring
- **Alert integration** capabilities
- **Historical trend analysis** and reporting

## Component Version History

### Retrofit Kit Evolution
- **v4**: Latest with full vendor support, observability, hardware profiles
- **v3**: Added Andor/Horiba support, Prometheus metrics, Grafana dashboards
- **v2**: Added Ocean Optics support, validation templates (IQ/OQ/PQ)
- **v1**: Base implementation with NI/Red Pitaya DAQ, simulation modes

### Add-on Components
- **Hardware Wizard v6**: Automatic NI/Ocean detection, interactive configuration
- **Config Overlays v6**: Pre-built configs for NI+Ocean, NI+Horiba, RedPitaya+Andor

### Unified Build Features (New in v1.0)
- **Installation System**: Automated dependency management and setup
- **Unified CLI**: Single interface for all system functions
- **Production Deployment**: Docker, nginx, SSL/TLS, observability
- **Enhanced Documentation**: Complete guides for all scenarios

## Breaking Changes

None in initial release v1.0.0.

## Migration Guide

This is the initial unified release. Users of individual component packages should:

1. **Backup existing configurations** and data
2. **Extract unified build** to new directory
3. **Run migration wizard**: `python scripts/unified_cli.py quickstart`
4. **Apply appropriate overlay** matching your hardware setup
5. **Restore custom recipes** and configurations as needed

## Known Issues

- Hardware drivers (NI-DAQmx, SeaBreeze) may require manual installation on some systems
- Some vendor SDKs (Horiba, Andor) require separate licensing agreements
- Docker hardware access may require additional container privileges
- Windows hardware detection may need administrator privileges

## Security Advisories

None at this time. Security issues should be reported to the maintainers privately.

## Deprecated Features

None in initial release.

## Removed Features

None in initial release.

---

## Version Support

| Version | Status | Support Until | Python | Hardware |
|---------|--------|---------------|--------|----------|
| 1.0.x | Current | TBD | 3.11+ | Full |

## Release Schedule

- **Major releases**: Annually or for breaking changes
- **Minor releases**: Quarterly for new features
- **Patch releases**: Monthly or as needed for bug fixes
- **Security releases**: As needed

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to POLYMORPH-4 Lite.

## Support

For support and questions:
- **Issues**: https://github.com/dawsonblock/POLYMORPH_Lite/issues
- **Discussions**: https://github.com/dawsonblock/POLYMORPH_Lite/discussions
- **Documentation**: Complete guides in `docs/` directory