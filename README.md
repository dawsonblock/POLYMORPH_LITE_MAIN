<div align="center">

# 🔬 POLYMORPH-4 Lite v2.0.0

### AI-Powered Laboratory Automation Platform
### Real-time Polymorph Detection | 21 CFR Part 11 Compliance | Production Ready

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/dawsonblock/POLYMORPH_LITE_MAIN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![Production Ready](https://img.shields.io/badge/production-ready-brightgreen.svg)](#)

**[Quick Start](#-quick-start)** • **[Features](#-key-features)** • **[Documentation](docs/)** • **[Deploy](#-deployment)**

---

</div>

## 🎯 What is POLYMORPH-4 Lite?

POLYMORPH-4 Lite is a **production-ready laboratory automation platform** that transforms existing analytical instruments into intelligent, AI-powered systems. Built for pharmaceutical R&D, quality control, and production environments.

### 💡 Value Proposition

- **🚀 Deploy in Minutes**: One-command Docker deployment
- **🤖 AI-Powered**: Real-time polymorph detection with deep learning
- **🔒 Compliance Ready**: Built-in 21 CFR Part 11 features (audit trails, e-signatures)
- **🎨 Premium UI**: Modern glassmorphism design with real-time monitoring
- **📊 Observable**: Prometheus metrics + Grafana dashboards
- **🔧 Multi-Vendor**: Supports NI, Ocean Optics, Horiba, Red Pitaya, and more

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🤖 AI Integration
- **BentoML Service**: Optimized AI inference
- **Circuit Breaker**: Resilient failure handling
- **Real-time Analysis**: <50ms inference latency
- **Auto-detection**: Crystallization events

</td>
<td width="50%">

### 🎨 Modern Interface
- **Scientific Dark Mode**: Eye-friendly design
- **Glassmorphism UI**: Premium aesthetics
- **Real-time Dashboard**: Live spectral data
- **WebSocket Updates**: Instant notifications

</td>
</tr>
<tr>
<td>

### 🔒 Compliance & Security
- **21 CFR Part 11**: Full compliance
- **Electronic Signatures**: RSA-based
- **Audit Trails**: Immutable logs
- **RBAC**: Role-based access control
- **MFA**: Multi-factor authentication

</td>
<td>

### 📊 Monitoring & Observability
- **Health Checks**: Component monitoring
- **Prometheus Metrics**: System performance
- **Grafana Dashboards**: Visual analytics
- **Alert System**: Email notifications

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend development)

### ⚡ One-Command Deploy

```bash
# Clone repository
git clone https://github.com/dawsonblock/POLYMORPH_LITE_MAIN.git
cd POLYMORPH_LITE_MAIN

# Deploy with Docker
./deploy.sh
```

**That's it!** Access the application:
- 🌐 **Frontend**: http://localhost
- 📚 **API Docs**: http://localhost:8001/docs
- 📊 **Metrics**: http://localhost:9090
- 📈 **Grafana**: http://localhost:3030

### 🎬 What Happens
1. Builds optimized frontend (React + Vite)
2. Starts Redis for state persistence
3. Launches FastAPI backend with health checks
4. Starts AI service (BentoML)
5. Deploys NGINX reverse proxy
6. Verifies all services are healthy

---

## 📦 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    POLYMORPH-4 Lite v2.0                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────┐   ┌───────────┐   ┌─────────┐   ┌─────────┐ │
│  │  Frontend │───│  Backend  │───│   AI    │───│  Redis  │ │
│  │  (React)  │   │ (FastAPI) │   │(BentoML)│   │ (Cache) │ │
│  └───────────┘   └───────────┘   └─────────┘   └─────────┘ │
│       │               │                │                      │
│  ┌────▼───────────────▼────────────────▼──────────────────┐ │
│  │         Hardware Abstraction Layer                      │ │
│  │  ┌──────┐  ┌──────┐  ┌───────┐  ┌────────┐           │ │
│  │  │  NI  │  │Ocean │  │Horiba │  │  Red   │           │ │
│  │  │ DAQ  │  │Optics│  │ Raman │  │Pitaya  │           │ │
│  │  └──────┘  └──────┘  └───────┘  └────────┘           │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | React 18 + Vite + TailwindCSS | Modern, responsive UI |
| **Backend** | FastAPI + Python 3.11 | High-performance API |
| **AI Service** | BentoML + PyTorch | ML model serving |
| **Database** | Redis + SQLite | State & persistence |
| **Monitoring** | Prometheus + Grafana | Metrics & dashboards |
| **Deployment** | Docker + NGINX | Containerization |

---

## 📸 Screenshots

### 🎨 Modern Dashboard
> Real-time system overview with glassmorphism design, live spectral data, and component health monitoring

### 📊 Spectral Analysis  
> AI-powered real-time Raman spectroscopy with automatic peak detection and polymorph classification

### 🔐 Compliance Features
> Electronic signatures, audit trail viewer, and role-based access control

### 📈 Monitoring Dashboards
> Grafana dashboards showing system metrics, experiment history, and performance analytics

---

## 🎯 Use Cases

### 🔬 Research & Development
- Automated crystallization monitoring
- Real-time polymorph screening
- Data integrity for publications
- Rapid prototyping with simulation modes

### 🏭 Quality Control
- Batch release testing with e-records
- Automated pass/fail criteria
- Multi-user approval workflows
- Compliance documentation

### ⚗️ Process Development
- Scale-up monitoring
- Process optimization (DOE)
- Critical parameter tracking
- Technology transfer docs

### 🏥 Production
- LIMS/MES integration via REST API
- Real-time release testing
- Regulatory compliance (FDA/EMA/ICH)
- Retrofit existing instruments

---

## 🛠️ Installation Options

### Option 1: Docker (Recommended)
```bash
./deploy.sh  # One command!
```

### Option 2: Development Setup
```bash
# Backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m retrofitkit.main

# Frontend
cd gui-v2/frontend
npm install
npm run dev
```

### Option 3: Manual Docker Compose
```bash
cp .env.production.example .env
# Edit .env with your configuration
docker-compose up -d
```

---

## 📚 Documentation

Comprehensive documentation located in `/docs`:

| Document | Description |
|----------|-------------|
| **[User Manual](docs/USER_MANUAL.md)** | Complete guide for lab operators |
| **[API Documentation](docs/API_DOCUMENTATION.md)** | RESTful API reference + SDK examples |
| **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** | Production deployment instructions |
| **[Quick Start](docs/QUICKSTART.md)** | Get started in 5 minutes |
| **[Quick Deploy](DEPLOY_NOW.md)** | Single-page deployment reference |

---

## 🔧 Hardware Support

### Data Acquisition (DAQ)
- ✅ National Instruments (USB-6343, PCIe-6363, PXI-6733)
- ✅ Red Pitaya (network-connected SCPI/TCP)
- ✅ Gamry Potentiostats
- ✅ Software Simulator (no hardware required)

### Raman Spectrometers
- ✅ Ocean Optics (USB via SeaBreeze)
- ✅ Horiba (vendor SDK)
- ✅ Andor (camera-based systems)
- ✅ Software Simulator

### Safety I/O
- ✅ E-stop integration
- ✅ Door interlocks
- ✅ Watchdog timers
- ✅ Configurable failsafe logic

---

## 🚀 Deployment

### Production Checklist

- [ ] Copy `.env.production.example` to `.env`
- [ ] Update `SECRET_KEY` (generate with: `python -c 'import secrets; print(secrets.token_urlsafe(32))'`)
- [ ] Set `REDIS_PASSWORD`
- [ ] Configure SSL certificates
- [ ] Update `CORS_ORIGINS`
- [ ] Set up backup strategy
- [ ] Configure email alerts

### Deploy
```bash
./deploy.sh
```

### Verify
```bash
curl http://localhost:8001/health
curl http://localhost:3000/healthz
```

### Manage
```bash
docker-compose logs -f        # View logs
docker-compose ps             # Check status
docker-compose restart        # Restart services
docker-compose down           # Stop all
```

---

## 📊 Metrics & Monitoring

### Built-in Metrics
- System performance (CPU, memory, disk)
- Experiment execution stats
- AI inference latency
- Hardware status
- User activity logs

### Grafana Dashboards
Access at `http://localhost:3030` (admin/admin):
- System Health Overview
- Experiment History
- AI Performance Metrics
- Hardware Status

---

## 🔐 Security & Compliance

### 21 CFR Part 11 Features
- ✅ **Audit Trails**: Cryptographically secured, append-only
- ✅ **Electronic Signatures**: RSA-based, legally binding
- ✅ **Access Control**: Role-based (Admin, Operator, QA, Guest)
- ✅ **Password Policies**: Enforced complexity & expiration
- ✅ **Two-Person Rule**: Configurable approval workflows

### Security Best Practices
- ✅ JWT authentication with expiration
- ✅ MFA support (TOTP)
- ✅ Rate limiting
- ✅ CORS protection
- ✅ SQL injection prevention
- ✅ XSS protection

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint/Prettier for JavaScript/TypeScript
- Write tests for new features
- Update documentation

---

## 🐛 Troubleshooting

### Common Issues

**Services won't start**
```bash
docker-compose logs backend    # Check backend logs
docker-compose down -v          # Reset and try again
```

**Health check fails**
```bash
curl -v http://localhost:8001/health  # Detailed health info
```

**Frontend build errors**
```bash
cd gui-v2/frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

For more help, see [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) or open an [issue](https://github.com/dawsonblock/POLYMORPH_LITE_MAIN/issues).

---

## 📋 System Requirements

### Minimum
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 100 GB SSD
- **OS**: Ubuntu 20.04+, macOS 12+, Windows 11

### Recommended
- **CPU**: 8+ cores
- **RAM**: 16 GB
- **Storage**: 500 GB NVMe SSD
- **GPU**: NVIDIA GPU for AI acceleration (optional)

---

## 📝 Version History

### v2.0.0 (2025-11-25) - Production Ready
- ✅ Complete GUI modernization (scientific dark mode + glassmorphism)
- ✅ AI service integration with circuit breaker pattern
- ✅ Comprehensive documentation (148KB)
- ✅ Docker Compose infrastructure
- ✅ Health monitoring & metrics
- ✅ Production deployment automation
- ✅ 21 CFR Part 11 compliance features

### v1.0 - Initial Release
- Core automation platform
- Multi-vendor hardware support
- Recipe-based workflows

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Compliance Notice
> ⚠️ **Important**: This software includes 21 CFR Part 11-style controls (audit trails, e-signatures, authority checks). **Validation and certification are the operator's responsibility.** The system provides the mechanisms; formal qualification (IQ/OQ/PQ) must be performed by your organization.

---

## 🏆 Acknowledgments

- **FastAPI** community for excellent web framework
- **React** team for modern UI library
- **BentoML** for ML serving infrastructure
- **Scientific Python** ecosystem (NumPy, SciPy, PyTorch)
- **Hardware vendors** (NI, Ocean Optics, Horiba) for SDK support
- **Open source community** for tools and libraries

---

## 📞 Support & Contact

### Enterprise Support
For production deployments, validation assistance, and custom integrations:
- 📧 Email: support@polymorph4.com
- 🌐 Website: https://polymorph4.com

### Community Support
- 💬 **GitHub Issues**: [Report bugs](https://github.com/dawsonblock/POLYMORPH_LITE_MAIN/issues)
- 📚 **Documentation**: Complete guides in `/docs`
- 🔧 **Deployment Help**: See `DEPLOY_NOW.md`

---

<div align="center">

### 🎯 Ready to Transform Your Laboratory?

```bash
git clone https://github.com/dawsonblock/POLYMORPH_LITE_MAIN.git
cd POLYMORPH_LITE_MAIN
./deploy.sh
```

**⭐ Star this repo** if POLYMORPH-4 Lite helps your research!

---

**Built with ❤️ for the scientific community** | **© 2025 POLYMORPH-4 Research Team**

[Documentation](docs/) • [Quick Start](#-quick-start) • [GitHub](https://github.com/dawsonblock/POLYMORPH_LITE_MAIN)

</div>