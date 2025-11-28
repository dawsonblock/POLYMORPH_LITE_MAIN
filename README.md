<div align="center">

![POLYMORPH-LITE](https://img.shields.io/badge/POLYMORPH--LITE-v3.0.0-blue?style=for-the-badge)

# 🔬 POLYMORPH-LITE

### AI-Powered Laboratory Operating System
**Enterprise-Grade | PostgreSQL | RBAC | 21 CFR Part 11–Aligned Architecture**

[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/dawsonblock/POLYMORPH_LITE_MAIN/releases)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Production](https://img.shields.io/badge/production-ready-brightgreen.svg)](#)

[🚀 Quick Start](#-quick-start) • [✨ Features](#-key-features) • [📖 Documentation](#-documentation) • [🏗️ Architecture](#-architecture) • [🤝 Contributing](#-contributing)

---

</div>

## 🎯 Overview

POLYMORPH-LITE transforms legacy analytical instruments into intelligent, AI-powered laboratory systems. Built for pharmaceutical R&D, quality control, and GMP environments with enterprise-grade database architecture and 21 CFR Part 11–aligned compliance features.

### 🆕 What's New in v3.0

<table>
<tr>
<td width="50%">

**💾 Unified Database**
- PostgreSQL with 27 normalized tables
- Alembic auto-migrations
- Eliminated split-brain architecture
- Production-grade data integrity

</td>
<td width="50%">

**🔐 Enhanced Security**
- Role-Based Access Control (RBAC)
- JWT authentication
- Cryptographic audit trail
- Password history & account locking

</td>
</tr>
<tr>
<td>

**🧪 LIMS Features**
- Sample tracking with lineage
- Inventory management
- Calibration logging
- Project/batch organization

</td>
<td>

**🚀 DevOps Ready**
- One-command Docker deployment
- Auto-migrations on startup
- Health monitoring scripts
- Database backup utilities

</td>
</tr>
</table>

### ⚙️ Backend Modernization (Pydantic v2 & Timezones)
- Pydantic models migrated to v2 (`ConfigDict`, `model_config`, `model_dump()`).
- Backend timestamps are now timezone-aware UTC (`datetime.now(timezone.utc)` via helpers).
- Device registry uses explicit keys for DAQ and Raman simulators to avoid name collisions.
- Hardware-dependent tests are marked with `pytest.mark.hardware` and are skipped by default unless explicitly enabled.

---

## ✨ Key Features

### 🗄️ Database & Data Management
- **27-Table Schema**: Comprehensive data model for lab operations
- **PostgreSQL-First**: Enterprise database with ACID compliance
- **Alembic Migrations**: Version-controlled schema evolution
- **Sample Lineage**: Parent-child tracking for traceability
- **Inventory Control**: Stock management with expiration tracking

### 3. Workflow Engine
- **Linear Execution**: Supports sequential execution of steps (Action, Wait, Hold).
- **Note**: As of v3.0.0, the workflow engine supports linear sequences only. Loops and conditional branching are planned for future releases.
- **Visual Builder**: Drag-and-drop interface for creating recipes.
- **Safety Integration**: Pre-execution safety checks and runtime monitoring.

### 🤖 AI Integration
- **BentoML Inference**: Optimized AI model serving
- **Circuit Breaker**: Resilient failure handling
- **Real-Time Analysis**: <50ms inference latency
- **Auto-Detection**: Crystallization event recognition

### 🔒 Compliance & Audit

**Architecture Designed for 21 CFR Part 11 Compliance**

POLYMORPH-LITE provides technical features that support 21 CFR Part 11 requirements:
- ✅ **Hash-Chain Audit Trail**: Immutable, cryptographically verified audit logs
- ✅ **Unique User Logins**: Individual accounts with MFA support
- ✅ **Role-Based Access Control**: Granular permissions (admin, scientist, technician, compliance)
- ✅ **Electronic Signatures**: RSA-based cryptographic signatures
- ✅ **Workflow Approval & Versioning**: Locked, approved workflow definitions  
- ✅ **Password Security**: History tracking, complexity requirements, account locking

> **Note on Compliance**: Final 21 CFR Part 11 compliance requires lab-specific validation 
> (IQ/OQ/PQ), written SOPs, organizational policies, and procedures. This software provides 
> the technical foundation to support compliant operations when properly validated and deployed  
> according to your organization's quality system.

### 🎨 Modern Interface
- **React 18**: Fast, responsive UI
- **Real-Time Updates**: WebSocket-based live data
- **Scientific Dark Mode**: Eye-friendly design
- **Glassmorphism**: Premium visual aesthetics

---

## 🚀 Quick Start

### One-Command Deployment

```bash
# Clone repository
git clone https://github.com/dawsonblock/POLYMORPH_LITE_MAIN.git
cd POLYMORPH_LITE_MAIN

# Configure environment
cp .env.example .env
# Edit .env: Set POSTGRES_PASSWORD, SECRET_KEY, JWT_SECRET_KEY

# Deploy (migrations run automatically!)
docker-compose up -d

# Create admin user
docker-compose exec backend python scripts/create_admin_user.py

# Access system
open http://localhost
```

**Default Login**: `admin@polymorph.local` / `admin123` ⚠️ Change immediately!

### Verification

```bash
# Check database health
docker-compose exec backend python scripts/check_db_health.py

# View logs
docker-compose logs -f backend

# API health check
curl http://localhost:8001/health
```

---

## 📊 Architecture

```mermaid
graph TD
    A[Frontend React] -->|REST API| B[FastAPI Backend]
    B -->|ORM| C[PostgreSQL 27 Tables]
    B -->|Inference| D[BentoML AI Service]
    B -->|Cache| E[Redis]
    B -->|Control| F[Hardware Drivers]
    F -->|USB/Serial| G[Lab Instruments]
    
    C -->|Migrations| H[Alembic]
    B -->|Audit| I[Hash-Chain Trail]
    B -->|Auth| J[RBAC + JWT]
```

### Database Schema (27 Tables)

| Domain | Tables | Purpose |
|--------|--------|---------|
| **Multi-Site** | organizations, labs, nodes, device_hubs | Enterprise deployment |
| **Auth & RBAC** | users, roles, user_roles | Authentication & permissions |
| **Audit** | audit | Immutable audit trail |
| **Devices** | devices, device_status | Instrument management |
| **LIMS** | projects, batches, containers, samples, sample_lineage | Sample tracking |
| **Inventory** | vendors, inventory_items, stock_lots | Stock management |
| **Calibration** | calibration_entries | Instrument calibration |
| **Workflows** | workflow_versions, executions, assignments, config_snapshots | Workflow management |

---

## 🔐 RBAC & Permissions

| Role | Create | Read | Update | Delete | Use Case |
|------|--------|------|--------|--------|----------|
| **admin** | ✅ All | ✅ All | ✅ All | ✅ All | System administrators |
| **scientist** | ✅ Workflows, Samples | ✅ All | ✅ Own data | ❌ | Research scientists |
| **technician** | ✅ Runs, Inventory | ✅ Workflows, Samples | ✅ Inventory | ❌ | Lab technicians |
| **compliance** | ❌ | ✅ Audit, Runs | ❌ | ❌ | QA/QC officers |

---

## 📝 API Examples

### Authentication
```bash
curl -X POST http://localhost:8001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@polymorph.local","password":"admin123"}'
```

### Create Sample (LIMS)
```bash
curl -X POST http://localhost:8001/api/samples \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "sample_id": "SAMPLE-001",
    "status": "active",
    "extra_data": {"concentration": "10mg/mL"}
  }'
```

### Check Low Stock (Inventory)
```bash
curl http://localhost:8001/api/inventory/low-stock \
  -H "Authorization: Bearer $TOKEN"
```

---

## 🔧 Production Tools

### Database Management
```bash
# Health check
python scripts/check_db_health.py

# Backup database
python scripts/backup_database.py

# Create admin user
python scripts/create_admin_user.py --email admin@lab.com

# Generate cryptographic keys
python scripts/generate_keys.py
```

### Monitoring
- **Health Endpoint**: `/health`
- **API Documentation**: `/docs` (Swagger UI)
- **Metrics**: Prometheus-compatible (optional)

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - overview and quick start |
| [RELEASE_NOTES.md](RELEASE_NOTES.md) | Version history and breaking changes |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Production deployment instructions |
| [STRUCTURE.md](STRUCTURE.md) | Repository organization guide |
| [SECURITY.md](SECURITY.md) | Security policies and reporting |
| [TESTING.md](TESTING.md) | Testing guide and coverage |
| [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) | Complete API reference |
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | End-user guide |

---

## 🏗️ Technology Stack

### Backend
- **FastAPI** - High-performance async API framework
- **PostgreSQL 15** - Enterprise-grade database
- **SQLAlchemy 2.0** - Modern ORM with type hints
- **Alembic** - Database migration management
- **Pydantic v2** - Data validation
- **BentoML** - AI model serving

### Frontend
- **React 18** - UI library
- **Vite** - Fast build tool
- **TypeScript** - Type-safe JavaScript
- **Socket.IO** - Real-time communication

### DevOps
- **Docker & Docker Compose** - Containerization
- **Nginx** - Reverse proxy
- **Redis** - Caching and pub/sub

---

## 🧪 Supported Hardware

- **NI DAQ** - National Instruments data acquisition
- **Ocean Optics** - Spectrometers
- **Horiba Raman** - Raman spectroscopy
- **Red Pitaya** - Signal processing
- **Custom Devices** - Modular driver architecture

---

## 📦 Requirements

### Minimum System Requirements
- **OS**: Linux, macOS, Windows (WSL2)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB available space
- **Docker**: 20.10+
- **Docker Compose**: 2.0+

### For Development
- **Python**: 3.11+
- **Node.js**: 18+
- **PostgreSQL**: 15+ (if not using Docker)

---

## 🛣️ Roadmap

### ✅ v3.0 (Current)
- Unified PostgreSQL database
- Complete LIMS functionality
- RBAC with 4 roles
- Production deployment tools

### 🔄 v3.1 (Next)
- [ ] Frontend LIMS UI pages
- [ ] Workflow builder → orchestrator integration
- [ ] Advanced analytics dashboards
- [ ] Multi-tenant organization features

### 🔮 Future
- [ ] SSO integration (SAML, OAuth)
- [ ] Mobile app for lab monitoring
- [ ] Advanced AI models
- [ ] Cloud deployment guides (AWS, Azure, GCP)

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Clone repository
git clone https://github.com/dawsonblock/POLYMORPH_LITE_MAIN.git
cd POLYMORPH_LITE_MAIN

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start PostgreSQL
docker-compose up -d postgres

# Run migrations
alembic upgrade head

# Start development server
uvicorn retrofitkit.api.server:app --reload
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with ❤️ for the pharmaceutical and research community
- Powered by FastAPI, React, PostgreSQL, and BentoML
- Inspired by real-world lab automation challenges

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/dawsonblock/POLYMORPH_LITE_MAIN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dawsonblock/POLYMORPH_LITE_MAIN/discussions)
- **Security**: See [SECURITY.md](SECURITY.md)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

[![GitHub stars](https://img.shields.io/github/stars/dawsonblock/POLYMORPH_LITE_MAIN?style=social)](https://github.com/dawsonblock/POLYMORPH_LITE_MAIN)
[![GitHub forks](https://img.shields.io/github/forks/dawsonblock/POLYMORPH_LITE_MAIN?style=social)](https://github.com/dawsonblock/POLYMORPH_LITE_MAIN)

Made with 💙 for better labs everywhere

[⬆ Back to Top](#-polymorph-lite)

</div>