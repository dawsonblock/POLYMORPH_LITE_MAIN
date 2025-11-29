# POLYMORPH-LITE

Production-ready laboratory automation platform with unified PostgreSQL database.

## 📁 Directory Structure

```
POLYMORPH_LITE_MAIN/
├── README.md                    # Main documentation
├── RELEASE_NOTES.md            # Version history
├── DEPLOYMENT_GUIDE.md         # Deployment instructions
├── SECURITY.md                 # Security policies
├── TESTING.md                  # Testing guide
├── LICENSE                     # MIT License
│
├── .env.example                # Environment template
├── docker-compose.yml          # Docker deployment
├── docker-entrypoint.sh        # Container startup script
├── Dockerfile.backend          # Backend container
│
├── alembic/                    # Database migrations
│   ├── alembic.ini             # Alembic config
│   ├── env.py                  # Migration environment
│   └── versions/               # Migration files
│
├── retrofitkit/                # Main application
│   ├── api/                    # FastAPI routes
│   ├── db/                     # Database layer (unified)
│   ├── compliance/             # RBAC, audit, users
│   ├── core/                   # Orchestrator, config
│   ├── drivers/                # Hardware drivers
│   └── ...
│
├── scripts/                    # Production utilities
│   ├── create_admin_user.py    # Admin setup
│   ├── check_db_health.py      # Health monitoring
│   ├── backup_database.py      # Database backup
│   ├── generate_keys.py        # Cryptographic keys
│   └── unified_cli.py          # CLI interface
│
├── frontend/                   # React frontend
│   ├── src/                    # React components
│   ├── package.json            # Dependencies
│   └── vite.config.ts          # Build config
│
├── bentoml_service/            # AI inference service
│
├── docs/                       # Additional documentation
│   ├── API_DOCUMENTATION.md    # API reference
│   ├── QUICKSTART.md           # Quick start guide
│   ├── USER_MANUAL.md          # User guide
│   └── validation/             # IQ/OQ/PQ templates
│
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── conftest.py             # Pytest configuration
│
├── config/                     # Configuration files
│   ├── workflows/              # Workflow definitions
│   └── hardware/               # Hardware profiles
│
├── recipes/                    # Example workflows
└── workflows/                  # Workflow YAML files
```

## 🗂️ Key Components

### Core Application (`retrofitkit/`)
- **api/** - REST API endpoints (FastAPI)
- **db/** - Unified database layer (27 tables)
- **compliance/** - RBAC, audit, authentication
- **core/** - Workflow orchestrator
- **drivers/** - Hardware device drivers

### Database (`alembic/`)
- **versions/** - Migration history
- **env.py** - Migration configuration
- All 27 tables auto-created on first run

### Scripts (`scripts/`)
Production utilities for database management and system administration.

### Frontend (`frontend/`)
React-based UI with real-time monitoring and LIMS features.

### Documentation (`docs/`)
Comprehensive guides for API,installation, validation, and usage.

## 🚀 Quick Navigation

- **Getting Started**: See [README.md](../README.md)
- **Deployment**: See [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)
- **API Documentation**: See [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)
- **Testing**: See [TESTING.md](../TESTING.md)

## 📝 File Organization Principles

1. **No Duplicates**: Single source of truth for each concept
2. **Clear Hierarchy**: Logical grouping by function
3. **Production-First**: Only essential files included
4. **Docker-Centric**: Optimized for containerized deployment
