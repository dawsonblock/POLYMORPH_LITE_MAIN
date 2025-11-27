# POLYMORPH-LITE v3.0.0 - Release Notes

## 🎉 Major Release: Unified Database Architecture

**Release Date**: November 27, 2025  
**Version**: 3.0.0  
**Type**: Major (Breaking Changes)

---

## 🚨 Breaking Changes

### Database Migration
- **Old**: Scattered SQLite files (`system.db`, `audit.db`)
- **New**: Unified PostgreSQL database with 27 tables
- **Impact**: Existing user accounts must be recreated
- **Migration**: No automatic data migration from v2.0

### API Changes
- Added RBAC enforcement to routes
- JWT authentication now required for all protected endpoints
- Session management migrated to dependency injection

---

## ✨ New Features

### Database & ORM
- ✅ **Unified PostgreSQL Database** with 27 tables
- ✅ **Alembic Migrations** for version-controlled schema
- ✅ **Single ORM Layer** (`retrofitkit/db/`) - eliminated split-brain
- ✅ **Automatic Migrations** on Docker startup

### LIMS Functionality
- ✅ **Sample Tracking** with parent-child lineage
- ✅ **Inventory Management** with stock lots and expiration
- ✅ **Calibration Logging** for instruments
- ✅ **Project/Batch Organization**

### Security & Compliance
- ✅ **Enhanced RBAC** with 4 roles (admin, scientist, technician, compliance)
- ✅ **Cryptographic Audit Trail** with hash-chain verification
- ✅ **Password History** tracking (prevent reuse)
- ✅ **Account Locking** after failed login attempts
- ✅ **JWT Authentication** with configurable expiration

### Production Tools
- ✅ **Health Check Script** (`scripts/check_db_health.py`)
- ✅ **Database Backup Utility** (`scripts/backup_database.py`)
- ✅ **Admin User Setup** (`scripts/create_admin_user.py`)
- ✅ **Docker Entrypoint** with auto-migrations

### Developer Experience
- ✅ **Improved Documentation** with API examples
- ✅ **Clear Migration Path** from v2.0
- ✅ **Comprehensive Testing Utilities**

---

## 🔧 Improvements

### Performance
- Connection pooling with PostgreSQL
- Optimized query patterns
- Indexed columns for faster lookups

### Code Quality
- Reduced `database/models.py` from 600 → 44 lines (compatibility shim)
- Organized models into domain-specific files
- Eliminated duplicate ORM definitions

### Deployment
- One-command Docker deployment
- Automatic role seeding
- Environment-based configuration

---

## 📊 Database Schema

### New Tables (27 Total)

**Authentication & RBAC** (3)
- `users` - User accounts with MFA
- `roles` - Role definitions with permissions
- `user_roles` - User-role assignments

**Audit** (1)
- `audit` - Hash-chain audit trail

**Devices** (2)
- `devices` - Instrument registry
- `device_status` - Calibration & health tracking

**LIMS** (5)
- `projects` - Study containers
- `containers` - Physical storage
- `batches` - Sample batching
- `samples` - Sample tracking
- `sample_lineage` - Parent-child relationships

**Inventory** (3)
- `vendors` - Supplier info
- `inventory_items` - Item master data
- `stock_lots` - Lot tracking with expiration

**Calibration** (1)
- `calibration_entries` - Instrument calibration logs

**Workflows** (4)
- `workflow_versions` - Versioned definitions
- `workflow_executions` - Run history
- `workflow_sample_assignments` - Sample-run mapping
- `config_snapshots` - Immutable config

**Multi-Site** (4)
- `organizations` - Top-level tenants
- `labs` - Sites within orgs
- `nodes` - Edge compute nodes
- `device_hubs` - Device registries

---

## 🔄 Upgrade Guide

### From v2.0 to v3.0

#### Prerequisites
- PostgreSQL 15+ installed or Docker
- Updated `.env` file with PostgreSQL credentials

#### Steps

1. **Backup v2.0 Data** (if needed)
   ```bash
   # Backup SQLite databases
   cp data/system.db backup/
   cp data/audit.db backup/
   ```

2. **Update Code**
   ```bash
   git pull origin main
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with PostgreSQL credentials
   ```

4. **Deploy v3.0**
   ```bash
   docker-compose up -d
   # Migrations run automatically
   ```

5. **Create Admin User**
   ```bash
   docker-compose exec backend python scripts/create_admin_user.py
   ```

6. **Verify**
   ```bash
   docker-compose exec backend python scripts/check_db_health.py
   ```

#### Data Migration (Optional)

If you need to preserve v2.0 data:
- User accounts: Recreate manually or via API
- Audit logs: Export from SQLite, import via API
- Workflows: Re-upload YAML definitions
- Samples: Import via bulk API endpoint

---

## 🐛 Bug Fixes

- Fixed Docker entrypoint (was referencing non-existent `retrofitkit.main`)
- Fixed split-brain database issue (two competing ORM systems)
- Fixed missing Alembic configuration
- Fixed import errors in compliance modules
- Fixed JWT authentication in tokens.py

---

## 📝 Known Issues

- Frontend LIMS pages not yet implemented (API ready)
- Workflow builder → orchestrator integration incomplete
- Old `database/models.py` still exists as compatibility shim (can be removed later)

---

## 🎯 Next Release (v3.1.0 - Planned)

- Frontend LIMS UI (React pages for samples/inventory)
- Workflow builder orchestrator integration
- Advanced audit reports
- Multi-site organization features
- Token revocation model
- Comprehensive test coverage

---

## 📚 Documentation

- [README](README.md) - Complete feature overview
- [Walkthrough](walkthrough.md) - Deployment guide
- [API Docs](http://localhost:8001/docs) - Swagger UI
- [Migration Guide](alembic/versions/001_unified_schema.py) - Database schema

---

## 🤝 Contributors

- Core team
- Community feedback and testing

---

## 📜 License

MIT License - see LICENSE file

---

## 🔗 Links

- **Repository**: https://github.com/dawsonblock/POLYMORPH_LITE_MAIN
- **Issues**: https://github.com/dawsonblock/POLYMORPH_LITE_MAIN/issues
- **Releases**: https://github.com/dawsonblock/POLYMORPH_LITE_MAIN/releases

---

**Full Changelog**: v2.0.0...v3.0.0
