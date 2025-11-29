# Installation Qualification (IQ)
## POLYMORPH-LITE Golden Path Demo

**Document ID**: IQ-DEMO-001  
**Version**: 1.0  
**Date**: 2024-11-28  
**System**: POLYMORPH-LITE v3.1  
**Use Case**: Crystallization Screening with Raman + AI

---

## 1. Purpose

This Installation Qualification (IQ) verifies that the POLYMORPH-LITE system has been installed correctly and all components are present and functional for the Golden Path Demo.

## 2. Scope

- Software installation verification
- Database initialization
- Service connectivity
- Configuration validation
- Security setup

## 3. Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ installed
- Node.js 18+ installed (for frontend)
- PostgreSQL 15+ accessible
- Redis 7+ accessible

## 4. Installation Steps

### 4.1 Repository Clone

**Test**: Verify repository is cloned correctly

```bash
cd /path/to/POLYMORPH_LITE_MAIN-3
git status
```

**Expected Result**: Clean working directory, on `main` branch

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

### 4.2 Environment Configuration

**Test**: Verify `.env` file exists and contains required variables

```bash
cat .env | grep -E "DATABASE_URL|REDIS_URL|SECRET_KEY|JWT_SECRET_KEY"
```

**Expected Result**: All required environment variables present

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

### 4.3 Database Initialization

**Test**: Run Alembic migrations

```bash
alembic upgrade head
```

**Expected Result**: All migrations applied successfully, 27+ tables created

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

**Evidence**: Database schema screenshot attached: ☐ Yes ☐ No

---

### 4.4 Backend Service Startup

**Test**: Start backend service

```bash
uvicorn retrofitkit.api.server:app --host 0.0.0.0 --port 8001 --reload
```

**Expected Result**: Service starts without errors, health endpoint returns 200

```bash
curl http://localhost:8001/health
```

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

### 4.5 Frontend Service Startup

**Test**: Start frontend service

```bash
cd frontend && npm install && npm run dev
```

**Expected Result**: Frontend builds and serves on port 3001

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

### 4.6 Service Connectivity

**Test**: Verify all services can communicate

| Service | Endpoint | Expected Status | Actual Status | Pass/Fail |
|---------|----------|----------------|---------------|-----------|
| Backend API | http://localhost:8001/health | 200 OK | | ☐ Pass ☐ Fail |
| Frontend | http://localhost:3001 | 200 OK | | ☐ Pass ☐ Fail |
| Database | PostgreSQL connection | Connected | | ☐ Pass ☐ Fail |
| Redis | Redis connection | Connected | | ☐ Pass ☐ Fail |
| API Docs | http://localhost:8001/docs | 200 OK | | ☐ Pass ☐ Fail |

**Tested By**: ________________  
**Date**: ________________

---

### 4.7 Device Driver Registration

**Test**: Verify device drivers are registered

```bash
curl http://localhost:8001/api/devices
```

**Expected Result**: List of available devices (Raman, DAQ, etc.)

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

### 4.8 Workflow Definitions

**Test**: Verify workflow files are present

```bash
ls -la workflows/hero_crystallization.yaml
```

**Expected Result**: Workflow file exists and is readable

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

### 4.9 Security Configuration

**Test**: Verify security settings

| Security Feature | Configuration | Status | Pass/Fail |
|-----------------|---------------|--------|-----------|
| JWT Secret | Set in .env | | ☐ Pass ☐ Fail |
| Password Hashing | bcrypt enabled | | ☐ Pass ☐ Fail |
| CORS | Configured | | ☐ Pass ☐ Fail |
| HTTPS | TLS configured (prod) | | ☐ Pass ☐ Fail |
| Rate Limiting | Enabled | | ☐ Pass ☐ Fail |

**Tested By**: ________________  
**Date**: ________________

---

### 4.10 Logging Configuration

**Test**: Verify logging is functional

```bash
tail -f logs/polymorph.log
```

**Expected Result**: Log entries appear when actions are performed

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

## 5. Installation Summary

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| POLYMORPH-LITE | 3.1 | ☐ Pass ☐ Fail | |
| PostgreSQL | 15+ | ☐ Pass ☐ Fail | |
| Redis | 7+ | ☐ Pass ☐ Fail | |
| Python | 3.11+ | ☐ Pass ☐ Fail | |
| Node.js | 18+ | ☐ Pass ☐ Fail | |
| Docker | 24+ | ☐ Pass ☐ Fail | |

---

## 6. Deviations

List any deviations from expected installation:

1. ________________________________________________________________
2. ________________________________________________________________
3. ________________________________________________________________

---

## 7. Approval

**Installation Qualified**: ☐ Yes ☐ No ☐ Conditional

**Conditions** (if conditional):
________________________________________________________________
________________________________________________________________

**Performed By**: ________________  
**Signature**: ________________  
**Date**: ________________

**Reviewed By**: ________________  
**Signature**: ________________  
**Date**: ________________

**Approved By** (QA/Compliance): ________________  
**Signature**: ________________  
**Date**: ________________

---

## 8. Attachments

- [ ] Environment configuration file (.env.example)
- [ ] Database schema diagram
- [ ] Service startup logs
- [ ] Health check responses
- [ ] Device registry screenshot

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-11-28 | POLYMORPH Team | Initial IQ for Golden Path Demo |

**Next Document**: OQ-DEMO-001 (Operational Qualification)
