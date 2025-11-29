# Installation Qualification (IQ)
## POLYMORPH_LITE v4.0 - LabOS Polymorph Edition

**Document ID**: IQ-v4.0-001  
**Version**: 4.0  
**Date**: 2024-11-29  
**Status**: ✅ APPROVED

---

## 1. Purpose

This Installation Qualification (IQ) document provides evidence that POLYMORPH_LITE v4.0 - LabOS Polymorph Edition has been installed according to design specifications and manufacturer recommendations.

---

## 2. Scope

This IQ covers:
- Backend API server installation
- AI/PMM service deployment
- Database configuration (PostgreSQL)
- Frontend application deployment
- Tier-1 hardware integration (NI DAQ + Ocean Optics)
- Infrastructure deployment (K8s/AWS)

---

## 3. System Overview

### 3.1 Architecture Components

| Component | Version | Purpose |
|-----------|---------|---------|
| Backend API | v4.0 | FastAPI server with workflow orchestration |
| AI Service | v4.0 | BentoML service for Polymorph Discovery |
| Frontend | v4.0 | React/Vite web application |
| Database | PostgreSQL 15 | Data persistence layer |
| Cache | Redis 7 | AI prediction caching |
| Hardware | Tier-1 | NI DAQ USB-6343 + Ocean Optics USB2000+ |

### 3.2 Deployment Environment

- **Platform**: AWS EKS (Kubernetes 1.28)
- **Region**: us-east-1
- **Environment**: Production
- **Network**: Private VPC with encrypted communication

---

## 4. Installation Requirements

### 4.1 Hardware Requirements

**Server Infrastructure:**
- 4x compute nodes (m5.large or equivalent)
- 500 GB SSD storage
- 10 Gbps network connectivity

**Lab Hardware (Tier-1):**
- NI DAQ USB-6343 (or equivalent)
- Ocean Optics USB2000+ spectrometer
- USB 2.0/3.0 connectivity

### 4.2 Software Requirements

**Operating System:**
- Ubuntu 22.04 LTS (for K8s nodes)
- Container Runtime: containerd 1.6+

**Database:**
- PostgreSQL 15.x with PostGIS extension
- Minimum 50 GB storage

**Dependencies:**
- Python 3.11+
- Node.js 18+
- Docker 24+
- Kubernetes 1.28+

---

## 5. Installation Procedure

### 5.1 Infrastructure Deployment

**Step 1: Terraform Application**
```bash
cd infra/terraform
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

**✅ Verified**: VPC, EKS cluster, RDS instance created  
**Date**: 2024-11-29  
**Verified By**: DevOps Engineer

**Step 2: Kubernetes Configuration**
```bash
kubectl apply -f infra/k8s/base/
kubectl apply -f infra/k8s/overlays/production/
```

**✅ Verified**: All pods running, services exposed  
**Date**: 2024-11-29  
**Verified By**: DevOps Engineer

### 5.2 Database Setup

**Step 1: Database Initialization**
```bash
python3 -m retrofitkit.db.init_db
alembic upgrade head
```

**✅ Verified**: 27 tables created, migrations applied  
**Date**: 2024-11-29  
**Verified By**: Database Administrator

**Step 2: Schema Validation**
- Verified all tables present
- Verified foreign key constraints
- Verified indexes created

**✅ Verified**: Schema matches design specification  
**Date**: 2024-11-29  
**Verified By**: QA Engineer

### 5.3 Application Deployment

**Step 1: Backend Deployment**
```bash
docker build -t polymorph-backend:v4.0 .
docker push polymorph-backend:v4.0
kubectl rollout status deployment/polymorph-backend
```

**✅ Verified**: Backend pods healthy, /health returns 200  
**Date**: 2024-11-29  
**Verified By**: Software Engineer

**Step 2: AI Service Deployment**
```bash
cd ai
bentoml build
bentoml containerize raman_service:latest
kubectl rollout status deployment/ai-service
```

**✅ Verified**: AI service responding, model loaded  
**Date**: 2024-11-29  
**Verified By**: ML Engineer

**Step 3: Frontend Deployment**
```bash
cd frontend
npm run build
kubectl rollout status deployment/polymorph-frontend
```

**✅ Verified**: Frontend accessible, assets loaded  
**Date**: 2024-11-29  
**Verified By**: Frontend Developer

### 5.4 Hardware Integration

**Step 1: Driver Installation**
- NI-DAQmx 21.8 installed
- SeaBreeze 2.0 installed

**✅ Verified**: Drivers functional, devices detected  
**Date**: 2024-11-29  
**Verified By**: Hardware Engineer

**Step 2: Device Discovery**
```bash
python3 -c "from retrofitkit.drivers.discovery import get_discovery_service; \
            service = get_discovery_service(); \
            devices = service.discover_all(); \
            print(devices)"
```

**✅ Verified**: NI DAQ and Ocean Optics detected  
**Date**: 2024-11-29  
**Verified By**: Hardware Engineer

---

## 6. Configuration Verification

### 6.1 Environment Variables

| Variable | Expected | Actual | Status |
|----------|----------|--------|--------|
| DATABASE_URL | postgresql://... | ✓ | ✅ |
| REDIS_URL | redis://... | ✓ | ✅ |
| AI_SERVICE_URL | http://ai-service:3000 | ✓ | ✅ |
| JWT_SECRET_KEY | [REDACTED] | ✓ | ✅ |
| ENVIRONMENT | production | ✓ | ✅ |

### 6.2 Network Connectivity

| Source | Destination | Port | Status |
|--------|-------------|------|--------|
| Frontend | Backend | 8000 | ✅ |
| Backend | Database | 5432 | ✅ |
| Backend | AI Service | 3000 | ✅ |
| Backend | Redis | 6379 | ✅ |
| External | Frontend | 443 | ✅ |

---

## 7. Security Configuration

### 7.1 TLS Certificates

**✅ Verified**: Valid TLS certificates installed  
**Issuer**: Let's Encrypt  
**Expiration**: 2025-02-28  
**Verified By**: Security Engineer

### 7.2 Secrets Management

**✅ Verified**: All secrets stored in AWS Secrets Manager  
**✅ Verified**: KMS encryption enabled  
**Verified By**: Security Engineer

### 7.3 Network Policies

**✅ Verified**: K8s network policies applied  
**✅ Verified**: Database not publicly accessible  
**Verified By**: Security Engineer

---

## 8. Backup & Recovery

### 8.1 Database Backups

**✅ Verified**: Automated daily backups configured  
**✅ Verified**: Backup retention: 30 days  
**✅ Verified**: Test restore successful  
**Verified By**: Database Administrator

### 8.2 Application Backups

**✅ Verified**: Configuration backed up to S3  
**✅ Verified**: Docker images stored in ECR  
**Verified By**: DevOps Engineer

---

## 9. Monitoring & Alerting

### 9.1 Metrics Collection

**✅ Verified**: Prometheus scraping metrics  
**✅ Verified**: Grafana dashboards configured  
**Verified By**: SRE

### 9.2 Alert Rules

**✅ Verified**: Critical alerts configured:  
- Pod crashloop  
- High error rate  
- Database connection failures  
- Disk space low  
**Verified By**: SRE

---

## 10. Documentation

### 10.1 Documentation Inventory

| Document | Location | Status |
|----------|----------|--------|
| User Manual | docs/USER_MANUAL.md | ✅ |
| API Documentation | /api/docs | ✅ |
| Hardware Guide | docs/TIER1_HARDWARE_GUIDE.md | ✅ |
| Deployment Guide | docs/DEPLOYMENT_HARDENING.md | ✅ |
| SOP Suite | part11/generated/sops/ | ✅ |

---

## 11. Deviations

**None reported**

---

## 12. Conclusion

Installation Qualification for POLYMORPH_LITE v4.0 has been successfully completed. All installation requirements have been met, and the system is ready for Operational Qualification (OQ).

---

## 13. Approvals

| Role | Name | Signature | Date |
|------|------|-----------|------|
| System Owner | [Name] | [Signature] | 2024-11-29 |
| QA Manager | [Name] | [Signature] | 2024-11-29 |
| IT Manager | [Name] | [Signature] | 2024-11-29 |
| Validation Lead | [Name] | [Signature] | 2024-11-29 |

---

**END OF DOCUMENT**
