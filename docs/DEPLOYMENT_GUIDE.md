# POLYMORPH-LITE v3.0 Deployment Guide

**Production-Ready Lab Operating System with LIMS, Workflow Builder, and 21 CFR Part 11 Compliance**

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start (Development)](#quick-start-development)
3. [Production Deployment](#production-deployment)
4. [Database Setup](#database-setup)
5. [Environment Configuration](#environment-configuration)
6. [Running Migrations](#running-migrations)
7. [Testing](#testing)
8. [Monitoring & Observability](#monitoring--observability)
9. [Backup & Recovery](#backup--recovery)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.11 or higher
- **PostgreSQL**: 15+ (production) or SQLite 3 (development)
- **Redis**: 7+ (for caching and sessions)
- **Node.js**: 20 LTS (for frontend, optional)
- **Docker**: 24+ and Docker Compose (recommended)

### Hardware Requirements
**Minimum (Development)**:
- CPU: 4 cores
- RAM: 8 GB
- Disk: 20 GB

**Recommended (Production)**:
- CPU: 8+ cores
- RAM: 16+ GB
- Disk: 100+ GB SSD

---

## Quick Start (Development)

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/dawsonblock/POLYMORPH_LITE_MAIN.git
cd POLYMORPH_LITE_MAIN

# 2. Copy environment file
cp .env.example .env

# 3. Start all services
docker-compose up -d

# 4. Check logs
docker-compose logs -f backend

# 5. Access the application
# Backend API: http://localhost:8001
# API Docs: http://localhost:8001/docs
# Frontend: http://localhost (if frontend container is running)
```

### Option 2: Local Python Environment

```bash
# 1. Clone and navigate
git clone https://github.com/dawsonblock/POLYMORPH_LITE_MAIN.git
cd POLYMORPH_LITE_MAIN

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env

# 5. Initialize database
alembic upgrade head

# 6. Run the server
uvicorn retrofitkit.api.server:app --host 0.0.0.0 --port 8001 --reload
```

---

## Production Deployment

### Step 1: Prepare Environment

```bash
# 1. Set up production server (Ubuntu 22.04 example)
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip \
    postgresql-15 redis-server nginx

# 2. Create application user
sudo useradd -m -s /bin/bash polymorph
sudo su - polymorph

# 3. Clone repository
git clone https://github.com/dawsonblock/POLYMORPH_LITE_MAIN.git
cd POLYMORPH_LITE_MAIN
```

### Step 2: Configure Production Environment

```bash
# 1. Copy production template
cp .env.production.template .env.production

# 2. Generate secrets
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))" >> .env.production
python3 -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))" >> .env.production

# 3. Edit .env.production
nano .env.production

# Required changes:
# - DATABASE_URL: Set to your PostgreSQL connection string
# - REDIS_PASSWORD: Set strong password
# - ALLOWED_ORIGINS: Set to your domain
# - All ${VARIABLE} placeholders
```

### Step 3: Set Up PostgreSQL

```bash
# 1. Create database and user
sudo -u postgres psql <<EOF
CREATE DATABASE polymorph_prod;
CREATE USER polymorph WITH PASSWORD 'STRONG_PASSWORD_HERE';
GRANT ALL PRIVILEGES ON DATABASE polymorph_prod TO polymorph;
\c polymorph_prod
GRANT ALL ON SCHEMA public TO polymorph;
EOF

# 2. Update .env.production with connection string
DATABASE_URL=postgresql://polymorph:STRONG_PASSWORD_HERE@localhost:5432/polymorph_prod
```

### Step 4: Generate RSA Keys for E-Signatures

```bash
# 1. Create secrets directory
mkdir -p secrets

# 2. Generate RSA key pair
openssl genrsa -out secrets/private.pem 2048
openssl rsa -in secrets/private.pem -pubout -out secrets/public.pem

# 3. Secure permissions
chmod 600 secrets/private.pem
chmod 644 secrets/public.pem

# 4. Update .env.production
RSA_PRIVATE_KEY_PATH=./secrets/private.pem
RSA_PUBLIC_KEY_PATH=./secrets/public.pem
```

### Step 5: Install and Configure

```bash
# 1. Install Python dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Run database migrations
export $(cat .env.production | xargs)
alembic upgrade head

# 3. Create initial admin user (optional script needed)
python scripts/create_admin.py
```

### Step 6: Set Up Systemd Service

Create `/etc/systemd/system/polymorph.service`:

```ini
[Unit]
Description=POLYMORPH-LITE Backend Service
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=polymorph
Group=polymorph
WorkingDirectory=/home/polymorph/POLYMORPH_LITE_MAIN
Environment="PATH=/home/polymorph/POLYMORPH_LITE_MAIN/venv/bin"
EnvironmentFile=/home/polymorph/POLYMORPH_LITE_MAIN/.env.production
ExecStart=/home/polymorph/POLYMORPH_LITE_MAIN/venv/bin/uvicorn \
    retrofitkit.api.server:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable polymorph
sudo systemctl start polymorph
sudo systemctl status polymorph
```

### Step 7: Configure Nginx Reverse Proxy

Create `/etc/nginx/sites-available/polymorph`:

```nginx
server {
    listen 80;
    server_name polymorph.yourdomain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name polymorph.yourdomain.com;

    # SSL certificates (use Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/polymorph.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/polymorph.yourdomain.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    # Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # API documentation
    location /docs {
        proxy_pass http://127.0.0.1:8001;
    }

    # Frontend (if deployed separately)
    location / {
        root /var/www/polymorph/frontend;
        try_files $uri $uri/ /index.html;
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/polymorph /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 8: SSL Certificate (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d polymorph.yourdomain.com
```

---

## Database Setup

### PostgreSQL Configuration

Recommended `/etc/postgresql/15/main/postgresql.conf` settings:

```ini
# Connections
max_connections = 100

# Memory
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
work_mem = 32MB

# Checkpoint
checkpoint_completion_target = 0.9
wal_buffers = 16MB

# Logging
log_min_duration_statement = 1000  # Log slow queries (>1s)
```

### Running Migrations

```bash
# Check current version
alembic current

# View migration history
alembic history

# Upgrade to latest
alembic upgrade head

# Downgrade one version
alembic downgrade -1

# Generate new migration (after model changes)
alembic revision --autogenerate -m "Description of changes"
```

---

## Environment Configuration

### Critical Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `production` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@localhost:5432/db` |
| `SECRET_KEY` | Application secret key | Generate with `secrets.token_urlsafe(32)` |
| `JWT_SECRET_KEY` | JWT signing key | Generate with `secrets.token_urlsafe(32)` |
| `REDIS_HOST` | Redis server hostname | `localhost` or `redis` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `https://polymorph.yourdomain.com` |
| `LOG_LEVEL` | Logging level | `WARNING` for production |

### Compliance Settings (21 CFR Part 11)

| Variable | Description | Default |
|----------|-------------|---------|
| `PASSWORD_EXPIRY_DAYS` | Password expiration period | `90` |
| `PASSWORD_HISTORY_COUNT` | Passwords to remember | `5` |
| `MAX_FAILED_LOGIN_ATTEMPTS` | Login attempts before lockout | `5` |
| `ACCOUNT_LOCKOUT_DURATION_MINUTES` | Lockout duration | `30` |
| `SESSION_TIMEOUT_MINUTES` | Session inactivity timeout | `30` |

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=retrofitkit --cov-report=html

# Run specific test file
pytest tests/test_api_samples.py

# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x
```

### Test Categories

- **Unit Tests**: Individual component testing
- **API Tests**: REST endpoint testing
- **Integration Tests**: Multi-component workflows
- **Compliance Tests**: Audit trail, signatures, reports

---

## Monitoring & Observability

### Prometheus Metrics

Metrics available at `http://localhost:8001/metrics`:

- `polymorph_requests_total` - Total API requests
- `polymorph_run_duration_seconds` - Workflow execution time
- `polymorph_audit_events_total` - Audit events logged
- `polymorph_samples_created_total` - Samples created

### Logging

Logs location (production):
- Application logs: `/var/log/polymorph/app.log`
- Nginx access: `/var/log/nginx/access.log`
- Nginx errors: `/var/log/nginx/error.log`
- PostgreSQL: `/var/log/postgresql/postgresql-15-main.log`

### Health Checks

```bash
# Application health
curl http://localhost:8001/api/health

# Detailed diagnostics
curl http://localhost:8001/api/health/diagnostics
```

---

## Backup & Recovery

### Database Backup

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/backup/polymorph"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -U polymorph polymorph_prod | gzip > ${BACKUP_DIR}/polymorph_${DATE}.sql.gz

# Keep last 30 days
find ${BACKUP_DIR} -name "polymorph_*.sql.gz" -mtime +30 -delete
```

### Automated Backups (Cron)

```bash
# Add to crontab
0 2 * * * /usr/local/bin/backup-polymorph.sh
```

### Recovery

```bash
# Restore from backup
gunzip < /backup/polymorph/polymorph_20250127_020000.sql.gz | \
    psql -U polymorph polymorph_prod
```

---

## Troubleshooting

### Common Issues

**Issue**: Database connection errors
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
psql -U polymorph -d polymorph_prod -h localhost

# Check DATABASE_URL in .env
echo $DATABASE_URL
```

**Issue**: Migration failures
```bash
# Check current version
alembic current

# Try manual migration
alembic upgrade head --sql > migration.sql
# Review migration.sql, then apply manually
```

**Issue**: Permission denied errors
```bash
# Fix ownership
sudo chown -R polymorph:polymorph /home/polymorph/POLYMORPH_LITE_MAIN

# Fix permissions
chmod 600 .env.production
chmod 600 secrets/private.pem
```

**Issue**: API returns 500 errors
```bash
# Check application logs
sudo journalctl -u polymorph -f

# Enable debug mode temporarily
export LOG_LEVEL=DEBUG
sudo systemctl restart polymorph
```

---

## Security Checklist

- [ ] Change all default passwords
- [ ] Generate unique SECRET_KEY and JWT_SECRET_KEY
- [ ] Configure firewall (UFW/iptables)
- [ ] Enable SSL/TLS (Let's Encrypt)
- [ ] Set up automated backups
- [ ] Configure log rotation
- [ ] Enable audit logging
- [ ] Restrict database access
- [ ] Set up monitoring alerts
- [ ] Review and harden Nginx configuration

---

## Support & Documentation

- **API Documentation**: `https://your-domain.com/docs`
- **GitHub Repository**: https://github.com/dawsonblock/POLYMORPH_LITE_MAIN
- **Issue Tracker**: https://github.com/dawsonblock/POLYMORPH_LITE_MAIN/issues

---

**Last Updated**: 2025-11-27
**Version**: 3.0.0
