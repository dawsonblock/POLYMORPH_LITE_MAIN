# POLYMORPH-LITE Deployment Guide

This guide provides comprehensive instructions for deploying POLYMORPH-LITE in production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Production Deployment](#production-deployment)
- [Environment Configuration](#environment-configuration)
- [Database Setup](#database-setup)
- [Monitoring](#monitoring)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- **Docker**: Version 24.0+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.20+ (included with Docker Desktop)
- **Git**: Version 2.30+

### System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 20 GB free space
- Network: 100 Mbps

**Recommended (Production):**
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 100+ GB (SSD recommended)
- Network: 1 Gbps

## Quick Start

### Development Environment

```bash
# Clone the repository
git clone https://github.com/dawsonblock/POLYMORPH_LITE_MAIN.git
cd POLYMORPH_LITE_MAIN

# Copy environment file and configure
cp .env.example .env
# Edit .env with your configuration

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Access the application
# - Frontend: http://localhost
# - Backend API: http://localhost:8001
# - API Documentation: http://localhost:8001/docs
```

### Stopping Services

```bash
docker-compose down
```

### Cleaning Up

```bash
# Stop and remove containers, networks, and volumes
docker-compose down -v

# Remove built images
docker rmi polymorph-backend:latest polymorph-frontend:latest
```

## Production Deployment

### Step 1: Prepare the Server

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin
```

### Step 2: Clone and Configure

```bash
# Clone repository
git clone https://github.com/dawsonblock/POLYMORPH_LITE_MAIN.git
cd POLYMORPH_LITE_MAIN

# Copy production environment template
cp .env.production .env

# CRITICAL: Edit .env and change all passwords and secrets!
nano .env
```

### Step 3: Generate Secure Keys

```bash
# Generate SECRET_KEY
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# Generate JWT_SECRET_KEY
python3 -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"

# Add these to your .env file
```

### Step 4: Build and Deploy

```bash
# Build images
docker-compose build

# Start core services (without AI and monitoring)
docker-compose up -d postgres redis backend frontend

# Check service health
docker-compose ps

# View backend logs
docker-compose logs -f backend

# Access the application
# Frontend: http://YOUR_SERVER_IP
# Backend: http://YOUR_SERVER_IP:8001
```

### Step 5: Start Optional Services

```bash
# Start with AI service
docker-compose --profile with-ai up -d

# Start with monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Start everything
docker-compose --profile with-ai --profile monitoring up -d
```

### Step 6: Configure Reverse Proxy (Recommended)

For production, use nginx or Traefik as a reverse proxy with SSL/TLS:

```nginx
# /etc/nginx/sites-available/polymorph
server {
    listen 80;
    server_name polymorph.yourdomain.com;

    # Redirect to HTTPS
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

    location / {
        proxy_pass http://localhost:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://localhost:8001/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /socket.io/ {
        proxy_pass http://localhost:8001/socket.io/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Environment Configuration

### Required Variables

```bash
# Security (MUST CHANGE IN PRODUCTION)
SECRET_KEY=your-random-secret-key-here
JWT_SECRET_KEY=your-random-jwt-secret-key-here

# Database
POSTGRES_USER=polymorph
POSTGRES_PASSWORD=your-strong-database-password
POSTGRES_DB=polymorph_db

# Redis
REDIS_PASSWORD=your-strong-redis-password
```

### Optional Variables

See `.env.production` for all available configuration options.

## Database Setup

### Initial Migration

Migrations run automatically on container startup. To run manually:

```bash
docker-compose exec backend python -m alembic upgrade head
```

### Create Admin User

```bash
docker-compose exec backend python scripts/create_admin_user.py
```

### Database Backup

```bash
# Backup
docker-compose exec postgres pg_dump -U polymorph polymorph_db > backup.sql

# Restore
docker-compose exec -T postgres psql -U polymorph polymorph_db < backup.sql
```

## Monitoring

### Access Monitoring Dashboards

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (default: admin/admin)
- **Backend Metrics**: http://localhost:8001/metrics

### Health Checks

```bash
# Backend health
curl http://localhost:8001/health

# Database health
docker-compose exec postgres pg_isready

# Redis health
docker-compose exec redis redis-cli ping
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend

# Since timestamp
docker-compose logs --since 2024-01-01T00:00:00
```

## Backup and Recovery

### Automated Backup Script

Create a backup script (`backup.sh`):

```bash
#!/bin/bash
BACKUP_DIR="/backups/polymorph"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker-compose exec -T postgres pg_dump -U polymorph polymorph_db | \
    gzip > "$BACKUP_DIR/db_$DATE.sql.gz"

# Backup application data
tar -czf "$BACKUP_DIR/data_$DATE.tar.gz" data/

# Keep only last 30 days
find $BACKUP_DIR -type f -mtime +30 -delete

echo "Backup completed: $DATE"
```

### Automated Backup with Cron

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /path/to/backup.sh >> /var/log/polymorph-backup.log 2>&1
```

## Troubleshooting

### Container Won't Start

```bash
# Check container status
docker-compose ps

# View detailed logs
docker-compose logs backend

# Restart specific service
docker-compose restart backend

# Rebuild and restart
docker-compose up -d --build backend
```

### Database Connection Issues

```bash
# Check database is running
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Verify connection
docker-compose exec backend python -c "from retrofitkit.db.session import engine; print(engine.connect())"
```

### Port Already in Use

```bash
# Find process using port 8001
sudo lsof -i :8001

# Change port in .env
BACKEND_PORT=8002

# Restart services
docker-compose up -d
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Increase worker count
# In .env:
WORKERS=8

# Scale services
docker-compose up -d --scale backend=2
```

### Reset Everything

```bash
# WARNING: This deletes all data!
docker-compose down -v
docker system prune -a
rm -rf data/
```

## Security Checklist

- [ ] Changed all default passwords and secrets
- [ ] Configured HTTPS/TLS with valid certificates
- [ ] Set up firewall rules (ufw/iptables)
- [ ] Enabled automatic security updates
- [ ] Configured log rotation
- [ ] Set up monitoring alerts
- [ ] Tested backup and restore procedures
- [ ] Configured CORS for production domain
- [ ] Reviewed and hardened database settings
- [ ] Enabled audit logging

## Support

For issues and questions:
- GitHub Issues: https://github.com/dawsonblock/POLYMORPH_LITE_MAIN/issues
- Documentation: See `docs/` directory

## License

See LICENSE file for details.
