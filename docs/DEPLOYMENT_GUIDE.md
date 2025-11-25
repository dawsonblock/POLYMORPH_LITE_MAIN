# POLYMORPH-4 Lite Deployment Guide

**Version**: 2.0.0  
**Last Updated**: November 25, 2025

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Configuration](#configuration)
6. [Security Hardening](#security-hardening)
7. [Monitoring & Logging](#monitoring--logging)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8 cores |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 100 GB SSD | 500 GB NVMe SSD |
| **Network** | 100 Mbps | 1 Gbps |
| **GPU** | None | NVIDIA GPU (for AI) |

### Software Requirements

- **Operating System**: Ubuntu 20.04+ / macOS 12+ / Windows 11
- **Python**: 3.11+
- **Node.js**: 18+
- **Docker**: 24.0+ (optional, for containerized deployment)
- **Redis**: 7.0+ (for state persistence)
- **PostgreSQL**: 14+ (optional, for production deployments)

### Hardware Interfaces

- **Raman Spectrometer**: Horiba LabRAM (USB or Ethernet)
- **DAQ Module**: Gamry potentiostat or National Instruments DAQ
- **Sensors**: Temperature (thermocouple), Pressure, Flow meters

---

## Quick Start

### Local Development Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/polymorph/polymorph-lite.git
   cd polymorph-lite
   ```

2. **Install Python Dependencies**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install Frontend Dependencies**
   ```bash
   cd gui-v2/frontend
   npm install
   cd ../..
   ```

4. **Start Redis**
   ```bash
   # macOS
   brew services start redis
   
   # Ubuntu
   sudo systemctl start redis
   
   # Docker
   docker run -d -p 6379:6379 redis:7-alpine
   ```

5. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

6. **Start BentoML AI Service**
   ```bash
   cd bentoml_service
   ./run_service.sh
   cd ..
   ```

7. **Start Backend**
   ```bash
   python -m retrofitkit.main
   ```

8. **Start Frontend**
   ```bash
   cd gui-v2/frontend
   npm run dev
   ```

9. **Access Application**
   - Frontend: `http://localhost:3000`
   - Backend API: `http://localhost:8001`
   - API Docs: `http://localhost:8001/docs`

---

## Production Deployment

### Architecture

```
                 ┌─────────────┐
                 │   Internet  │
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │  NGINX/LB   │ (SSL Termination)
                 └──────┬──────┘
                        │
       ┌────────────────┼────────────────┐
       │                │                │
┌──────▼─────┐   ┌─────▼──────┐  ┌─────▼──────┐
│  Frontend  │   │   Backend  │  │ AI Service │
│   (Vite)   │   │  (FastAPI) │  │ (BentoML)  │
└────────────┘   └─────┬──────┘  └────────────┘
                       │
              ┌────────┼────────┐
              │        │        │
        ┌─────▼───┐ ┌─▼────┐ ┌─▼──────┐
        │ Redis   │ │ DB   │ │Hardware│
        └─────────┘ └──────┘ └────────┘
```

### Prerequisites

1. **Domain Name**: `polymorph.yourlab.edu`
2. **SSL Certificate**: Let's Encrypt or corporate CA
3. **Server Access**: SSH with sudo privileges
4. **Firewall Rules**: Ports 80, 443, 8001 (backend), 3000 (AI)

### Step-by-Step Deployment

#### 1. Prepare Server

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
  python3.11 python3.11-venv python3-pip \
  nodejs npm \
  nginx \
  redis-server \
  git \
  certbot python3-certbot-nginx

# Start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

#### 2. Create Service User

```bash
sudo useradd -m -s /bin/bash polymorph
sudo usermod -aG sudo polymorph
sudo su - polymorph
```

#### 3. Deploy Application

```bash
# Clone repository
git clone https://github.com/polymorph/polymorph-lite.git /opt/polymorph
cd /opt/polymorph

# Setup Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build frontend
cd gui-v2/frontend
npm install
npm run build
cd ../..

# Set permissions
sudo chown -R polymorph:polymorph /opt/polymorph
```

#### 4. Configure Systemd Services

**Backend Service** (`/etc/systemd/system/polymorph-backend.service`):
```ini
[Unit]
Description=POLYMORPH-4 Lite Backend
After=network.target redis.service

[Service]
Type=simple
User=polymorph
WorkingDirectory=/opt/polymorph
Environment="PATH=/opt/polymorph/venv/bin"
ExecStart=/opt/polymorph/venv/bin/python -m retrofitkit.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**AI Service** (`/etc/systemd/system/polymorph-ai.service`):
```ini
[Unit]
Description=POLYMORPH-4 AI Service (BentoML)
After=network.target

[Service]
Type=simple
User=polymorph
WorkingDirectory=/opt/polymorph/bentoml_service
Environment="PATH=/opt/polymorph/venv/bin"
ExecStart=/opt/polymorph/bentoml_service/run_service.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and Start Services:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable polymorph-backend polymorph-ai
sudo systemctl start polymorph-backend polymorph-ai

# Check status
sudo systemctl status polymorph-backend
sudo systemctl status polymorph-ai
```

#### 5. Configure NGINX

**Create Configuration** (`/etc/nginx/sites-available/polymorph`):
```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name polymorph.yourlab.edu;
    return 301 https://$server_name$request_uri;
}

# HTTPS Server
server {
    listen 443 ssl http2;
    server_name polymorph.yourlab.edu;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/polymorph.yourlab.edu/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/polymorph.yourlab.edu/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Frontend (Static Files)
    location / {
        root /opt/polymorph/gui-v2/frontend/dist;
        try_files $uri $uri/ /index.html;
        expires 1h;
        add_header Cache-Control "public, immutable";
    }

    # Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health Check
    location /health {
        proxy_pass http://127.0.0.1:8001/health;
    }

    # WebSocket
    location /socket.io/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    # AI Service
    location /ai/ {
        proxy_pass http://127.0.0.1:3000/;
        proxy_set_header Host $host;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Logs
    access_log /var/log/nginx/polymorph-access.log;
    error_log /var/log/nginx/polymorph-error.log;
}
```

**Enable Site:**
```bash
sudo ln -s /etc/nginx/sites-available/polymorph /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 6. Obtain SSL Certificate

```bash
sudo certbot --nginx -d polymorph.yourlab.edu
```

---

## Docker Deployment

### Docker Compose Setup

**`docker-compose.yml`:**
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8001:8001"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - AI_SERVICE_URL=http://ai-service:3000/infer
    depends_on:
      - redis
    volumes:
      - ./data:/mnt/data
    restart: unless-stopped

  ai-service:
    image: polymorph:j52wavwkic73h4ri
    ports:
      - "3000:3000"
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    build:
      context: ./gui-v2/frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  redis-data:
```

**Backend Dockerfile** (`Dockerfile.backend`):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY retrofitkit ./retrofitkit
COPY config.yaml .

# Expose port
EXPOSE 8001

# Run application
CMD ["python", "-m", "retrofitkit.main"]
```

**Frontend Dockerfile** (`gui-v2/frontend/Dockerfile`):
```dockerfile
FROM node:18-alpine as builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

**Deploy with Docker Compose:**
```bash
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# System
ENVIRONMENT=production
LOG_LEVEL=INFO
DATA_DIR=/mnt/data

# Security
SECRET_KEY=your-secret-key-here-change-this
JWT_EXPIRATION=3600
MFA_ENABLED=true

# Database
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# AI Service
AI_SERVICE_URL=http://localhost:3000/infer
AI_TIMEOUT=5.0

# Hardware
DAQ_BACKEND=gamry
RAMAN_PROVIDER=horiba

# Email (Alerts)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@yourlab.edu
SMTP_PASSWORD=app-specific-password
```

### Configuration File

**`config.yaml`:**
```yaml
system:
  name: "POLYMORPH-4 Lite Production"
  data_dir: "/mnt/data/Polymorph4_Retrofit_Kit_v1/data"
  log_level: "INFO"

security:
  secret_key: "${SECRET_KEY}"
  jwt_expiration: 3600
  mfa_enabled: true
  password_policy:
    min_length: 12
    require_uppercase: true
    require_numbers: true
    require_special: true

api:
  host: "0.0.0.0"
  port: 8001
  cors_origins:
    - "https://polymorph.yourlab.edu"
  rate_limit:
    requests_per_minute: 100
    requests_per_hour: 1000

ai:
  service_url: "http://localhost:3000/infer"
  timeout: 5.0
  circuit_breaker:
    failure_threshold: 3
    recovery_timeout: 60

daq:
  backend: "gamry"
  timeout: 10.0
  retry_attempts: 3

raman:
  provider: "horiba"
  integration_time: 1.0
  laser_power: 50

monitoring:
  prometheus_port: 9090
  grafana_enabled: true
```

---

## Security Hardening

### 1. Firewall Configuration

```bash
# Ubuntu (UFW)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. Secure Redis

**Edit `/etc/redis/redis.conf`:**
```conf
# Bind to localhost only
bind 127.0.0.1 ::1

# Require password
requirepass your-strong-redis-password

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command KEYS ""
```

### 3. SSL/TLS Best Practices

- Use TLS 1.2+ only
- Strong cipher suites
- HSTS enabled
- Certificate pinning (optional)

### 4. Application Security

- **Change default passwords**
- **Enable MFA** for all users
- **Regular security audits**
- **Keep dependencies updated**
- **Monitor for CVEs**

---

## Monitoring & Logging

### Logging Configuration

**Edit `config.yaml`:**
```yaml
logging:
  level: INFO
  format: json
  outputs:
    - type: file
      path: /var/log/polymorph/app.log
      rotation: daily
      retention: 30
    - type: syslog
      facility: local0
    - type: stdout
```

### Metrics with Prometheus

**Exposed Metrics:**
- `polymorph_experiments_total`: Total experiments run
- `polymorph_experiments_active`: Currently running experiments
- `polymorph_ai_inference_duration_seconds`: AI inference latency
- `polymorph_hardware_temperature_celsius`: Current temperature
- `polymorph_alerts_total`: Total alerts triggered

**Prometheus Configuration** (`prometheus.yml`):
```yaml
scrape_configs:
  - job_name: 'polymorph'
    static_configs:
      - targets: ['localhost:9090']
```

### Grafana Dashboards

Import pre-built dashboards from `monitoring/grafana/`.

---

## Backup & Recovery

### Automated Backups

**Create Backup Script** (`/opt/polymorph/scripts/backup.sh`):
```bash
#!/bin/bash

BACKUP_DIR="/backups/polymorph"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup data directory
tar -czf "$BACKUP_DIR/data_$DATE.tar.gz" /mnt/data/Polymorph4_Retrofit_Kit_v1/

# Backup Redis
redis-cli --rdb "$BACKUP_DIR/redis_$DATE.rdb"

# Backup configuration
cp /opt/polymorph/.env "$BACKUP_DIR/env_$DATE"
cp /opt/polymorph/config.yaml "$BACKUP_DIR/config_$DATE.yaml"

# Remove backups older than 30 days
find "$BACKUP_DIR" -mtime +30 -delete

echo "Backup completed: $DATE"
```

**Schedule with Cron:**
```bash
# Daily at 2 AM
0 2 * * * /opt/polymorph/scripts/backup.sh
```

### Disaster Recovery

**Recovery Steps:**
1. Reinstall OS and dependencies
2. Restore application from backup
3. Restore Redis database
4. Restore data directory
5. Restart services
6. Verify system health

**RTO (Recovery Time Objective)**: < 4 hours  
**RPO (Recovery Point Objective)**: < 24 hours

---

## Troubleshooting

### Service Won't Start

**Check logs:**
```bash
sudo journalctl -u polymorph-backend -n 100
sudo journalctl -u polymorph-ai -n 100
```

**Common issues:**
- Port already in use
- Missing dependencies
- Invalid configuration
- Redis not running

### High Memory Usage

**Check processes:**
```bash
top -o %MEM
```

**Solution:**
- Increase server RAM
- Optimize AI model (quantization)
- Adjust worker processes

### Slow AI Inference

**Symptoms:** >1s inference time

**Solutions:**
- Enable GPU acceleration
- Use smaller model
- Batch processing
- Cache results

---

## Maintenance

### Updates

**Update Application:**
```bash
cd /opt/polymorph
git pull
source venv/bin/activate
pip install -r requirements.txt --upgrade
sudo systemctl restart polymorph-backend polymorph-ai
```

**Update Frontend:**
```bash
cd /opt/polymorph/gui-v2/frontend
npm install
npm run build
sudo systemctl reload nginx
```

### Database Maintenance

**Redis cleanup:**
```bash
redis-cli FLUSHDB  # Use with caution!
```

### Log Rotation

**Configure logrotate** (`/etc/logrotate.d/polymorph`):
```
/var/log/polymorph/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 polymorph polymorph
    sharedscripts
    postrotate
        systemctl reload polymorph-backend
    endscript
}
```

---

## Support

- **Documentation**: `https://docs.polymorph.lab`
- **Issues**: `https://github.com/polymorph/polymorph-lite/issues`
- **Email**: `support@polymorph.lab`
- **Slack**: `polymorph-users.slack.com`

---

**© 2025 POLYMORPH-4 Research Team. All Rights Reserved.**
