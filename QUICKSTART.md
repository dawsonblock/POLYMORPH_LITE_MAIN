# 🚀 Quick Start Guide

## Option 1: Docker (Recommended)

```bash
# Start all services
./start.sh

# Create admin user
docker-compose exec backend python scripts/create_admin_user.py

# Access
open http://localhost:8001/docs
```

## Option 2: Local Development

```bash
# Start locally (no Docker)
./start_dev.sh

# Access
open http://localhost:8001/docs
```

## Option 3: Manual

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python3 -m uvicorn retrofitkit.api.server:app --reload --port 8001
```

## Test It Works

```bash
# Run tests
python3 -m pytest tests/api/test_hardening.py -v

# Check health
curl http://localhost:8001/health

# View API docs
open http://localhost:8001/docs
```

## Stop Services

```bash
# Docker
docker-compose down

# Local (Ctrl+C to stop)
```

## Troubleshooting

**Port already in use:**
```bash
# Change port in command
python3 -m uvicorn retrofitkit.api.server:app --port 8002
```

**Database errors:**
```bash
# Use SQLite (no PostgreSQL needed)
export DATABASE_URL="sqlite+aiosqlite:///./data/polymorph.db"
```

**Import errors:**
```bash
pip install -r requirements.txt
```
