#!/bin/bash

# POLYMORPH-4 Hybrid Deployment Launcher
# Usage: ./run_hybrid.sh
# Requirement: Python 3.11+ installed on host, Docker Desktop running

echo "🔬 Starting POLYMORPH-4 Hybrid Mode..."

# 1. Start Infrastructure in Docker (Detach mode)
# We exclude the backend service from the compose file
echo "🐳 Spawning Infrastructure (Redis, DB, AI, Frontend)..."
docker compose -f docker-compose.yml up -d redis ai-service frontend

# 2. Set Environment for Local Backend
export ENVIRONMENT=production
export P4_CONFIG=config/config.yaml
export P4_DATA_DIR=$(pwd)/data
export REDIS_HOST=localhost
export DB_HOST=localhost
export AI_SERVICE_URL=http://localhost:3000/infer

# 3. Check Python Environment
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -r requirements-hw.txt
else
    source .venv/bin/activate
fi

# 4. Run Hardware Wizard if config missing
if [ ! -f "config/config.yaml" ]; then
    echo "🔧 Running Hardware Wizard..."
    python scripts/hardware_wizard.py
fi

# 5. Start Backend on Metal
echo "🚀 Starting Backend on Metal (Port 8001)..."
echo "   Hardware drivers running natively on host OS"
echo "   Log: logs/hybrid_backend.log"
mkdir -p logs
uvicorn retrofitkit.api.server:app --host 0.0.0.0 --port 8001 --log-level info
