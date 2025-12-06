#!/bin/bash
# Run POLYMORPH-LITE with proper Python environment

# Use Python 3.11 from your conda environment
PYTHON="/Users/dawsonblock/Documents/GitHub/OpenManus/.conda/bin/python"

# Install dependencies if needed
echo "📦 Installing dependencies..."
$PYTHON -m pip install -q pydantic-settings python-socketio aiofiles fastapi uvicorn sqlalchemy alembic psycopg2-binary asyncpg

# Run server
echo "🚀 Starting server..."
export PYTHONPATH=/Users/dawsonblock/POLYMORPH_LITE_MAIN-4:$PYTHONPATH
$PYTHON -m uvicorn retrofitkit.api.server:app --reload --port 8001
