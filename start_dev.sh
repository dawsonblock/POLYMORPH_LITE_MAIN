#!/bin/bash
# Development mode - run locally without Docker

set -e

echo "🔧 Starting POLYMORPH-LITE (Development Mode)..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

# Check .env
if [ ! -f .env ]; then
    echo "⚠️  .env not found, copying from .env.example"
    cp .env.example .env
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Run migrations (if using SQLite)
echo "🗄️  Setting up database..."
python3 -c "from retrofitkit.core.database import init_db; import asyncio; asyncio.run(init_db())"

# Start server
echo ""
echo "🚀 Starting server on http://localhost:8001"
echo "📖 API docs: http://localhost:8001/docs"
echo ""
python3 -m uvicorn retrofitkit.api.server:app --host 0.0.0.0 --port 8001 --reload
