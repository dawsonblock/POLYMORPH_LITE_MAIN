#!/bin/bash
# Quick start script for POLYMORPH-LITE

set -e

echo "🚀 Starting POLYMORPH-LITE..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env not found, copying from .env.example"
    cp .env.example .env
fi

# Start with Docker Compose
echo "📦 Starting services with Docker Compose..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check health
echo "🏥 Checking health..."
curl -s http://localhost:8001/health || echo "Backend not ready yet"

echo ""
echo "✅ Services started!"
echo ""
echo "📍 Access points:"
echo "   API:      http://localhost:8001"
echo "   Docs:     http://localhost:8001/docs"
echo "   Frontend: http://localhost:3001"
echo "   AI:       http://localhost:3000"
echo ""
echo "🔐 Create admin user:"
echo "   docker-compose exec backend python scripts/create_admin_user.py"
echo ""
echo "📊 View logs:"
echo "   docker-compose logs -f backend"
