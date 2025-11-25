#!/bin/bash
# Simplified Production Deployment for POLYMORPH-4 Lite

set -e

echo "🚀 POLYMORPH-4 Lite - Quick Deploy"
echo "===================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not installed. Please install Docker first."
    exit 1
fi

# Create .env if doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.production.example .env
    echo "⚠️  IMPORTANT: Edit .env and update SECRET_KEY and REDIS_PASSWORD!"
    echo "   Generate secret: python3 -c 'import secrets; print(secrets.token_urlsafe(32))'"
    echo ""
    echo "Press Enter to continue after editing .env, or Ctrl+C to abort..."
    read
fi

# Check if AI service image exists
if ! docker images | grep -q "polymorph.*j52wavwkic73h4ri"; then
    echo "⚠️  AI service Docker image not found"
    echo "   Building from bentoml_service/..."
    cd bentoml_service
    if [ -f "run_service.sh" ]; then
        echo "   Note: BentoML service should be built separately"
        echo "   For now, using simulator mode"
    fi
    cd ..
fi

# Build frontend
echo "🏗️  Building frontend..."
cd gui-v2/frontend
if [ ! -d "node_modules" ]; then
    echo "   Installing dependencies..."
    npm install
fi
npm run build
cd ../..
echo "✅ Frontend built"
echo ""

# Start with Docker Compose
echo "🐳 Starting services with Docker Compose..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 5

# Check health
echo "🔍 Checking service health..."
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "⚠️  Backend not responding yet (may still be starting)"
fi

echo ""
echo "===================================="
echo "✅ Deployment Complete!"
echo "===================================="
echo ""
echo "Access your application:"
echo "  🌐 Frontend: http://localhost"
echo "  📚 API Docs: http://localhost:8001/docs"
echo ""
echo "Manage services:"
echo "  View logs: docker-compose logs -f"
echo "  Stop all: docker-compose down"
echo "  Restart: docker-compose restart"
echo ""
