#!/bin/bash
# Production Deployment Script for POLYMORPH-4 Lite
# This script performs pre-deployment checks and deploys the system

set -e  # Exit on error

echo "🚀 POLYMORPH-4 Lite Production Deployment"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running with proper permissions
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}❌ Do not run as root${NC}"
   exit 1
fi

# Check for required files
echo "📋 Checking prerequisites..."
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found${NC}"
    echo "Copy .env.production.example to .env and configure it"
    exit 1
fi

if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ docker-compose.yml not found${NC}"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon is not running${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"
echo ""

# Run tests
echo "🧪 Running test suite..."
if ! PYTHONPATH=. python -m pytest tests/ -v --tb=short; then
    echo -e "${YELLOW}⚠️  Some tests failed. Continue anyway? (y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo -e "${GREEN}✅ Tests completed${NC}"
echo ""

# Build frontend
echo "🏗️  Building frontend..."
cd gui-v2/frontend
if ! npm run build; then
    echo -e "${RED}❌ Frontend build failed${NC}"
    exit 1
fi
cd ../..
echo -e "${GREEN}✅ Frontend built${NC}"
echo ""

# Build Docker images
echo "🐳 Building Docker images..."
if ! docker-compose build; then
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker images built${NC}"
echo ""

# Check if services are already running
if docker-compose ps | grep -q "Up"; then
    echo -e "${YELLOW}⚠️  Services are already running${NC}"
    echo "Do you want to restart them? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Stopping existing services..."
        docker-compose down
    else
        echo "Deployment cancelled"
        exit 0
    fi
fi

# Start services
echo "🚀 Starting services..."
if ! docker-compose up -d; then
    echo -e "${RED}❌ Failed to start services${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Services started${NC}"
echo ""

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend is healthy${NC}"
else
    echo -e "${RED}❌ Backend health check failed${NC}"
    echo "Check logs with: docker-compose logs backend"
    exit 1
fi

if curl -f http://localhost:3000/healthz > /dev/null 2>&1; then
    echo -e "${GREEN}✅ AI Service is healthy${NC}"
else
    echo -e "${YELLOW}⚠️  AI Service health check failed${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Access your application at:"
echo "  🌐 Frontend: http://localhost:3000"
echo "  📚 API Docs: http://localhost:8001/docs"
echo "  📊 Grafana: http://localhost:3030 (admin/admin)"
echo ""
echo "Useful commands:"
echo "  View logs: docker-compose logs -f"
echo "  Stop: docker-compose down"
echo "  Restart: docker-compose restart"
echo ""
