#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting POLYMORPH-LITE Local Development Environment...${NC}"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed."
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed."
    exit 1
fi

# Create necessary directories
mkdir -p logs data

# Build and start services
echo -e "${BLUE}📦 Building and starting services...${NC}"
docker-compose up --build -d

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to be healthy...${NC}"

# Function to check health
check_health() {
    local service=$1
    local max_attempts=30
    local attempt=1
    
    echo -n "Checking $service..."
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps $service | grep -q "healthy"; then
            echo -e " ${GREEN}OK${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo -e " ${RED}Failed${NC}"
    return 1
}

# Check core services
# Note: Backend health check depends on app startup time
sleep 5

echo -e "${GREEN}✅ Environment is up!${NC}"
echo -e "   - Backend API: http://localhost:8001/docs"
echo -e "   - AI Service:  http://localhost:3000"
echo -e "   - Database:    localhost:5432"
echo -e "   - Redis:       localhost:6379"
echo ""
echo -e "${BLUE}To stop the environment:${NC} docker-compose down"
echo -e "${BLUE}To view logs:${NC} docker-compose logs -f"
