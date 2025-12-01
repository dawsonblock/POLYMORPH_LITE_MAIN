#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🎬 Starting POLYMORPH v8.1 Full Demo...${NC}"

# 1. Start Environment
echo -e "\n${BLUE}[1/4] Starting Local Environment...${NC}"
./scripts/dev_runner.sh

# Wait for health
echo "Waiting for API to be ready..."
sleep 5
until curl -s -f http://localhost:8001/health > /dev/null; do
    echo -n "."
    sleep 2
done
echo -e " ${GREEN}OK${NC}"

# 2. Run Validation Suite
echo -e "\n${BLUE}[2/4] Running Validation Suite (IQ/OQ/PQ)...${NC}"

echo "Running IQ..."
python validation/run_iq.py || { echo "IQ Failed"; exit 1; }

echo "Running OQ..."
python validation/run_oq.py || { echo "OQ Failed"; exit 1; }

echo "Running PQ..."
# PQ requires AI service, which is running in docker
# We need to ensure the script can reach it.
# If running on host, localhost:3000 should work.
export P4_AI_SERVICE_URL="http://localhost:3000"
python validation/run_pq.py || { echo "PQ Failed"; exit 1; }

# 3. Simulate Workflow Activity
echo -e "\n${BLUE}[3/4] Simulating Workflow Activity...${NC}"
# We can use curl to trigger a workflow via the API
# First, login to get token (if auth enabled) - skipping for demo simplicity or using default
# Assuming dev mode allows open access or we use a helper script.
# For now, we'll just print instructions.

echo "Workflow simulation: Please open the UI to run a workflow manually."

# 4. Open UI
echo -e "\n${BLUE}[4/4] Opening User Interface...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:3000
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:3000
fi

echo -e "\n${GREEN}✅ Demo Environment Ready!${NC}"
echo "Dashboard: http://localhost:3000"
echo "API Docs:  http://localhost:8001/docs"
echo "Press Ctrl+C to stop."

# Keep alive to show logs or just exit?
# Let's tail logs
docker-compose logs -f backend
