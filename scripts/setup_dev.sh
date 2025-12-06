#!/bin/bash
# scripts/setup_dev.sh
# Automates local development environment setup

set -e

# Colors
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up POLYMORPH-LITE development environment...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed."
    exit 1
fi

# Check Node
if ! command -v npm &> /dev/null; then
    echo "Node.js/npm is not installed."
    exit 1
fi

# 1. Python Venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 2. Install Python Deps
echo "Installing backend dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r tests/requirements-test.txt 2>/dev/null || true

# 3. Install Node Deps
echo "Installing frontend dependencies..."
cd ui
npm install
cd ..

echo -e "${GREEN}Setup complete!${NC}"
echo "Activate venv with: source .venv/bin/activate"
