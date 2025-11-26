# POLYMORPH-4 Lite - Makefile
# Simplified development and testing workflows

.PHONY: help dev test test-fast test-integration clean install-deps run-ai run-backend run-gui docker-build docker-up docker-down

help:
	@echo "POLYMORPH-4 Lite - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make dev              - Create venv and install all dependencies"
	@echo "  make install-deps     - Install dependencies only (venv must exist)"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests (skip integration tests)"
	@echo "  make test-fast        - Run fast unit tests only"
	@echo "  make test-integration - Run integration tests (requires AI service)"
	@echo ""
	@echo "Running:"
	@echo "  make run-ai           - Start BentoML AI service (port 3000)"
	@echo "  make run-backend      - Start FastAPI backend (port 8001)"
	@echo "  make run-gui          - Start GUI dev server (port 5173)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker images"
	@echo "  make docker-up        - Start all services in Docker"
	@echo "  make docker-down      - Stop all Docker services"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove build artifacts and caches"

# Development environment setup
dev:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Installing dependencies..."
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -e .
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -r bentoml_service/requirements.txt
	@echo "✅ Development environment ready!"
	@echo "Activate with: source .venv/bin/activate"

install-deps:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -e .
	pip install -r requirements.txt
	pip install -r bentoml_service/requirements.txt
	@echo "✅ Dependencies installed!"

# Testing
test:
	@echo "Running tests (excluding integration)..."
	pytest -v -m "not integration"

test-fast:
	@echo "Running fast unit tests..."
	pytest -v tests/test_drivers.py tests/test_pmm_brain.py tests/test_raman_preprocessor.py

test-integration:
	@echo "Running integration tests (AI service required)..."
	P4_RUN_AI_INTEGRATION=1 pytest -v -m integration

# Running services
run-ai:
	@echo "Starting BentoML AI service on port 3000..."
	cd bentoml_service && bentoml serve service:svc --reload --port 3000

run-backend:
	@echo "Starting FastAPI backend on port 8001..."
	uvicorn retrofitkit.api.server:app --reload --host 0.0.0.0 --port 8001

run-gui:
	@echo "Starting GUI frontend on port 5173..."
	cd gui-v2/frontend && npm run dev

# Docker workflows
docker-build:
	@echo "Building Docker images..."
	docker compose -f docker-compose.yml build

docker-up:
	@echo "Starting all services with Docker Compose..."
	docker compose -f docker-compose.yml -f docker-compose.ai.yml up -d
	@echo "✅ Services started!"
	@echo "Backend: http://localhost:8001"
	@echo "AI Service: http://localhost:3000"
	@echo "Frontend: http://localhost:80"

docker-down:
	@echo "Stopping Docker services..."
	docker compose -f docker-compose.yml -f docker-compose.ai.yml down

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "✅ Cleanup complete!"
