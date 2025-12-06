# POLYMORPH-LITE Makefile

.PHONY: all help build up down logs shell-backend shell-frontend install test

# Default target
all: help

help:
	@echo "POLYMORPH-LITE Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make build           Build all Docker images"
	@echo "  make up              Start all services"
	@echo "  make down            Stop all services"
	@echo "  make logs            Follow logs for all services"
	@echo "  make shell-backend   Open a shell in the backend container"
	@echo "  make install         Install local dependencies (Python/Node)"
	@echo "  make test            Run backend tests"
	@echo ""

# -----------------------------------------------------------------------------
# Docker Commands
# -----------------------------------------------------------------------------

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

shell-backend:
	docker compose exec backend /bin/bash

# -----------------------------------------------------------------------------
# Local Development
# -----------------------------------------------------------------------------

venv:
	python3 -m venv .venv
	@echo "Run 'source .venv/bin/activate' to activate virtual environment."

install: venv
	./scripts/setup_dev.sh

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .venv
	rm -rf ui/.next ui/node_modules
