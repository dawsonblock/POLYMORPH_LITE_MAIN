# Developer Setup Guide

Quick setup guide for POLYMORPH-LITE development.

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ (or use Docker Compose)
- Node.js 18+ (for frontend)

## Backend Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install in Editable Mode

```bash
pip install -e .[dev]
```

This installs:
- Core dependencies
- Dev dependencies (pytest, black, mypy, etc.)
- Package in editable mode for import

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings:
# - DATABASE_URL (postgres://...)
# - SECRET_KEY
# - JWT_SECRET_KEY
```

### 4. Database Migrations

```bash
# Run migrations
alembic upgrade head

# Create admin user
python scripts/create_admin_user.py
```

### 5. Start Development Server

```bash
uvicorn retrofitkit.api.server:app --reload --host 0.0.0.0 --port 8001
```

API will be available at: `http://localhost:8001`

---

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at: `http://localhost:5173`

---

## Docker Compose (All-in-One)

```bash
docker-compose up -d
```

Includes:
- PostgreSQL database
- Backend API
- Frontend (if configured)

---

## Running Tests

### Backend Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=retrofitkit --cov-report=html

# Specific test file
pytest tests/test_api_samples.py

# With verbose output
pytest -v
```

### Frontend Tests

```bash
cd frontend

# Unit tests
npm test

# E2E tests (requires backend running)
npm run test:e2e
```

---

## Common Development Tasks

### Check Code Quality

```bash
# Format code
black retrofitkit/ tests/

# Type checking
mypy retrofitkit/

# Linting
flake8 retrofitkit/
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

### View API Docs

Start server and visit:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

---

## Troubleshooting

### Import Errors

If `import retrofitkit` fails:
```bash
pip install -e .
```

### Database Connection Errors

Check:
1. PostgreSQL is running
2. `DATABASE_URL` in `.env` is correct
3. Database exists: `createdb polymorph_lite`

### Port Already in Use

```bash
# Find process using port 8001
lsof -i :8001

# Kill it
kill -9 <PID>
```

---

## Project Structure

```
retrofitkit/          # Main package
├── api/             # FastAPI endpoints
├── core/            # Business logic
├── db/              # Database models
├── drivers/         # Hardware drivers
└── compliance/      # Audit, signatures

tests/               # Test suite
frontend/     # React frontend
scripts/             # Utility scripts
alembic/             # Database migrations
```

---

## Next Steps

1. Read `STRUCTURE.md` for architecture overview
2. Check `TESTING.md` for testing guidelines
3. See `DEPLOYMENT_GUIDE.md` for production deployment
4. Review `SECURITY.md` for security best practices
