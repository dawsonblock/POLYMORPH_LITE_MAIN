# POLYMORPH v8.0 Developer Guide

## Architecture
- **Backend**: Python FastAPI (`retrofitkit/`)
- **Frontend**: Next.js (`ui/`)
- **AI**: Scikit-Learn / PyTorch (`ai/`)
- **Database**: PostgreSQL
- **Infrastructure**: Docker + Kubernetes

## Development Setup
1. **Prerequisites**: Docker, Python 3.11, Node.js 18.
2. **Clone Repository**:
   ```bash
   git clone https://github.com/org/polymorph.git
   cd polymorph
   ```
3. **Start Dev Environment**:
   ```bash
   docker-compose up --build
   ```
4. **Access**:
   - API Docs: `http://localhost:8000/docs`
   - UI: `http://localhost:3000`

## Adding a New Driver
1. Create `retrofitkit/drivers/my_driver.py`.
2. Implement the driver class.
3. Add tests in `tests/drivers/`.

## Adding a New Workflow Step
1. Update `retrofitkit/core/workflow/runner.py` (register action).
2. Update UI component if custom input is needed.

## Testing
- **Unit Tests**: `pytest tests/`
- **Validation**: `python validation/run_iq.py` (IQ/OQ/PQ)

## Deployment
- CI/CD via GitHub Actions (`.github/workflows/ci.yml`).
- Docker images pushed to registry.
