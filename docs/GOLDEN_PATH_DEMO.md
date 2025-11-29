# Golden Path Demo - Quick Start Guide

## Overview

The Golden Path Demo demonstrates a complete laboratory workflow from sample creation through AI-driven analysis and reporting. This is the **"hero demo"** that shows POLYMORPH-LITE's full capabilities in one 60-second execution.

## What It Demonstrates

1. **Authentication** - JWT-based user login
2. **LIMS** - Project and sample creation with metadata
3. **Workflow Execution** - Raman spectrum acquisition
4. **AI Integration** - Polymorph classification with confidence scoring
5. **Compliance** - Audit trail generation and reporting
6. **End-to-End Traceability** - Sample → Data → Decision → Report

## Prerequisites

### Services Running

```bash
# Terminal 1: Backend
uvicorn retrofitkit.api.server:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2: Frontend (optional)
cd frontend && npm run dev
```

### Database Initialized

```bash
# Run migrations
alembic upgrade head

# Verify health
curl http://localhost:8001/health
```

## Running the Demo

### One-Command Execution

```bash
./scripts/run_hero_demo.sh
```

### Expected Output

```
╔════════════════════════════════════════════════════════════╗
║  POLYMORPH-LITE Golden Path Demo                          ║
║  Sample → Raman → AI Decision → Report                    ║
╚════════════════════════════════════════════════════════════╝

[1/7] Checking services...
✓ Backend running
✓ Frontend running

[2/7] Authenticating...
✓ Authenticated as demo@polymorph.local

[3/7] Creating demo project and sample...
✓ Project created: DEMO-2024-001
✓ Sample created: SAMPLE-DEMO-001

[4/7] Executing Raman acquisition workflow...
✓ Workflow started: run_abc123
   Waiting for workflow completion...
✓ Workflow completed successfully

[5/7] Retrieving AI decision results...

============================================================
WORKFLOW EXECUTION RESULTS
============================================================
Run ID: run_abc123
Status: completed
Started: 2024-11-28T21:00:00Z
Completed: 2024-11-28T21:01:00Z

AI Decision:
  Polymorph: Form II
  Confidence: 92.5%
  Classification: crystalline
============================================================

[6/7] Generating compliance report...

============================================================
AUDIT TRAIL (Last 10 Events)
============================================================
2024-11-28T21:01:00Z | WORKFLOW_COMPLETED
  Actor: demo@polymorph.local
  Subject: run_abc123

2024-11-28T21:00:30Z | AI_DECISION
  Actor: ai_service
  Subject: SAMPLE-DEMO-001

2024-11-28T21:00:15Z | SPECTRUM_ACQUIRED
  Actor: demo@polymorph.local
  Subject: SAMPLE-DEMO-001
============================================================

[7/7] Demo summary...

╔════════════════════════════════════════════════════════════╗
║  ✓ GOLDEN PATH DEMO COMPLETE                              ║
╚════════════════════════════════════════════════════════════╝

What was demonstrated:
  1. ✓ User authentication with JWT
  2. ✓ Project and sample creation (LIMS)
  3. ✓ Workflow execution (Raman acquisition)
  4. ✓ AI-driven decision making
  5. ✓ Audit trail generation
  6. ✓ Compliance reporting

Next steps:
  • View frontend: http://localhost:3001
  • Check API docs: http://localhost:8001/docs
  • Review audit logs in database
  • Explore workflow definitions in workflows/

For production deployment, see: DEPLOYMENT_GUIDE.md
```

## What Happens Behind the Scenes

### 1. Authentication (Step 2)

- Creates demo user if not exists
- Logs in with credentials
- Obtains JWT token for subsequent requests

### 2. Sample Creation (Step 3)

- Creates project: `DEMO-2024-001`
- Creates sample: `SAMPLE-DEMO-001` with metadata:
  - Compound: Aspirin
  - Target polymorph: Form II
  - Temperature: 25°C
  - Solvent: Ethanol

### 3. Workflow Execution (Step 4)

Executes `hero_crystallization` workflow:

1. Initialize Raman spectrometer (simulation mode)
2. Acquire baseline spectrum
3. Wait for thermal stabilization (5s)
4. Acquire sample spectrum (5 averages)
5. Send to AI service for classification
6. Generate report

### 4. AI Classification (Step 5)

- Preprocesses Raman spectrum
- Runs through trained model
- Returns:
  - Polymorph ID (e.g., "Form II")
  - Confidence score (0-1)
  - Classification (crystalline/amorphous)

### 5. Audit Trail (Step 6)

Every action is logged:
- User authentication
- Sample creation
- Workflow start/stop
- AI decisions
- Report generation

## Customization

### Change Sample Parameters

Edit the sample creation payload in `run_hero_demo.sh`:

```json
{
  "sample_id": "YOUR-SAMPLE-ID",
  "metadata": {
    "compound": "Your Compound",
    "temperature_c": 30,
    "solvent": "methanol"
  }
}
```

### Change Workflow Parameters

Edit the workflow execution payload:

```json
{
  "workflow_name": "hero_crystallization",
  "parameters": {
    "integration_time_ms": 2000,
    "laser_power_mw": 75
  }
}
```

### Use Real Hardware

1. Set `SIMULATION_MODE=false` in `.env`
2. Install vendor SDKs (see `docs/HARDWARE_INTEGRATION.md`)
3. Configure device connections
4. Run integration tests first

## Troubleshooting

### Backend Not Running

```bash
# Check if port 8001 is in use
lsof -i :8001

# Start backend
uvicorn retrofitkit.api.server:app --reload
```

### Authentication Failed

```bash
# Reset demo user
python3 scripts/create_admin_user.py --email demo@polymorph.local --password demo123
```

### Workflow Fails

```bash
# Check logs
tail -f logs/polymorph.log

# Verify devices are registered
curl http://localhost:8001/api/devices
```

### Database Issues

```bash
# Run migrations
alembic upgrade head

# Check database connection
python3 scripts/check_db_health.py
```

## Next Steps

### For Development

1. Explore API documentation: http://localhost:8001/docs
2. Review workflow definitions in `workflows/`
3. Add custom device drivers in `retrofitkit/drivers/`
4. Extend AI models in `bentoml_service/`

### For Production

1. Complete validation package (IQ/OQ/PQ)
2. Configure TLS/SSL
3. Set up monitoring (Prometheus + Grafana)
4. Review security checklist
5. See `DEPLOYMENT_GUIDE.md`

### For Lab Pilots

1. Define your specific use case
2. Configure real hardware
3. Train AI models on your data
4. Run validation protocols
5. Generate compliance documentation

## Support

- **Documentation**: `docs/`
- **API Reference**: http://localhost:8001/docs
- **Hardware Integration**: `docs/HARDWARE_INTEGRATION.md`
- **Deployment**: `DEPLOYMENT_GUIDE.md`
- **Security**: `SECURITY.md`

## License

MIT License - See LICENSE file for details.
