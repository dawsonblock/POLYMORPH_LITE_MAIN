# POLYMORPH v8.1 Validation Execution Guide

This guide details how to execute the Installation Qualification (IQ), Operational Qualification (OQ), and Performance Qualification (PQ) scripts for the POLYMORPH-LITE system.

## Prerequisites

- **Python 3.11+** installed.
- **Virtual Environment** activated with dependencies installed (`pip install -r requirements.txt`).
- **Docker** and **Docker Compose** running (for PQ integration tests).
- **Hardware Drivers** configured (or Simulation Mode enabled).

## 1. Installation Qualification (IQ)

Verifies that the software is correctly installed, directory structures are intact, and dependencies are met.

### Execution
```bash
python validation/run_iq.py
```

### Expected Output
- **Pass**: "✅ IQ PASSED: System environment is valid."
- **Fail**: "❌ IQ FAILED" with specific missing files or packages listed.

### Artifacts
- None generated (console output only).

## 2. Operational Qualification (OQ)

Verifies that the hardware drivers (Ocean Optics, Red Pitaya) and the Workflow Engine are functional.

### Execution
```bash
python validation/run_oq.py
```

### Configuration
- By default, this runs in **Simulation Mode** for hardware.
- To run with real hardware, ensure devices are connected and configured in `config/config.yaml` or via environment variables.

### Expected Output
- **Pass**: "✅ OQ PASSED: System functions correctly."
- **Fail**: "❌ OQ FAILED" with specific driver or workflow errors.

## 3. Performance Qualification (PQ)

Verifies the end-to-end performance of the system, including the Unified DAQ Pipeline and AI Inference.

### Execution
```bash
python validation/run_pq.py
```

### Description
1.  Initializes the `UnifiedDAQPipeline`.
2.  Executes a synchronized acquisition (DAQ + Raman).
3.  Sends data to the AI Service for inference.
4.  Validates that results are within expected ranges.

### Artifacts
Generated in `validation/pq_output/`:
- `PQ_Test_Sample_<timestamp>.json`: Metadata and analysis results.
- `PQ_Test_Sample_<timestamp>_daq.csv`: Raw DAQ voltage trace.
- `PQ_Test_Sample_<timestamp>_raman.csv`: Raw Raman spectrum.

### Expected Output
- **Pass**: "✅ PQ PASSED: End-to-end pipeline verified."
- **Fail**: "❌ PQ FAILED" with details on pipeline or AI errors.

## Troubleshooting

- **Connection Refused**: Ensure the backend and AI service are running (`docker-compose up -d`).
- **Driver Error**: Check USB connections or network reachability for Red Pitaya. Verify `P4_DAQ_BACKEND` env var.
- **AI Timeout**: Ensure the BentoML service is healthy (`curl http://localhost:3000/healthz`).
