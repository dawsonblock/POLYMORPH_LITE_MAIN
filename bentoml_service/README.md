# POLYMORPH-4 Lite BentoML Service

This directory contains the BentoML service for the POLYMORPH-4 Lite crystallization AI.

## Prerequisites

Ensure you have Python 3 installed.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Build

Build the Bento:
```bash
bentoml build
```

## Run

Serve the model locally using the helper script:
```bash
./run_service.sh
```

Or manually:
```bash
python3 -m bentoml serve .
```

The service will be available at `http://localhost:3000`.

## API

### Infer

**Endpoint:** `/infer`
**Method:** `POST`
**Content-Type:** `application/json`

**Input:**
```json
{
  "spectrum": [0.1, 0.2, ... 1024 values ...]
}
```

**Output:**
```json
{
  "active_modes": 4,
  "polymorphs_found": 0,
  "predicted_finish_sec": 1800,
  "new_polymorph": null,
  "status": "stable",
  "timestamp": "2023-10-27T10:00:00.000000"
}
```
