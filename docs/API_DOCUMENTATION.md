# POLYMORPH-4 Lite API Documentation

**Version**: 2.0.0  
**Last Updated**: November 25, 2025  
**Base URL**: `http://localhost:8001`

---

## Table of Contents
1. [Introduction](#introduction)
2. [Authentication](#authentication)
3. [Health & Monitoring](#health--monitoring)
4. [Experiment Management](#experiment-management)
5. [Data Access](#data-access)
6. [WebSocket Events](#websocket-events)
7. [AI Service API](#ai-service-api)
8. [Error Codes](#error-codes)
9. [SDK Examples](#sdk-examples)

---

## Introduction

The POLYMORPH-4 Lite API is a RESTful API built with FastAPI. All endpoints return JSON responses and support standard HTTP methods.

### Base URL
```
Development: http://localhost:8001
Production: https://api.polymorph.lab
```

### API Versioning
All endpoints are prefixed with `/api/v1/`.

### Rate Limiting
- 100 requests/minute per user
- 1000 requests/hour per IP address

---

## Authentication

### Login
**POST** `/api/v1/auth/login`

Authenticate and receive JWT token.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "secret",
  "mfa_code": "123456"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "role": "operator",
    "name": "John Doe"
  }
}
```

### Refresh Token
**POST** `/api/v1/auth/refresh`

Get a new access token using refresh token.

**Headers:**
```
Authorization: Bearer <refresh_token>
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600
}
```

### Logout
**POST** `/api/v1/auth/logout`

Invalidate current session.

**Headers:**
```
Authorization: Bearer <access_token>
```

---

## Health & Monitoring

### Basic Health Check
**GET** `/health`

Simple liveness check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-25T16:05:48Z",
  "uptime_seconds": 86400.5,
  "version": "2.0.0",
  "environment": "production"
}
```

### Component Health
**GET** `/health/components`

Detailed status of all system components.

**Response:**
```json
[
  {
    "name": "DAQ Driver",
    "status": "healthy",
    "last_check": "2025-11-25T16:05:48Z",
    "response_time_ms": 12.5,
    "details": {
      "last_voltage": 1.23
    }
  },
  {
    "name": "Raman Driver",
    "status": "healthy",
    "last_check": "2025-11-25T16:05:48Z",
    "response_time_ms": 45.2,
    "details": {
      "last_peak_nm": 532.0
    }
  },
  {
    "name": "AI Service",
    "status": "healthy",
    "last_check": "2025-11-25T16:05:48Z",
    "response_time_ms": 150.0,
    "details": {
      "url": "http://localhost:3000/infer"
    }
  }
]
```

### System Diagnostics
**GET** `/health/diagnostics`

Comprehensive system information.

**Response:**
```json
{
  "system_info": {
    "hostname": "polymorph-server",
    "platform": "Linux",
    "python_version": "3.11.5",
    "cpu_count": 8,
    "total_memory_gb": 32.0
  },
  "performance_metrics": {
    "cpu_percent": 25.3,
    "memory_percent": 42.1,
    "disk_usage_percent": 65.8,
    "uptime_hours": 240.5
  },
  "hardware_status": {
    "daq": {
      "backend": "gamry",
      "status": "connected"
    },
    "raman": {
      "provider": "horiba",
      "status": "connected"
    }
  }
}
```

---

## Experiment Management

### List Experiments
**GET** `/api/v1/experiments`

Get all experiments with optional filtering.

**Query Parameters:**
- `status` (optional): `running`, `completed`, `failed`
- `user_id` (optional): Filter by user
- `from_date` (optional): ISO 8601 timestamp
- `to_date` (optional): ISO 8601 timestamp
- `page` (default: 1)
- `per_page` (default: 20, max: 100)

**Response:**
```json
{
  "experiments": [
    {
      "id": "exp_abc123",
      "name": "Aspirin Crystallization",
      "status": "running",
      "recipe_id": "recipe_xyz",
      "started_at": "2025-11-25T15:00:00Z",
      "current_step": 2,
      "total_steps": 5,
      "progress": 40.0,
      "user": {
        "id": "user_123",
        "name": "John Doe"
      }
    }
  ],
  "total": 42,
  "page": 1,
  "per_page": 20
}
```

### Get Experiment
**GET** `/api/v1/experiments/{experiment_id}`

Get detailed information about a specific experiment.

**Response:**
```json
{
  "id": "exp_abc123",
  "name": "Aspirin Crystallization",
  "description": "Form I to Form II transformation study",
  "status": "running",
  "recipe": {
    "id": "recipe_xyz",
    "name": "Aspirin Recipe v2",
    "steps": [...]
  },
  "started_at": "2025-11-25T15:00:00Z",
  "current_step": 2,
  "total_steps": 5,
  "progress": 40.0,
  "parameters": {
    "temperature_setpoint": 60.0,
    "flow_rate": 2.0,
    "pressure_max": 2.5
  },
  "realtime_data": {
    "temperature": 59.8,
    "pressure": 1.5,
    "flow_rate": 2.0
  },
  "ai_insights": {
    "polymorphs_detected": ["Form I"],
    "confidence": 0.95,
    "last_analysis": "2025-11-25T16:05:48Z"
  }
}
```

### Start Experiment
**POST** `/api/v1/experiments`

Create and start a new experiment.

**Request:**
```json
{
  "recipe_id": "recipe_xyz",
  "name": "Aspirin Study #42",
  "description": "Testing new cooling protocol",
  "parameters": {
    "temperature_setpoint": 65.0,
    "flow_rate": 2.5
  },
  "signature": {
    "username": "john.doe",
    "password": "secret",
    "reason": "Protocol validation experiment"
  }
}
```

**Response:**
```json
{
  "id": "exp_new123",
  "status": "running",
  "started_at": "2025-11-25T16:10:00Z",
  "audit_log_id": "audit_log_456"
}
```

### Stop Experiment
**POST** `/api/v1/experiments/{experiment_id}/stop`

Stop a running experiment.

**Request:**
```json
{
  "reason": "Unexpected color change observed",
  "signature": {
    "username": "john.doe",
    "password": "secret"
  }
}
```

**Response:**
```json
{
  "status": "stopped",
  "stopped_at": "2025-11-25T16:15:00Z",
  "final_data_path": "/data/experiments/exp_abc123/final.csv"
}
```

### Emergency Stop
**POST** `/api/v1/experiments/{experiment_id}/emergency_stop`

Emergency shutdown of experiment.

**Request:**
```json
{
  "reason": "Safety hazard detected",
  "signature": {
    "username": "john.doe",
    "password": "secret"
  }
}
```

**Response:**
```json
{
  "status": "emergency_stopped",
  "stopped_at": "2025-11-25T16:16:00Z",
  "safety_protocols_triggered": ["pump_shutdown", "cooling_max"]
}
```

---

## Data Access

### Get Spectral Data
**GET** `/api/v1/experiments/{experiment_id}/spectral`

Retrieve Raman spectral data.

**Query Parameters:**
- `from_time` (optional): ISO 8601 timestamp
- `to_time` (optional): ISO 8601 timestamp
- `format` (optional): `json`, `csv`, `numpy`

**Response (JSON):**
```json
{
  "experiment_id": "exp_abc123",
  "data_points": 150,
  "wavelength_range": [200, 3500],
  "spectra": [
    {
      "timestamp": "2025-11-25T16:00:00Z",
      "wavelengths": [200, 201, 202, ...],
      "intensities": [0.12, 0.15, 0.14, ...]
    }
  ]
}
```

### Get Time-Series Data
**GET** `/api/v1/experiments/{experiment_id}/timeseries`

Retrieve sensor time-series data (temperature, pressure, flow).

**Query Parameters:**
- `sensors` (optional): Comma-separated list (e.g., `temperature,pressure`)
- `from_time` (optional)
- `to_time` (optional)
- `downsample` (optional): Seconds between points

**Response:**
```json
{
  "experiment_id": "exp_abc123",
  "sensors": {
    "temperature": {
      "unit": "°C",
      "data": [
        {"timestamp": "2025-11-25T16:00:00Z", "value": 60.2},
        {"timestamp": "2025-11-25T16:00:10Z", "value": 60.3}
      ]
    },
    "pressure": {
      "unit": "bar",
      "data": [...]
    }
  }
}
```

### Export Experiment Data
**POST** `/api/v1/experiments/{experiment_id}/export`

Generate and download complete experiment data package.

**Request:**
```json
{
  "format": "pdf",
  "include": [
    "spectral_data",
    "time_series",
    "ai_analysis",
    "audit_trail"
  ],
  "signature": {
    "username": "john.doe",
    "password": "secret",
    "reason": "Regulatory submission"
  }
}
```

**Response:**
```json
{
  "export_id": "export_789",
  "status": "processing",
  "estimated_completion": "2025-11-25T16:20:00Z"
}
```

**Download Link:**
```
GET /api/v1/exports/{export_id}/download
```

---

## WebSocket Events

### Connect to Real-Time Stream
```javascript
const socket = io('http://localhost:8001', {
  auth: {
    token: 'your_jwt_token'
  }
});
```

### Subscribe to Experiment Updates
```javascript
socket.emit('subscribe_experiment', {
  experiment_id: 'exp_abc123'
});

socket.on('experiment_update', (data) => {
  console.log('Temperature:', data.temperature);
  console.log('Progress:', data.progress);
});
```

### Event Types

#### `status_update`
System status changed.
```json
{
  "type": "status_update",
  "component": "daq",
  "status": "healthy",
  "timestamp": "2025-11-25T16:05:48Z"
}
```

#### `spectral_data`
New Raman spectrum available.
```json
{
  "type": "spectral_data",
  "experiment_id": "exp_abc123",
  "timestamp": "2025-11-25T16:05:48Z",
  "wavelengths": [...],
  "intensities": [...]
}
```

#### `ai_alert`
AI detected important event.
```json
{
  "type": "ai_alert",
  "experiment_id": "exp_abc123",
  "alert_type": "new_polymorph",
  "confidence": 0.97,
  "message": "Form II crystallization detected",
  "timestamp": "2025-11-25T16:05:48Z"
}
```

#### `parameter_change`
Experiment parameter updated.
```json
{
  "type": "parameter_change",
  "experiment_id": "exp_abc123",
  "parameter": "temperature_setpoint",
  "old_value": 60.0,
  "new_value": 65.0,
  "user_id": "user_123",
  "timestamp": "2025-11-25T16:05:48Z"
}
```

---

## AI Service API

### Inference Endpoint
**POST** `http://localhost:3000/infer`

Send spectrum to AI model for analysis.

**Request:**
```json
{
  "spectrum": [0.12, 0.15, 0.14, ...]
}
```

**Response:**
```json
{
  "status": "crystallizing",
  "polymorphs_found": ["Form I", "Form II"],
  "active_modes": [520, 1050, 1640],
  "confidence": 0.95,
  "alerts": [
    {
      "type": "new_polymorph",
      "message": "Form II detected at 45% confidence"
    }
  ],
  "inference_time_ms": 45.2
}
```

### Health Check
**GET** `http://localhost:3000/healthz`

Check AI service availability.

**Response:**
```
OK
```

---

## Error Codes

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing token |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - System overloaded |

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_SIGNATURE",
    "message": "Electronic signature validation failed",
    "details": {
      "field": "signature.password",
      "reason": "Incorrect password"
    },
    "timestamp": "2025-11-25T16:05:48Z",
    "request_id": "req_xyz789"
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_TOKEN` | JWT token expired or invalid |
| `INVALID_SIGNATURE` | Electronic signature failed validation |
| `EXPERIMENT_RUNNING` | Cannot start - experiment already in progress |
| `HARDWARE_NOT_READY` | Hardware component not responsive |
| `AI_SERVICE_UNAVAILABLE` | Cannot reach AI inference service |
| `INSUFFICIENT_PERMISSIONS` | User lacks required role |
| `RATE_LIMIT_EXCEEDED` | Too many requests |

---

## SDK Examples

### Python SDK

**Installation:**
```bash
pip install polymorph-client
```

**Usage:**
```python
from polymorph_client import PolymorphClient

# Initialize client
client = PolymorphClient(
    base_url="http://localhost:8001",
    email="user@example.com",
    password="secret"
)

# Start experiment
experiment = client.experiments.create(
    recipe_id="recipe_xyz",
    name="My Experiment",
    signature={
        "username": "john.doe",
        "password": "secret",
        "reason": "Research study"
    }
)

# Monitor progress
while experiment.status == "running":
    data = experiment.get_realtime_data()
    print(f"Temperature: {data.temperature}°C")
    print(f"Progress: {experiment.progress}%")
    time.sleep(10)

# Get results
spectra = experiment.get_spectral_data()
export = experiment.export(format="pdf")
export.download("experiment_report.pdf")
```

### JavaScript/TypeScript SDK

**Installation:**
```bash
npm install @polymorph/client
```

**Usage:**
```typescript
import { PolymorphClient } from '@polymorph/client';

// Initialize client
const client = new PolymorphClient({
  baseURL: 'http://localhost:8001',
  email: 'user@example.com',
  password: 'secret'
});

// Start experiment
const experiment = await client.experiments.create({
  recipeId: 'recipe_xyz',
  name: 'My Experiment',
  signature: {
    username: 'john.doe',
    password: 'secret',
    reason: 'Research study'
  }
});

// Real-time monitoring
client.on('spectral_data', (data) => {
  console.log('New spectrum:', data);
});

// Get results
const spectralData = await experiment.getSpectralData();
const export = await experiment.export({ format: 'pdf' });
```

### cURL Examples

**Login:**
```bash
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "secret"
  }'
```

**Get Experiments:**
```bash
curl -X GET http://localhost:8001/api/v1/experiments \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Start Experiment:**
```bash
curl -X POST http://localhost:8001/api/v1/experiments \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "recipe_id": "recipe_xyz",
    "name": "Test Experiment",
    "signature": {
      "username": "john.doe",
      "password": "secret",
      "reason": "Testing"
    }
  }'
```

---

## Rate Limiting

API responses include rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1638720000
```

When limit is exceeded:
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests",
    "retry_after": 60
  }
}
```

---

## Pagination

List endpoints support pagination:

**Request:**
```
GET /api/v1/experiments?page=2&per_page=50
```

**Response Headers:**
```
X-Total-Count: 420
X-Page: 2
X-Per-Page: 50
Link: <http://localhost:8001/api/v1/experiments?page=3>; rel="next",
      <http://localhost:8001/api/v1/experiments?page=9>; rel="last"
```

---

## Webhooks

Configure webhooks for automated notifications:

**Setup:**
```bash
POST /api/v1/webhooks
{
  "url": "https://your-server.com/webhook",
  "events": ["experiment_started", "experiment_completed", "ai_alert"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload:**
```json
{
  "event": "experiment_completed",
  "experiment_id": "exp_abc123",
  "timestamp": "2025-11-25T18:00:00Z",
  "data": {
    "duration_minutes": 120,
    "polymorphs_detected": ["Form I", "Form II"]
  },
  "signature": "sha256=abc123..."
}
```

---

**© 2025 POLYMORPH-4 Research Team. All Rights Reserved.**
