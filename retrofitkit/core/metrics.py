"""
Custom Prometheus metrics for POLYMORPH-LITE.

Provides business-specific metrics for monitoring and alerting.
"""
from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Optional
import time


# Workflow metrics
workflow_runs_total = Counter(
    'polymorph_workflow_runs_total',
    'Total number of workflow runs',
    ['org_id', 'workflow_name', 'status']
)

workflow_step_duration_seconds = Histogram(
    'polymorph_workflow_step_duration_seconds',
    'Duration of workflow steps in seconds',
    ['org_id', 'workflow_name', 'step_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

workflow_execution_duration_seconds = Histogram(
    'polymorph_workflow_execution_duration_seconds',
    'Total workflow execution duration in seconds',
    ['org_id', 'workflow_name'],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0]
)

# Device metrics
device_errors_total = Counter(
    'polymorph_device_errors_total',
    'Total number of device errors',
    ['device_type', 'device_id', 'error_type']
)

device_status = Gauge(
    'polymorph_device_status',
    'Device status (1=online, 0=offline)',
    ['device_type', 'device_id']
)

device_health_score = Gauge(
    'polymorph_device_health_score',
    'Device health score (0-1)',
    ['device_type', 'device_id']
)

device_calibration_days_since = Gauge(
    'polymorph_device_calibration_days_since',
    'Days since last calibration',
    ['device_type', 'device_id']
)

# AI service metrics
ai_inference_duration_seconds = Histogram(
    'polymorph_ai_inference_duration_seconds',
    'AI inference duration in seconds',
    ['model_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

ai_confidence_score = Histogram(
    'polymorph_ai_confidence_score',
    'AI prediction confidence scores',
    ['model_name'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

ai_circuit_breaker_state = Gauge(
    'polymorph_ai_circuit_breaker_state',
    'AI circuit breaker state (0=closed, 1=open)',
    []
)

# Sample metrics
samples_created_total = Counter(
    'polymorph_samples_created_total',
    'Total number of samples created',
    ['org_id', 'project_id']
)

samples_active = Gauge(
    'polymorph_samples_active',
    'Number of active samples',
    ['org_id']
)

# API metrics
api_requests_total = Counter(
    'polymorph_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

api_request_duration_seconds = Histogram(
    'polymorph_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Rate limiting metrics
rate_limit_exceeded_total = Counter(
    'polymorph_rate_limit_exceeded_total',
    'Total number of rate limit violations',
    ['endpoint', 'user_type']
)

# Audit metrics
audit_events_total = Counter(
    'polymorph_audit_events_total',
    'Total audit events logged',
    ['org_id', 'event_type']
)

# System info
system_info = Info(
    'polymorph_system',
    'POLYMORPH-LITE system information'
)


class MetricsCollector:
    """Helper class for collecting metrics."""
    
    @staticmethod
    def record_workflow_run(org_id: str, workflow_name: str, status: str):
        """Record a workflow run."""
        workflow_runs_total.labels(
            org_id=org_id,
            workflow_name=workflow_name,
            status=status
        ).inc()
        
    @staticmethod
    def record_workflow_step(org_id: str, workflow_name: str, step_name: str, duration: float):
        """Record workflow step duration."""
        workflow_step_duration_seconds.labels(
            org_id=org_id,
            workflow_name=workflow_name,
            step_name=step_name
        ).observe(duration)
        
    @staticmethod
    def record_workflow_execution(org_id: str, workflow_name: str, duration: float):
        """Record total workflow execution duration."""
        workflow_execution_duration_seconds.labels(
            org_id=org_id,
            workflow_name=workflow_name
        ).observe(duration)
        
    @staticmethod
    def record_device_error(device_type: str, device_id: str, error_type: str):
        """Record a device error."""
        device_errors_total.labels(
            device_type=device_type,
            device_id=device_id,
            error_type=error_type
        ).inc()
        
    @staticmethod
    def set_device_status(device_type: str, device_id: str, online: bool):
        """Set device online/offline status."""
        device_status.labels(
            device_type=device_type,
            device_id=device_id
        ).set(1 if online else 0)
        
    @staticmethod
    def set_device_health(device_type: str, device_id: str, health_score: float):
        """Set device health score."""
        device_health_score.labels(
            device_type=device_type,
            device_id=device_id
        ).set(health_score)
        
    @staticmethod
    def record_ai_inference(model_name: str, duration: float, confidence: float):
        """Record AI inference metrics."""
        ai_inference_duration_seconds.labels(model_name=model_name).observe(duration)
        ai_confidence_score.labels(model_name=model_name).observe(confidence)
        
    @staticmethod
    def set_circuit_breaker_state(is_open: bool):
        """Set AI circuit breaker state."""
        ai_circuit_breaker_state.set(1 if is_open else 0)
        
    @staticmethod
    def record_sample_created(org_id: str, project_id: str):
        """Record sample creation."""
        samples_created_total.labels(
            org_id=org_id,
            project_id=project_id
        ).inc()
        
    @staticmethod
    def record_api_request(method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics."""
        api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        api_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
    @staticmethod
    def record_rate_limit_exceeded(endpoint: str, user_type: str):
        """Record rate limit violation."""
        rate_limit_exceeded_total.labels(
            endpoint=endpoint,
            user_type=user_type
        ).inc()
        
    @staticmethod
    def record_audit_event(org_id: str, event_type: str):
        """Record audit event."""
        audit_events_total.labels(
            org_id=org_id,
            event_type=event_type
        ).inc()


# Global metrics collector instance
metrics = MetricsCollector()
