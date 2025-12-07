"""
Prometheus Metrics for POLYMORPH-LITE.

Exposes key metrics for monitoring:
- AI mode statistics
- Polymorph discovery rate
- Gating events
- Safety trips
- Workflow performance
"""

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/metrics", tags=["metrics"])

# ============================================================================
# AI / PMM Metrics
# ============================================================================

pmm_active_modes = Gauge(
    "polymorph_pmm_active_modes",
    "Number of currently active PMM modes",
    ["org_id"]
)

pmm_max_modes = Gauge(
    "polymorph_pmm_max_modes",
    "Maximum number of PMM modes",
    ["org_id"]
)

polymorphs_detected_total = Counter(
    "polymorph_polymorphs_detected_total",
    "Total number of polymorphs detected",
    ["org_id", "device_id"]
)

polymorphs_per_hour = Gauge(
    "polymorph_polymorphs_per_hour",
    "Rolling polymorph discovery rate per hour",
    ["org_id"]
)

# ============================================================================
# Gating Metrics
# ============================================================================

gating_events_total = Counter(
    "polymorph_gating_events_total",
    "Total gating trigger events",
    ["org_id", "rule_name"]
)

gating_cooldown_active = Gauge(
    "polymorph_gating_cooldown_active",
    "Whether gating cooldown is currently active",
    ["org_id", "rule_name"]
)

# ============================================================================
# Safety Metrics
# ============================================================================

safety_trips_total = Counter(
    "polymorph_safety_trips_total",
    "Total safety trip events",
    ["org_id", "trip_type"]
)

safety_last_trip_timestamp = Gauge(
    "polymorph_safety_last_trip_timestamp",
    "Unix timestamp of last safety trip",
    ["org_id"]
)

# ============================================================================
# Workflow Metrics
# ============================================================================

workflow_duration_seconds = Histogram(
    "polymorph_workflow_duration_seconds",
    "Workflow execution duration in seconds",
    ["org_id", "workflow_name", "status"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)
)

workflow_active = Gauge(
    "polymorph_workflow_active",
    "Number of currently running workflows",
    ["org_id"]
)

workflow_completed_total = Counter(
    "polymorph_workflow_completed_total",
    "Total completed workflows",
    ["org_id", "workflow_name", "status"]
)

# ============================================================================
# System Metrics
# ============================================================================

spectra_processed_total = Counter(
    "polymorph_spectra_processed_total",
    "Total spectra processed",
    ["org_id", "device_id"]
)

inference_latency_seconds = Histogram(
    "polymorph_inference_latency_seconds",
    "AI inference latency in seconds",
    ["org_id"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

system_info = Info(
    "polymorph_system",
    "System information"
)


# ============================================================================
# Metrics API Endpoint
# ============================================================================

@router.get("")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================================
# Helper Functions for Metrics Updates
# ============================================================================

class MetricsManager:
    """Helper class for updating metrics."""
    
    def __init__(self, org_id: str = "default"):
        self.org_id = org_id
    
    def record_polymorph_detected(self, device_id: str = "unknown"):
        """Record a new polymorph detection."""
        polymorphs_detected_total.labels(
            org_id=self.org_id,
            device_id=device_id
        ).inc()
    
    def update_pmm_stats(self, active_modes: int, max_modes: int):
        """Update PMM mode statistics."""
        pmm_active_modes.labels(org_id=self.org_id).set(active_modes)
        pmm_max_modes.labels(org_id=self.org_id).set(max_modes)
    
    def record_gating_event(self, rule_name: str):
        """Record a gating trigger event."""
        gating_events_total.labels(
            org_id=self.org_id,
            rule_name=rule_name
        ).inc()
    
    def record_safety_trip(self, trip_type: str):
        """Record a safety trip."""
        safety_trips_total.labels(
            org_id=self.org_id,
            trip_type=trip_type
        ).inc()
        import time
        safety_last_trip_timestamp.labels(org_id=self.org_id).set(time.time())
    
    def record_workflow_complete(
        self,
        workflow_name: str,
        status: str,
        duration_seconds: float
    ):
        """Record workflow completion."""
        workflow_duration_seconds.labels(
            org_id=self.org_id,
            workflow_name=workflow_name,
            status=status
        ).observe(duration_seconds)
        workflow_completed_total.labels(
            org_id=self.org_id,
            workflow_name=workflow_name,
            status=status
        ).inc()
    
    def record_inference(self, latency_seconds: float):
        """Record AI inference latency."""
        inference_latency_seconds.labels(org_id=self.org_id).observe(latency_seconds)
    
    def record_spectrum_processed(self, device_id: str = "unknown"):
        """Record spectrum processing."""
        spectra_processed_total.labels(
            org_id=self.org_id,
            device_id=device_id
        ).inc()


def set_system_info(version: str, environment: str, git_commit: str = "unknown"):
    """Set system information."""
    system_info.info({
        "version": version,
        "environment": environment,
        "git_commit": git_commit
    })
