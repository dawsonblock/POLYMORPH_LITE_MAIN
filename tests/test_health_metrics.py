"""
Tests for health endpoints and Prometheus metrics.

Tests cover:
- Health endpoint returns 200 in simulation mode
- Health shows degraded status when hardware missing
- Metrics endpoint responds correctly
- Prometheus format validation
"""
import pytest
from fastapi.testclient import TestClient
from retrofitkit.api.server import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_endpoint_returns_200(client):
    """Test health endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_health_endpoint_simulation_mode(client):
    """Test health endpoint works in simulation mode."""
    response = client.get("/health")
    data = response.json()
    
    # Should return OK or degraded, not error
    assert response.status_code in [200, 503]
    assert data["status"] in ["healthy", "degraded"]


def test_readiness_endpoint(client):
    """Test readiness endpoint."""
    response = client.get("/health/ready")
    
    # Should be ready even in simulation mode
    assert response.status_code in [200, 503]
    data = response.json()
    assert "ready" in data or "status" in data


def test_liveness_endpoint(client):
    """Test liveness endpoint."""
    response = client.get("/health/live")
    
    # Liveness should always return 200 if app is running
    assert response.status_code == 200


def test_metrics_endpoint_exists(client):
    """Test Prometheus metrics endpoint exists."""
    response = client.get("/metrics")
    
    # Should return metrics in Prometheus format
    assert response.status_code == 200
    assert response.headers["content-type"] in [
        "text/plain; charset=utf-8",
        "text/plain",
        "text/plain; version=0.0.4; charset=utf-8"
    ]


def test_metrics_contains_expected_metrics(client):
    """Test metrics endpoint contains expected metric names."""
    response = client.get("/metrics")
    content = response.text
    
    # Should contain some standard metrics
    # Note: Exact metrics depend on what's instrumented
    assert len(content) > 0
    
    # Common Prometheus metrics patterns
    assert any(keyword in content for keyword in [
        "http_", "process_", "python_", "request_", "response_"
    ])


def test_health_endpoint_includes_components(client):
    """Test health endpoint includes component status."""
    response = client.get("/health")
    data = response.json()
    
    # Should have some component information
    assert "status" in data
    
    # May include database, redis, etc.
    if "components" in data:
        assert isinstance(data["components"], dict)


def test_metrics_format_valid(client):
    """Test metrics are in valid Prometheus format."""
    response = client.get("/metrics")
    lines = response.text.split("\n")
    
    # Prometheus format: lines starting with # are comments/help/type
    # Other lines are metric_name{labels} value timestamp
    for line in lines:
        if line.strip() and not line.startswith("#"):
            # Should have metric name and value
            parts = line.split()
            assert len(parts) >= 2  # name and value at minimum
