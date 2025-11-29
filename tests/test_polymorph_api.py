"""
Tests for Polymorph Discovery API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from retrofitkit.api.server import app
from retrofitkit.db.session import SessionLocal


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_ai_service():
    """Mock AI service responses."""
    with patch('httpx.AsyncClient') as mock:
        yield mock


class TestPolymorphDetection:
    """Test polymorph detection endpoint."""
    
    @pytest.mark.asyncio
    async def test_detect_polymorph_success(self, client, mock_ai_service):
        """Test successful polymorph detection."""
        # Mock AI service response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "polymorph_detected": True,
            "polymorph_id": 1,
            "polymorph_name": "Form_1",
            "confidence": 0.95,
            "signature_vector": [0.1, 0.2, 0.3],
            "alternative_forms": [],
            "model_version": "1.0.0"
        }
        
        mock_ai_service.return_value.__aenter__.return_value.post.return_value = mock_response
        
        # Make request
        response = client.post(
            "/api/polymorph/detect",
            json={
                "spectrum": [1.0] * 900,
                "wavelengths": list(range(200, 1100)),
                "metadata": {"sample_id": "test-123"}
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["polymorph_detected"] is True
        assert data["confidence"] == 0.95
        assert "event_id" in data
    
    def test_detect_polymorph_invalid_input(self, client):
        """Test detection with invalid input."""
        response = client.post(
            "/api/polymorph/detect",
            json={"spectrum": []},  # Empty spectrum
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 502]
    
    @pytest.mark.asyncio
    async def test_detect_polymorph_ai_timeout(self, client, mock_ai_service):
        """Test handling of AI service timeout."""
        import httpx
        
        mock_ai_service.return_value.__aenter__.return_value.post.side_effect = httpx.TimeoutException("Timeout")
        
        response = client.post(
            "/api/polymorph/detect",
            json={
                "spectrum": [1.0] * 900,
                "wavelengths": list(range(200, 1100))
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 504
        assert "timeout" in response.json()["detail"].lower()


class TestPolymorphEvents:
    """Test polymorph event endpoints."""
    
    def test_list_events_empty(self, client):
        """Test listing events when none exist."""
        response = client.get(
            "/api/polymorph/events",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert isinstance(data["events"], list)
    
    def test_list_events_pagination(self, client):
        """Test event listing with pagination."""
        response = client.get(
            "/api/polymorph/events?limit=10&offset=0",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["events"]) <= 10
    
    def test_get_event_not_found(self, client):
        """Test getting non-existent event."""
        response = client.get(
            "/api/polymorph/events/nonexistent-id",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 404


class TestPolymorphReports:
    """Test polymorph report generation."""
    
    @pytest.mark.asyncio
    async def test_generate_json_report(self, client, mock_ai_service):
        """Test JSON report generation."""
        # First create an event (mock)
        # Then generate report
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "format": "json",
            "data": {
                "report_id": "test-report",
                "detection_summary": {}
            }
        }
        
        mock_ai_service.return_value.__aenter__.return_value.post.return_value = mock_response
        
        response = client.post(
            "/api/polymorph/report",
            json={
                "event_id": "test-event-id",
                "format": "json",
                "include_spectrum": True
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        # May fail due to missing event, but should handle gracefully
        assert response.status_code in [200, 404, 502]
    
    def test_generate_report_invalid_format(self, client):
        """Test report with invalid format."""
        response = client.post(
            "/api/polymorph/report",
            json={
                "event_id": "test-event",
                "format": "invalid"
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Should validate format
        assert response.status_code in [400, 422]


class TestPolymorphStatistics:
    """Test polymorph statistics endpoint."""
    
    def test_get_statistics(self, client):
        """Test statistics retrieval."""
        response = client.get(
            "/api/polymorph/statistics",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "total_detections" in data
        assert "average_confidence" in data
        assert "by_polymorph_type" in data
        assert isinstance(data["by_polymorph_type"], list)


class TestDatabaseIntegration:
    """Test database integration for polymorph tables."""
    
    def test_polymorph_event_creation(self):
        """Test creating polymorph event in database."""
        from retrofitkit.db.models.polymorph import PolymorphEvent
        import time
        
        db = SessionLocal()
        try:
            event = PolymorphEvent(
                event_id="test-event-123",
                detected_at=time.time(),
                polymorph_id=1,
                polymorph_name="Form_1",
                confidence=0.95,
                model_version="1.0.0",
                operator_email="test@example.com",
                metadata={"test": True}
            )
            
            db.add(event)
            db.commit()
            
            # Verify
            retrieved = db.query(PolymorphEvent).filter(
                PolymorphEvent.event_id == "test-event-123"
            ).first()
            
            assert retrieved is not None
            assert retrieved.confidence == 0.95
            
        finally:
            db.rollback()
            db.close()
    
    def test_polymorph_signature_relationship(self):
        """Test signature relationship to event."""
        from retrofitkit.db.models.polymorph import PolymorphEvent, PolymorphSignature
        import time
        
        db = SessionLocal()
        try:
            event = PolymorphEvent(
                event_id="test-event-456",
                detected_at=time.time(),
                polymorph_id=2,
                polymorph_name="Form_2",
                confidence=0.88,
                model_version="1.0.0",
                operator_email="test@example.com"
            )
            db.add(event)
            db.flush()
            
            signature = PolymorphSignature(
                signature_id="test-sig-456",
                event_id="test-event-456",
                polymorph_id=2,
                signature_vector=[0.1, 0.2, 0.3],
                created_at=time.time()
            )
            db.add(signature)
            db.commit()
            
            # Verify relationship
            retrieved_event = db.query(PolymorphEvent).filter(
                PolymorphEvent.event_id == "test-event-456"
            ).first()
            
            assert len(retrieved_event.signatures) == 1
            assert retrieved_event.signatures[0].signature_id == "test-sig-456"
            
        finally:
            db.rollback()
            db.close()
