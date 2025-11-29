"""
Tests for AI Service Client Circuit Breaker functionality.

Tests cover:
- Circuit breaker opening after N failures
- Circuit breaker recovery after timeout
- Critical vs non-critical call handling
- Failsafe error raising
"""
import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock
from retrofitkit.core.ai_client import AIServiceClient, AIFailsafeError


@pytest.fixture
def ai_client():
    """Create AI client with test configuration."""
    client = AIServiceClient(service_url="http://test-ai-service:8000")
    client._failure_threshold = 3
    client._recovery_timeout = 2.0  # Short timeout for testing
    return client


@pytest.mark.asyncio
async def test_successful_prediction(ai_client):
    """Test successful AI prediction returns expected data."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "polymorph_id": "Form_A",
        "confidence": 0.95,
        "class": "crystalline"
    }
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        result = await ai_client.predict([1.0, 2.0, 3.0], critical=True)
        
        assert result["polymorph_id"] == "Form_A"
        assert result["confidence"] == 0.95
        assert ai_client._failures == 0
        assert not ai_client._circuit_open


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures(ai_client):
    """Test circuit breaker opens after threshold failures."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        # Make 3 non-critical calls to trigger circuit breaker
        for i in range(3):
            result = await ai_client.predict([1.0, 2.0], critical=False)
            assert result == {}
        
        # Circuit should now be open
        assert ai_client._circuit_open
        assert ai_client._failures >= 3


@pytest.mark.asyncio
async def test_critical_call_raises_when_circuit_open(ai_client):
    """Test critical calls raise AIFailsafeError when circuit is open."""
    # Force circuit open
    ai_client._circuit_open = True
    ai_client._last_failure_time = time.time()
    
    with pytest.raises(AIFailsafeError, match="Circuit Breaker OPEN"):
        await ai_client.predict([1.0, 2.0], critical=True)


@pytest.mark.asyncio
async def test_non_critical_call_returns_empty_when_circuit_open(ai_client):
    """Test non-critical calls return empty dict when circuit is open."""
    # Force circuit open
    ai_client._circuit_open = True
    ai_client._last_failure_time = time.time()
    
    result = await ai_client.predict([1.0, 2.0], critical=False)
    assert result == {}


@pytest.mark.asyncio
async def test_circuit_breaker_recovery(ai_client):
    """Test circuit breaker recovers after timeout."""
    # Force circuit open with old failure time
    ai_client._circuit_open = True
    ai_client._failures = 5
    ai_client._last_failure_time = time.time() - 10.0  # 10 seconds ago
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"polymorph_id": "Form_B"}
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        result = await ai_client.predict([1.0, 2.0], critical=False)
        
        # Circuit should recover
        assert not ai_client._circuit_open
        assert ai_client._failures == 0
        assert result["polymorph_id"] == "Form_B"


@pytest.mark.asyncio
async def test_timeout_exception_handling(ai_client):
    """Test timeout exceptions are handled gracefully."""
    import httpx
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.TimeoutException("Connection timeout")
        )
        
        # Non-critical should return empty dict
        result = await ai_client.predict([1.0, 2.0], critical=False)
        assert result == {}
        assert ai_client._failures == 1
        
        # Critical should raise
        with pytest.raises(AIFailsafeError, match="Timeout"):
            await ai_client.predict([1.0, 2.0], critical=True)


@pytest.mark.asyncio
async def test_connection_error_handling(ai_client):
    """Test connection errors are handled gracefully."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        
        # Non-critical should return empty dict
        result = await ai_client.predict([1.0, 2.0], critical=False)
        assert result == {}
        
        # Critical should raise
        with pytest.raises(AIFailsafeError, match="Connection Failed"):
            await ai_client.predict([1.0, 2.0], critical=True)


def test_circuit_breaker_status(ai_client):
    """Test circuit breaker status reporting."""
    status = ai_client.status
    
    assert "circuit_open" in status
    assert "failures" in status
    assert "service_url" in status
    assert status["service_url"] == "http://test-ai-service:8000"
    assert status["circuit_open"] is False
    assert status["failures"] == 0


@pytest.mark.asyncio
async def test_multiple_failures_increment_counter(ai_client):
    """Test that multiple failures properly increment the failure counter."""
    mock_response = MagicMock()
    mock_response.status_code = 503  # Service unavailable
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        # Make multiple non-critical calls
        for i in range(5):
            await ai_client.predict([1.0], critical=False)
        
        assert ai_client._failures >= 3  # Circuit opens at threshold
        assert ai_client._circuit_open is True
