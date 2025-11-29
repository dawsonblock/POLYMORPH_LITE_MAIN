import pytest
import math
from unittest.mock import MagicMock, AsyncMock, patch
from retrofitkit.core.ai_client import AIServiceClient

@pytest.mark.asyncio
async def test_ai_input_validation():
    """Verify AI client rejects invalid inputs."""
    client = AIServiceClient("http://mock-ai")
    
    # Empty list
    with pytest.raises(ValueError, match="non-empty list"):
        await client.predict([])
        
    # Non-numeric
    with pytest.raises(ValueError, match="non-numeric"):
        await client.predict(["a", "b"])
        
    # NaNs
    with pytest.raises(ValueError, match="NaN/Inf"):
        await client.predict([1.0, float('nan'), 2.0])

@pytest.mark.asyncio
async def test_ai_output_validation():
    """Verify AI client validates response."""
    client = AIServiceClient("http://mock-ai")
    
    # Mock httpx response
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    from retrofitkit.core.ai_client import AIFailsafeError

    # Case 1: Invalid JSON structure (not a dict)
    mock_response.json.return_value = ["not", "a", "dict"]
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        with pytest.raises(AIFailsafeError, match="Invalid response format"):
            await client.predict([1.0, 2.0])

    # Case 2: Invalid concentration (NaN)
    mock_response.json.return_value = {"concentration": float('nan')}
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        with pytest.raises(AIFailsafeError, match="Invalid concentration value"):
            await client.predict([1.0, 2.0])

    # Case 3: Valid response
    mock_response.json.return_value = {"concentration": 0.95}
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        result = await client.predict([1.0, 2.0])
        assert result["concentration"] == 0.95
