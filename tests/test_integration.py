import pytest
import asyncio
import os
from unittest.mock import MagicMock, AsyncMock
from retrofitkit.core.orchestrator import Orchestrator
from retrofitkit.core.app import AppContext
from retrofitkit.core.config import PolymorphConfig

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("P4_RUN_AI_INTEGRATION", "0") == "1",
    reason="AI integration test requires running Bentoml service on localhost:3000. Set P4_RUN_AI_INTEGRATION=1 to enable."
)
async def test_ai_integration(tmp_path):
    """
    Integration test for AI service connectivity.
    
    Requires:
    - Bentoml service running on localhost:3000
    - Set environment variable: P4_RUN_AI_INTEGRATION=1
    
    To run: P4_RUN_AI_INTEGRATION=1 pytest tests/test_integration.py -v
    """
    # Setup context
    config = PolymorphConfig()
    # Ensure we point to the running local service
    config.ai.service_url = "http://localhost:3000/infer"
    config.system.data_dir = str(tmp_path)
    
    ctx = MagicMock(spec=AppContext)
    ctx.config = config
    
    # Initialize Orchestrator
    # We need to mock make_daq and make_raman to avoid hardware init
    with pytest.MonkeyPatch.context() as m:
        m.setattr("retrofitkit.core.orchestrator.make_daq", lambda c: AsyncMock())
        m.setattr("retrofitkit.core.orchestrator.make_raman", lambda c: AsyncMock())
        m.setattr("retrofitkit.core.orchestrator.redis.Redis", lambda **k: AsyncMock())
        
        orch = Orchestrator(ctx)
        
        # Test the _call_inference_service method directly
        # Create a fake spectrum
        spectrum = [100.0] * 1024
        
        print("Calling AI service...")
        result = await orch._call_inference_service(spectrum)
        
        print(f"AI Result: {result}")
        
    # Verify response structure
    assert "status" in result
    assert "active_modes" in result
    assert "polymorphs_found" in result
    assert "predicted_finish_sec" in result
    assert "timestamp" in result
    
    # Status can be stable, crystallizing, or degrading (slope-based)
    assert result["status"] in ["stable", "crystallizing", "degrading"]
