import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from retrofitkit.core.orchestrator import Orchestrator
from retrofitkit.core.app import AppContext
from retrofitkit.core.config import PolymorphConfig

@pytest.mark.asyncio
async def test_ai_integration(tmp_path):
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
