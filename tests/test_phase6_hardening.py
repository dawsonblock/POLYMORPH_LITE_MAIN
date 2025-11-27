import pytest
import asyncio
from unittest.mock import MagicMock, patch
from retrofitkit.core.registry import registry
from retrofitkit.core.app import AppContext
from retrofitkit.core.workflows.executor import WorkflowExecutor
from retrofitkit.core.recipe import Step

# --- Driver Registration Tests ---

def test_driver_registration():
    # Force import of drivers
    import retrofitkit.drivers
    
    drivers = registry.list_drivers()
    assert "andor_raman" in drivers
    assert "redpitaya_daq" in drivers
    assert "horiba_raman" in drivers
    assert "ocean_optics" in drivers

# --- Config Unification Tests ---

def test_app_context_config_loading():
    # Mock ConfigLoader
    with patch("retrofitkit.core.config_loader.get_loader") as mock_get_loader:
        mock_loader = MagicMock()
        mock_get_loader.return_value = mock_loader
        mock_loader.load_base.return_value = mock_loader
        
        # Mock resolved config
        mock_config = MagicMock()
        mock_loader.resolve.return_value = mock_config
        
        ctx = AppContext.load()
        assert ctx.config == mock_config

# --- Safety Enforcement Tests ---

@pytest.mark.asyncio
async def test_workflow_safety_enforcement():
    ctx = MagicMock()
    interlocks = MagicMock()
    ctx.safety.interlocks = interlocks # Assuming context has safety access or we inject it
    
    # We need to inject interlocks into executor manually for this test since it usually gets it from context/config
    # But executor __init__ takes context. Let's mock context.
    
    # Mock InterlockController
    with patch("retrofitkit.core.safety.interlocks.InterlockController") as MockInterlocks:
        mock_interlocks_instance = MockInterlocks.return_value
        
        mock_logger = MagicMock()
        executor = WorkflowExecutor(ctx, mock_logger)
        executor.interlocks = mock_interlocks_instance # Inject mock
        
        # Mock handler
        from unittest.mock import AsyncMock
        executor._handle_test_action = AsyncMock()
        executor._handle_test_action.return_value = {}
        
        step = Step(name="test", action="test_action", type="test_action", params={})
        # Step type maps to handler name: _handle_{step.type}
        # We need to set step.type to match our mocked handler
        step.type = "test_action"
        
        await executor._execute_step(step)
        
        # Verify check_safe was called twice (before and after)
        assert mock_interlocks_instance.check_safe.call_count >= 2
