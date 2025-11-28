import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from retrofitkit.core.orchestrator import Orchestrator
from retrofitkit.core.app import AppContext
from retrofitkit.core.recipe import Recipe, Step

@pytest.mark.asyncio
async def test_orchestrator_integration():
    # 1. Mock Context & Config
    mock_ctx = MagicMock()
    mock_ctx.config.system.data_dir = "/tmp"
    mock_ctx.config.ai.service_url = "http://localhost:3000"
    mock_ctx.config.gating.rules = [{"name": "peak_threshold", "threshold": 100, "direction": "above"}]
    mock_ctx.config.safety.estop_line = 1
    mock_ctx.config.safety.door_line = 2
    
    # 2. Mock Registry & Drivers
    with patch("retrofitkit.core.orchestrator.registry") as mock_registry:
        mock_daq = AsyncMock()
        mock_raman = AsyncMock()
        mock_registry.create.side_effect = lambda name, cfg: mock_daq if "daq" in name else mock_raman
        
        # 3. Initialize Orchestrator
        orch = Orchestrator(mock_ctx)
        
        # Verify Interlocks initialized
        assert orch.interlocks is not None
        # Verify Gating initialized
        assert orch.gating_engine is not None
        
        # 4. Execute Recipe
        recipe = Recipe(name="Test", steps=[Step(type="raman", params={"exposure_time": 1})])
        
        # Mock Executor to avoid real DB/Execution logic, OR verify executor creation
        # Let's mock execute_recipe to avoid full execution complexity in this unit test,
        # but we want to verify executor gets the components.
        
        # Actually, let's inspect the executor creation inside execute_recipe if possible,
        # or just trust the code we wrote and test the components are passed.
        # Since execute_recipe instantiates WorkflowExecutor locally, we can't easily inspect it without mocking the class.
        
        with patch("retrofitkit.core.workflows.executor.WorkflowExecutor") as MockExecutor:
            mock_executor_instance = MockExecutor.return_value
            mock_executor_instance.execute = AsyncMock()
            
            await orch.run(recipe, "test@example.com")
            
            # Verify Executor initialized with correct args
            # args: config, db_logger, ai_client, gating_engine
            args, _ = MockExecutor.call_args
            assert args[2] == orch.ai_client
            assert args[3] == orch.gating_engine
            
            # Verify execute called
            mock_executor_instance.execute.assert_called_once()

@pytest.mark.asyncio
async def test_interlock_polling():
    # Test that watchdog polls interlocks
    mock_ctx = MagicMock()
    # Minimal config
    mock_ctx.config.system.data_dir = "/tmp"
    
    with patch("retrofitkit.core.orchestrator.registry"):
        orch = Orchestrator(mock_ctx)
        orch.interlocks = AsyncMock()
        orch.daq = AsyncMock()
        
        # Run watchdog for a brief moment
        task = asyncio.create_task(orch._watchdog_loop())
        await asyncio.sleep(1.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
            
        # Verify check_status called
        assert orch.interlocks.check_status.called
