import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from retrofitkit.core.workflows.executor import WorkflowExecutor
from retrofitkit.core.recipe import Recipe, Step
from retrofitkit.core.ai_client import AIServiceClient

@pytest.mark.asyncio
async def test_ai_decision_step():
    # 1. Mock Config & Logger
    mock_config = MagicMock()
    mock_logger = MagicMock()
    
    # 2. Mock AI Client
    mock_ai_client = MagicMock(spec=AIServiceClient)
    mock_ai_client.predict = AsyncMock(return_value={"polymorph": "alpha", "confidence": 0.95})
    
    # 3. Initialize Executor
    executor = WorkflowExecutor(mock_config, mock_logger, mock_ai_client)
    
    # 4. Mock Router & Driver
    mock_router = MagicMock()
    mock_raman = AsyncMock()
    # Return a dict simulating a spectrum
    mock_raman.acquire_spectrum.return_value = {
        "wavelengths": [1, 2, 3],
        "intensities": [10, 20, 30],
        "meta": {}
    }
    executor.router = mock_router
    mock_router.get_driver.return_value = mock_raman
    
    # 5. Define Recipe
    recipe = Recipe(
        name="Test AI Recipe",
        steps=[
            Step(type="raman", params={"exposure_time": 100}),
            Step(type="ai_decision", params={"critical": True})
        ]
    )
    
    # 6. Execute
    await executor.execute(recipe, "test@example.com")
    
    # 7. Verify
    # Check if AI client was called
    mock_ai_client.predict.assert_called_once()
    
    # Check arguments: should be the intensities from the raman step
    args, kwargs = mock_ai_client.predict.call_args
    assert args[0] == [10, 20, 30]
    assert kwargs["critical"] is True
    
    # Check logs
    mock_logger.log_step_complete.assert_any_call(1, "ai_decision", {"prediction": {"polymorph": "alpha", "confidence": 0.95}})
