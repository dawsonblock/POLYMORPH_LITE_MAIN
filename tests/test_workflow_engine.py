import pytest
import asyncio
import uuid
from unittest.mock import MagicMock, AsyncMock, patch
from retrofitkit.core.workflows.executor import WorkflowExecutor
from retrofitkit.core.workflows.db_logger import DatabaseLogger
from retrofitkit.core.recipe import Recipe, Step

# --- Mock Data ---

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.daq.backend = "mock_daq"
    cfg.raman.provider = "mock_raman"
    return cfg

@pytest.fixture
def mock_db_logger():
    logger = MagicMock(spec=DatabaseLogger)
    logger.log_run_start.return_value = "RUN-123"
    return logger

@pytest.fixture
def mock_router():
    with patch("retrofitkit.core.workflows.executor.get_router") as mock:
        router = MagicMock()
        mock.return_value = router
        
        # Mock Drivers
        daq = AsyncMock()
        daq.read_ai.return_value = 1.23
        
        raman = AsyncMock()
        raman.acquire_spectrum.return_value = {"wavelengths": [1, 2], "intensities": [10, 20]}
        
        router.get_driver.side_effect = lambda type, cfg: daq if type == "daq" else raman
        yield router

# --- Tests ---

@pytest.mark.asyncio
async def test_executor_run_success(mock_config, mock_db_logger, mock_router):
    # Create Recipe
    recipe = Recipe(
        name="Test Recipe",
        steps=[
            Step(type="wait", params={"seconds": 0.01}),
            Step(type="daq", params={"action": "read_ai", "channel": 1}),
            Step(type="raman", params={"exposure_time": 0.1})
        ]
    )
    recipe.id = uuid.uuid4() # Mock ID
    
    executor = WorkflowExecutor(mock_config, mock_db_logger)
    
    # Mock interlocks to avoid safety errors
    executor.interlocks = MagicMock()
    executor.interlocks.check_safe.return_value = None
    
    await executor.execute(recipe, "test@example.com")
    
    # Verify Logger Calls
    mock_db_logger.log_run_start.assert_called_once()
    assert mock_db_logger.log_step_start.call_count == 3
    assert mock_db_logger.log_step_complete.call_count == 3
    mock_db_logger.log_run_complete.assert_called_with("completed")
    
    # Verify Driver Calls
    daq = mock_router.get_driver("daq", mock_config)
    daq.read_ai.assert_called_with(1)
    
    raman = mock_router.get_driver("raman", mock_config)
    raman.acquire_spectrum.assert_called_with(exposure_time=0.1)

@pytest.mark.asyncio
async def test_executor_stop(mock_config, mock_db_logger, mock_router):
    recipe = Recipe(
        name="Long Recipe",
        steps=[
            Step(type="wait", params={"seconds": 0.1}),
            Step(type="wait", params={"seconds": 0.1})
        ]
    )
    recipe.id = uuid.uuid4()
    
    executor = WorkflowExecutor(mock_config, mock_db_logger)
    
    # Start execution in background
    task = asyncio.create_task(executor.execute(recipe, "test@example.com"))
    
    # Stop immediately
    executor.stop()
    
    await task
    
    # Should be aborted
    mock_db_logger.log_run_complete.assert_called_with("aborted", "Stopped by user")
