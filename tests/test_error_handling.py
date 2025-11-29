import pytest
import asyncio
from unittest.mock import MagicMock
from sqlalchemy.exc import IntegrityError
from retrofitkit.db.session import safe_db_commit
from retrofitkit.core.hardware_utils import hardware_call
from retrofitkit.core.error_codes import ErrorCode

# --- DB Tests ---
def test_safe_db_commit_success():
    mock_db = MagicMock()
    with safe_db_commit(mock_db):
        pass
    mock_db.commit.assert_called_once()
    mock_db.rollback.assert_not_called()

def test_safe_db_commit_failure():
    mock_db = MagicMock()
    with pytest.raises(ValueError):
        with safe_db_commit(mock_db):
            raise ValueError("DB Error")
    mock_db.commit.assert_not_called()
    mock_db.rollback.assert_called_once()

# --- Hardware Tests ---
@pytest.mark.asyncio
async def test_hardware_call_success():
    @hardware_call()
    async def success_func():
        return "ok"
    
    result = await success_func()
    assert result == "ok"

@pytest.mark.asyncio
async def test_hardware_call_timeout():
    @hardware_call(timeout=0.1, retries=0)
    async def slow_func():
        await asyncio.sleep(0.2)
        return "ok"
    
    with pytest.raises(TimeoutError):
        await slow_func()

@pytest.mark.asyncio
async def test_hardware_call_retry_success():
    attempts = 0
    @hardware_call(retries=2, retry_delay=0.01)
    async def flaky_func():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ValueError("Fail")
        return "ok"
    
    result = await flaky_func()
    assert result == "ok"
    assert attempts == 2

@pytest.mark.asyncio
async def test_hardware_call_retry_failure():
    attempts = 0
    @hardware_call(retries=1, retry_delay=0.01, error_code=ErrorCode.HARDWARE_CONFIG_ERROR)
    async def fail_func():
        nonlocal attempts
        attempts += 1
        raise ValueError("Fail")
    
    with pytest.raises(ValueError):
        await fail_func()
    assert attempts == 2
