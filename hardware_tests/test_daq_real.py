import pytest
import os
from unittest.mock import MagicMock, patch
from retrofitkit.drivers.daq.ni import NIDAQ

@pytest.mark.asyncio
async def test_daq_real_sdk_missing():
    """Test failure when USE_REAL_HARDWARE=1 but SDK missing."""
    with patch.dict(os.environ, {"USE_REAL_HARDWARE": "1"}):
        with patch("retrofitkit.drivers.daq.ni.nidaqmx", None):
            with pytest.raises(RuntimeError, match="nidaqmx SDK is not installed"):
                NIDAQ()

@pytest.mark.asyncio
async def test_daq_real_connection_failure():
    """Test failure when real hardware connection fails."""
    with patch.dict(os.environ, {"USE_REAL_HARDWARE": "1"}):
        with patch("retrofitkit.drivers.daq.ni.nidaqmx") as mock_nidaq:
            # Simulate Task() raising an error
            mock_nidaq.Task.side_effect = Exception("Device not found")

            daq = NIDAQ()
            
            with pytest.raises(RuntimeError, match="Failed to connect to real NI hardware"):
                await daq.connect()
