import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from retrofitkit.core.safety.interlocks import InterlockController, SafetyError
from retrofitkit.core.safety.watchdog import SystemWatchdog
from retrofitkit.drivers.raman.vendor_andor import AndorRaman
from retrofitkit.drivers.daq.redpitaya import RedPitayaDAQ

# --- Safety Tests ---

@pytest.mark.asyncio
async def test_interlock_controller_safe():
    config = MagicMock()
    config.safety.estop_line = 0
    config.safety.door_line = 1
    
    daq = AsyncMock()
    # Safe state: E-Stop=1 (High/Safe), Door=1 (Closed/Safe)
    daq.read_di.side_effect = lambda line: True 
    
    ctrl = InterlockController(config)
    ctrl.set_daq(daq)
    
    status = await ctrl.check_status()
    assert status["estop_active"] == False
    assert status["door_open"] == False
    
    # Should not raise
    ctrl.check_safe()

@pytest.mark.asyncio
async def test_interlock_controller_unsafe():
    config = MagicMock()
    config.safety.estop_line = 0
    config.safety.door_line = 1
    
    daq = AsyncMock()
    # Unsafe state: E-Stop=0 (Low/Active)
    daq.read_di.side_effect = lambda line: False
    
    ctrl = InterlockController(config)
    ctrl.set_daq(daq)
    
    status = await ctrl.check_status()
    assert status["estop_active"] == True
    
    with pytest.raises(SafetyError):
        ctrl.check_safe()

# --- Driver Tests ---

@pytest.mark.asyncio
async def test_andor_driver_simulation():
    config = MagicMock()
    driver = AndorRaman(config)
    
    # Mock interlocks to be safe
    driver.interlocks = MagicMock()
    driver.interlocks.check_safe.return_value = None
    
    await driver.connect()
    assert driver.connected
    
    # Test acquisition
    data = await driver.acquire_spectrum(integration_time_ms=10.0)
    assert data.wavelengths is not None
    assert data.intensities is not None
    assert len(data.wavelengths) == 1024
    
    # Patch the class method directly
    with patch("retrofitkit.drivers.raman.vendor_andor.AndorRaman.get_temperature", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = -45.0
        temp = await driver.get_temperature()
        assert temp < 20.0 # Should have started cooling in simulation

@pytest.mark.asyncio
async def test_redpitaya_driver_safety_check():
    config = MagicMock()
    config.daq.redpitaya_host = "localhost"
    config.daq.redpitaya_port = 5000
    
    driver = RedPitayaDAQ(config)
    
    # Mock connection to avoid real network call
    driver.connected = True
    driver._send_cmd = AsyncMock()
    driver._query = AsyncMock(return_value="1.23")
    
    # Mock interlocks to raise error
    driver.interlocks = MagicMock()
    driver.interlocks.check_safe.side_effect = SafetyError("Unsafe")
    
    # Should fail due to safety check
    with pytest.raises(SafetyError):
        await driver.write_ao(0, 1.0)
