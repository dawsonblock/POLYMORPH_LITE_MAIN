import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from retrofitkit.core.calibration.spectrometer import SpectrometerCalibrator, CalibrationError
from scripts.unified_cli import cmd_test_hardware, cmd_calibrate

# --- Calibration Tests ---

def test_pixel_to_nm():
    config = MagicMock()
    config.raman.calibration_coeffs = [500, 0.5, 0] # 500 + 0.5x
    
    cal = SpectrometerCalibrator(config)
    nm = cal.pixel_to_nm([0, 10, 100])
    assert nm == [500.0, 505.0, 550.0]

def test_calibrate_wavelength():
    config = MagicMock()
    config.raman.calibration_coeffs = [0, 1, 0]
    cal = SpectrometerCalibrator(config)
    
    # Linear fit: y = 2x + 10
    pixels = [0, 10, 20]
    nm = [10, 30, 50]
    
    coeffs = cal.calibrate_wavelength(pixels, nm, order=1)
    # numpy polyfit returns [slope, intercept] -> [2, 10]
    # We store as [intercept, slope] -> [10, 2]
    assert pytest.approx(coeffs[0], 0.01) == 10.0
    assert pytest.approx(coeffs[1], 0.01) == 2.0

def test_subtract_baseline():
    config = MagicMock()
    cal = SpectrometerCalibrator(config)
    
    spec = [100, 200, 300]
    dark = [10, 10, 10]
    res = cal.subtract_baseline(spec, dark)
    assert res == [90, 190, 290]

# --- CLI Tests ---

@pytest.mark.asyncio
async def test_cli_test_hardware():
    args = MagicMock()
    config = MagicMock()
    
    with patch("scripts.unified_cli.get_router") as mock_get_router:
        router = MagicMock()
        mock_get_router.return_value = router
        
        driver = AsyncMock()
        driver.health.return_value = {"status": "ok"}
        router.get_driver.return_value = driver
        
        await cmd_test_hardware(args, config)
        
        assert driver.connect.call_count == 2 # DAQ and Raman
        assert driver.disconnect.call_count == 2

@pytest.mark.asyncio
async def test_cli_calibrate():
    args = MagicMock()
    args.device = "spectrometer"
    config = MagicMock()
    config.raman.calibration_coeffs = [1, 2, 3]
    
    # Should just log info, not crash
    await cmd_calibrate(args, config)
