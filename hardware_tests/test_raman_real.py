import pytest
import os
from unittest.mock import MagicMock, patch
from retrofitkit.drivers.raman.vendor_ocean_optics import OceanOpticsSpectrometer

@pytest.mark.asyncio
async def test_raman_real_connection_failure():
    """Test failure when USE_REAL_HARDWARE=1 but no device found."""
    with patch.dict(os.environ, {"USE_REAL_HARDWARE": "1"}):
        with patch("retrofitkit.drivers.raman.vendor_ocean_optics.sb") as mock_sb:
            mock_sb.list_devices.return_value = [] # No devices

            raman = OceanOpticsSpectrometer()
            
            with pytest.raises(RuntimeError, match="no Ocean Optics devices found"):
                await raman.connect()

@pytest.mark.asyncio
async def test_raman_real_sdk_missing():
    """Test failure when USE_REAL_HARDWARE=1 but SDK missing."""
    with patch.dict(os.environ, {"USE_REAL_HARDWARE": "1"}):
        # Patch the module where 'sb' is imported to be None
        with patch("retrofitkit.drivers.raman.vendor_ocean_optics.sb", None):
            with pytest.raises(RuntimeError, match="seabreeze SDK is not installed"):
                OceanOpticsSpectrometer()
