import pytest
import os
import numpy as np
from unittest.mock import MagicMock, patch
from retrofitkit.drivers.daq.ni import NIDAQ
from retrofitkit.drivers.raman.vendor_ocean_optics import OceanOpticsSpectrometer

@pytest.mark.asyncio
async def test_hardware_pipeline_simulation():
    """Test that pipeline defaults to simulation when flag is unset."""
    # Ensure flag is unset
    if "USE_REAL_HARDWARE" in os.environ:
        del os.environ["USE_REAL_HARDWARE"]

    daq = NIDAQ()
    raman = OceanOpticsSpectrometer()

    await daq.connect()
    await raman.connect()

    daq_health = await daq.health()
    raman_health = await raman.health()

    assert daq_health["mode"] == "simulation"
    assert raman_health["mode"] == "simulation"

    # Test acquisition
    await daq.set_voltage(1.23)
    assert await daq.read_ai() == 1.23

    spectrum = await raman.acquire_spectrum()
    assert spectrum.meta["mode"] == "simulation"
    assert len(spectrum.wavelengths) == 1024

    await daq.disconnect()
    await raman.disconnect()

@pytest.mark.asyncio
async def test_hardware_pipeline_real_mocked():
    """Test real hardware path with mocked SDKs."""
    with patch.dict(os.environ, {"USE_REAL_HARDWARE": "1"}):
        with patch("retrofitkit.drivers.daq.ni.nidaqmx") as mock_nidaq, \
             patch("retrofitkit.drivers.raman.vendor_ocean_optics.sb") as mock_sb:
            
            # Setup mocks
            mock_sb.list_devices.return_value = ["mock_spec_0"]
            mock_spec = MagicMock()
            mock_spec.wavelengths.return_value = np.array([500.0, 501.0])
            mock_spec.intensities.return_value = np.array([100.0, 200.0])
            mock_spec.model = "MockSpec"
            mock_sb.Spectrometer.return_value = mock_spec

            daq = NIDAQ()
            raman = OceanOpticsSpectrometer()

            await daq.connect()
            await raman.connect()

            # Verify modes
            assert (await daq.health())["mode"] == "hardware"
            assert (await raman.health())["mode"] == "hardware"

            # Verify DAQ calls
            await daq.set_voltage(5.0)
            # Check if Task() was called
            assert mock_nidaq.Task.called

            # Verify Raman calls
            spectrum = await raman.acquire_spectrum()
            assert spectrum.meta["mode"] == "hardware"
            assert spectrum.meta["model"] == "MockSpec"
            assert len(spectrum.wavelengths) == 2

            await daq.disconnect()
            await raman.disconnect()
