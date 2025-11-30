"""
Tests for Hardware Drivers.
"""
import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch

from retrofitkit.drivers.ocean_optics_driver import OceanOpticsDriver
from retrofitkit.drivers.red_pitaya_driver import RedPitayaDriver
from retrofitkit.pipelines.daq_to_raman import UnifiedDAQPipeline

class TestOceanOpticsDriver:
    def test_initialization_simulated(self):
        driver = OceanOpticsDriver(simulate=True)
        assert driver.is_simulated()
        assert driver._wavelengths is not None
        assert len(driver._wavelengths) == 2048
        driver.disconnect()

    def test_acquire_spectrum(self):
        driver = OceanOpticsDriver(simulate=True)
        wl, inten = driver.acquire_spectrum(average=1)
        assert len(wl) == 2048
        assert len(inten) == 2048
        assert np.all(inten >= 0)
        driver.disconnect()

    def test_integration_time(self):
        driver = OceanOpticsDriver(simulate=True)
        # Should not raise error
        driver.set_integration_time(10000)
        driver.disconnect()

class TestRedPitayaDriver:
    def test_initialization_simulated(self):
        driver = RedPitayaDriver(simulate=True)
        assert driver.simulate
        driver.disconnect()

    def test_waveform_acquisition(self):
        driver = RedPitayaDriver(simulate=True)
        data = driver.get_waveform(channel=1, num_samples=1024)
        assert len(data) == 1024
        # Simulation returns sine wave
        assert np.std(data) > 0
        driver.disconnect()

    def test_self_test(self):
        driver = RedPitayaDriver(simulate=True)
        assert driver.self_test() is True
        driver.disconnect()

class TestUnifiedPipeline:
    def test_pipeline_execution(self, tmp_path):
        # Use tmp_path for output
        pipeline = UnifiedDAQPipeline(output_dir=str(tmp_path))
        
        # Run pipeline
        result = pipeline.run_acquisition(sample_name="test_sample")
        
        assert result["meta"]["sample_name"] == "test_sample"
        assert "raman" in result
        assert "daq" in result
        assert os.path.exists(result["files"]["json"])
        assert os.path.exists(result["files"]["csv_raman"])
        
        pipeline.close()
