import pytest
import numpy as np
from retrofitkit.drivers.daq.simdaq_v2 import SimDAQv2

@pytest.mark.asyncio
async def test_simdaq_acquisition():
    config = type('Config', (), {'seed': 42})()
    daq = SimDAQv2(config)
    
    # Test acquisition
    results = await daq.acquire_waveform(channels=[0, 1], sample_rate=1000, duration=1.0)
    
    assert "ch0" in results
    assert "ch1" in results
    assert len(results["ch0"]) == 1000
    
    # Verify deterministic output (Sine wave on Ch0)
    # Mean should be close to 0
    assert np.abs(np.mean(results["ch0"])) < 0.1
    # Std dev should be significant (signal + noise)
    assert np.std(results["ch0"]) > 0.5

@pytest.mark.asyncio
async def test_simdaq_generation():
    config = type('Config', (), {'seed': 42})()
    daq = SimDAQv2(config)
    
    waveform = np.zeros(100)
    result = await daq.generate_waveform(channel=0, waveform=waveform, sample_rate=1000)
    
    assert result["status"] == "generated"
    assert result["samples"] == 100

@pytest.mark.asyncio
async def test_simdaq_error_mode():
    config = type('Config', (), {'seed': 42})()
    daq = SimDAQv2(config)
    
    daq.set_error_mode(True)
    
    with pytest.raises(RuntimeError, match="Simulated hardware failure"):
        await daq.acquire_waveform(channels=[0], sample_rate=1000, duration=1.0)
