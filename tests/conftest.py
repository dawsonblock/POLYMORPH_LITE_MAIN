"""
Pytest configuration and fixtures for POLYMORPH-4 Lite
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import yaml

from retrofitkit.core.app import AppContext
from retrofitkit.drivers.daq.simulator import SimDAQ
from retrofitkit.drivers.raman.simulator import SimRaman

# ... (rest of file)


@pytest.fixture
def temp_config_dir():
    """Create temporary configuration directory for testing."""
    temp_dir = tempfile.mkdtemp()
    config_dir = Path(temp_dir) / "config"
    config_dir.mkdir(parents=True)
    
    # Create test configuration
    test_config = {
        "system": {
            "name": "Test System",
            "mode": "simulation",
            "timezone": "UTC",
            "data_dir": str(temp_dir / "data"),
            "logs_dir": str(temp_dir / "logs")
        },
        "daq": {
            "backend": "simulator",
            "simulator": {
                "noise_std": 0.001,
                "estop": False,
                "door_open": False
            }
        },
        "raman": {
            "provider": "simulator",
            "simulator": {
                "peak_nm": 532.0,
                "base_intensity": 1200.0,
                "noise_std": 3.0,
                "drift_per_s": 0.8
            }
        },
        "security": {
            "password_policy": {
                "min_length": 8,
                "require_upper": False,
                "require_digit": False,
                "require_symbol": False
            },
            "two_person_signoff": False,
            "jwt_exp_minutes": 60
        },
        "safety": {
            "interlocks": {
                "estop_line": 0,
                "door_line": 1
            },
            "watchdog_seconds": 2.0
        }
    }
    
    config_file = config_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_ni_daq():
    """Mock National Instruments DAQ for testing."""
    with patch('retrofitkit.drivers.daq.ni.nidaqmx') as mock_nidaqmx:
        mock_task = Mock()
        mock_task.read.return_value = 2.5
        mock_task.write.return_value = None
        
        mock_nidaqmx.Task.return_value = mock_task
        mock_nidaqmx.constants.AcquisitionType.FINITE = 'finite'
        
        yield mock_nidaqmx


@pytest.fixture  
def mock_ocean_raman():
    """Mock Ocean Optics Raman spectrometer for testing."""
    with patch('retrofitkit.drivers.raman.vendor_ocean_optics.sb') as mock_sb:
        mock_spec = Mock()
        mock_spec.wavelengths.return_value = [530.0, 532.0, 534.0]
        mock_spec.intensities.return_value = [1000.0, 1500.0, 1200.0]
        mock_spec.integration_time_micros = Mock()
        
        mock_sb.list_devices.return_value = ['USB4000_12345']
        mock_sb.Spectrometer.return_value = mock_spec
        
        yield mock_sb


@pytest.fixture
def sample_recipe():
    """Sample recipe for testing."""
    return {
        "name": "Test Crystallization Recipe",
        "description": "Test recipe for unit testing",
        "steps": [
            {
                "type": "bias_set", 
                "voltage": 2.5,
                "description": "Set initial bias"
            },
            {
                "type": "hold",
                "duration": 5,
                "description": "Short hold for testing"
            },
            {
                "type": "wait_for_raman",
                "condition": {
                    "type": "peak_threshold",
                    "line_nm": 532.0,
                    "threshold": 1400.0,
                    "direction": "above"
                },
                "timeout": 30,
                "description": "Wait for test peak"
            }
        ]
    }


# mock_app fixture removed


@pytest.fixture
def sample_spectral_data():
    """Sample spectral data for testing."""
    return {
        "wavelengths": list(range(400, 800, 2)),  # 400-800 nm, 2nm steps
        "intensities": [1000 + 500 * (1 if 530 <= w <= 534 else 0) + 
                       10 * (w % 50) for w in range(400, 800, 2)],
        "timestamp": "2025-09-08T12:00:00Z",
        "integration_time_ms": 100,
        "metadata": {
            "temperature": 25.0,
            "laser_power": 100.0
        }
    }


@pytest.fixture
def hardware_detection_mock():
    """Mock hardware detection for testing."""
    mock_hardware = {
        "ni_devices": ["Dev1", "Dev2"],
        "ni_info": {
            "Dev1": {
                "ai": ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"],
                "ao": ["Dev1/ao0", "Dev1/ao1"], 
                "di": ["Dev1/port0/line0", "Dev1/port0/line1", "Dev1/port0/line2"],
                "do": ["Dev1/port0/line0", "Dev1/port0/line1", "Dev1/port0/line2"]
            },
            "Dev2": {
                "ai": ["Dev2/ai0", "Dev2/ai1"],
                "ao": ["Dev2/ao0"],
                "di": ["Dev2/port0/line0", "Dev2/port0/line1"],
                "do": ["Dev2/port0/line0", "Dev2/port0/line1"]
            }
        },
        "ocean_devices": ["USB4000_12345", "USB2000_67890"]
    }
    
    with patch('nidaqmx.system.System.local') as mock_system, \
         patch('seabreeze.spectrometers.list_devices') as mock_ocean:
        
        # Setup NI mock
        mock_devices = []
        for dev_name, dev_info in mock_hardware["ni_info"].items():
            mock_dev = Mock()
            mock_dev.name = dev_name
            mock_dev.product_type = f"NI USB-6343"
            mock_dev.ai_physical_chans = [Mock(name=ch.split('/')[-1]) for ch in dev_info["ai"]]
            mock_dev.ao_physical_chans = [Mock(name=ch.split('/')[-1]) for ch in dev_info["ao"]]
            mock_dev.di_lines = [Mock(name=ch.split('/')[-1]) for ch in dev_info["di"]]
            mock_dev.do_lines = [Mock(name=ch.split('/')[-1]) for ch in dev_info["do"]]
            mock_devices.append(mock_dev)
            
        mock_system.return_value.devices = mock_devices
        
        # Setup Ocean mock
        mock_ocean.return_value = mock_hardware["ocean_devices"]
        
        yield mock_hardware