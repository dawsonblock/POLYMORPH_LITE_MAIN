import pytest
import os
import yaml
from retrofitkit.core.config_loader import ConfigLoader
from retrofitkit.core.driver_router import DriverRouter
from retrofitkit.core.config import PolymorphConfig

# --- ConfigLoader Tests ---

def test_config_loader_base(tmp_path):
    # Create a dummy base config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    base_config = config_dir / "config.yaml"
    
    data = {
        "system": {"name": "TestSystem", "mode": "testing"},
        "daq": {"backend": "simulator"}
    }
    with open(base_config, "w") as f:
        yaml.dump(data, f)
        
    loader = ConfigLoader(base_config_path=str(base_config))
    loader.load_base()
    
    config = loader.resolve()
    assert config.system.name == "TestSystem"
    assert config.daq.backend == "simulator"

def test_config_loader_overlay(tmp_path):
    # Base config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    base_config = config_dir / "config.yaml"
    with open(base_config, "w") as f:
        yaml.dump({"system": {"name": "Base"}}, f)
        
    # Overlay config
    overlay = tmp_path / "overlay.yaml"
    with open(overlay, "w") as f:
        yaml.dump({"system": {"name": "Overridden"}}, f)
        
    loader = ConfigLoader(base_config_path=str(base_config))
    loader.load_base()
    loader.apply_overlay(str(overlay))
    
    config = loader.resolve()
    assert config.system.name == "Overridden"

# --- DriverRouter Tests ---

class MockDriver:
    def __init__(self, config):
        self.config = config

def test_driver_router_registration():
    router = DriverRouter()
    router.register_driver("daq", "mock", MockDriver)
    
    assert router._registry["daq"]["mock"] == MockDriver

def test_driver_router_resolution(tmp_path):
    router = DriverRouter()
    router.register_driver("daq", "mock", MockDriver)
    
    # Create config that selects "mock" backend
    # We need to trick the config into having backend="mock"
    # Since DAQConfig.backend is typed, we might need to use a valid value or mock the config object
    
    # Let's mock the config object directly for simplicity
    class MockConfig:
        class DAQ:
            backend = "mock"
        daq = DAQ()
        
        class Raman:
            provider = "none"
        raman = Raman()
        
    config = MockConfig()
    
    driver = router.get_driver("daq", config)
    assert isinstance(driver, MockDriver)
    assert driver.config == config

def test_driver_router_missing_driver():
    router = DriverRouter()
    
    class MockConfig:
        class DAQ:
            backend = "nonexistent"
        daq = DAQ()
        
    with pytest.raises(ValueError):
        router.get_driver("daq", MockConfig())
