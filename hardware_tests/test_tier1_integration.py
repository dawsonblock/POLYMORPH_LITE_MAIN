"""
Integration tests for Tier-1 Hardware Stack.

Tests the complete DAQ+Raman pipeline including:
- Device discovery
- DAQ operations (AO/AI)
- Raman acquisition
- Synchronized multi-device workflows
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from retrofitkit.drivers.discovery import (
    DeviceDiscoveryService,
    DeviceType,
    DeviceStatus,
    DiscoveredDevice
)
from retrofitkit.core.app import AppContext
from retrofitkit.core.config import PolymorphConfig
from retrofitkit.core.recipe import Recipe
from datetime import datetime


@pytest.fixture
def mock_ni_device():
    """Mock NI DAQ device."""
    device = Mock()
    device.name = "Dev1"
    device.product_type = "USB-6343"
    device.ai_physical_chans = [Mock() for _ in range(8)]
    device.ao_physical_chans = [Mock() for _ in range(4)]
    device.di_lines = [Mock() for _ in range(8)]
    device.do_lines = [Mock() for _ in range(8)]
    device.serial_num = "12345678"
    return device


@pytest.fixture
def mock_ocean_device():
    """Mock Ocean Optics spectrometer."""
    device = Mock()
    device.model = "USB2000+"
    device.serial_number = "USB2+H12345"
    device.minimum_integration_time_micros = 1000
    device.wavelengths.return_value = [i for i in range(200, 1100)]
    device.close = Mock()
    return device


@pytest.mark.tier1
class TestDeviceDiscovery:
    """Test device discovery functionality."""
    
    def test_ni_daq_discovery(self, mock_ni_device):
        """Test NI DAQ device discovery."""
        with patch('nidaqmx.system.System.local') as mock_system:
            mock_system.return_value.devices = [mock_ni_device]
            mock_system.return_value.driver_version.major_version = 21
            
            service = DeviceDiscoveryService()
            results = service.discover_all()
            
            assert DeviceType.NI_DAQ in results
            assert len(results[DeviceType.NI_DAQ]) == 1
            
            device = results[DeviceType.NI_DAQ][0]
            assert device.device_id == "Dev1"
            assert device.device_type == DeviceType.NI_DAQ
            assert device.capabilities["ai_channels"] == 8
            assert device.capabilities["ao_channels"] == 4
    
    def test_ocean_optics_discovery(self, mock_ocean_device):
        """Test Ocean Optics spectrometer discovery."""
        with patch('seabreeze.spectrometers.list_devices') as mock_list:
            with patch('seabreeze.spectrometers.Spectrometer') as mock_spec_class:
                mock_list.return_value = [Mock()]
                mock_spec_class.return_value = mock_ocean_device
                
                service = DeviceDiscoveryService()
                results = service.discover_all()
                
                assert DeviceType.OCEAN_OPTICS in results
                assert len(results[DeviceType.OCEAN_OPTICS]) == 1
                
                device = results[DeviceType.OCEAN_OPTICS][0]
                assert device.device_type == DeviceType.OCEAN_OPTICS
                assert device.name == "USB2000+"
                assert device.capabilities["pixels"] == 900
    
    def test_tier1_device_pairing(self, mock_ni_device, mock_ocean_device):
        """Test Tier-1 device pair discovery."""
        with patch('nidaqmx.system.System.local') as mock_ni_system:
            with patch('seabreeze.spectrometers.list_devices') as mock_ocean_list:
                with patch('seabreeze.spectrometers.Spectrometer') as mock_spec_class:
                    mock_ni_system.return_value.devices = [mock_ni_device]
                    mock_ni_system.return_value.driver_version.major_version = 21
                    mock_ocean_list.return_value = [Mock()]
                    mock_spec_class.return_value = mock_ocean_device
                    
                    service = DeviceDiscoveryService()
                    service.discover_all()
                    
                    tier1_devices = service.get_tier1_devices()
                    
                    assert tier1_devices["daq"] is not None
                    assert tier1_devices["raman"] is not None
                    assert tier1_devices["daq"].device_type == DeviceType.NI_DAQ
                    assert tier1_devices["raman"].device_type == DeviceType.OCEAN_OPTICS


@pytest.mark.tier1
class TestDAQOperations:
    """Test DAQ analog operations."""
    
    @pytest.mark.asyncio
    async def test_daq_analog_output(self):
        """Test DAQ analog output setting."""
        from retrofitkit.drivers.daq.factory import make_daq
        from retrofitkit.core.config import DAQConfig
        
        config = DAQConfig(backend="simulator")
        daq = make_daq(Mock(daq=config))
        
        # Test voltage setting
        await daq.set_voltage(2.5)
        
        # In simulator, should not raise error
        assert True
    
    @pytest.mark.asyncio
    async def test_daq_analog_input(self):
        """Test DAQ analog input reading."""
        from retrofitkit.drivers.daq.factory import make_daq
        from retrofitkit.core.config import DAQConfig
        
        config = DAQConfig(backend="simulator")
        daq = make_daq(Mock(daq=config))
        
        # Test reading
        value = await daq.read_ai(channel=0, samples=10)
        
        assert isinstance(value, (int, float))


@pytest.mark.tier1
class TestRamanAcquisition:
    """Test Raman spectroscopy acquisition."""
    
    @pytest.mark.asyncio
    async def test_raman_acquire(self):
        """Test Raman spectrum acquisition."""
        from retrofitkit.drivers.raman.factory import make_raman
        from retrofitkit.core.config import RamanConfig
        
        config = RamanConfig(provider="simulator")
        raman = make_raman(Mock(raman=config))
        
        # Acquire spectrum
        spectrum = await raman.acquire(integration_time_ms=100)
        
        assert "wavelengths" in spectrum
        assert "intensities" in spectrum
        assert len(spectrum["wavelengths"]) > 0
        assert len(spectrum["intensities"]) > 0


@pytest.mark.tier1
class TestSynchronizedWorkflow:
    """Test synchronized multi-device workflows."""
    
    @pytest.mark.asyncio
    async def test_daq_raman_workflow(self):
        """Test combined DAQ+Raman workflow execution."""
        from retrofitkit.core.orchestrator import Orchestrator
        from retrofitkit.core.recipe import Recipe
        
        # Load Tier-1 workflow
        workflow_path = "workflows/tier1_daq_raman_sweep.yaml"
        
        # Create orchestrator with test config
        config = PolymorphConfig()
        config.daq.backend = "simulator"
        config.raman.provider = "simulator"
        
        ctx = AppContext(config=config)
        orchestrator = Orchestrator(ctx)
        
        try:
            recipe = Recipe.from_yaml(workflow_path)
            
            # Execute workflow in simulation mode
            run_id = await orchestrator.run(
                recipe=recipe,
                operator_email="test@example.com",
                simulation=True
            )
            
            assert run_id is not None
            
        except FileNotFoundError:
            pytest.skip("Workflow file not found - expected in deployment")


@pytest.mark.tier1
class TestDeviceAllocation:
    """Test device allocation and release."""
    
    def test_device_allocation(self):
        """Test allocating devices for exclusive use."""
        service = DeviceDiscoveryService()
        
        # Create mock device
        device = DiscoveredDevice(
            device_id="test_dev",
            device_type=DeviceType.NI_DAQ,
            name="Test Device",
            status=DeviceStatus.AVAILABLE,
            capabilities={},
            metadata={},
            discovered_at=datetime.now()
        )
        service.registry.register(device)
        
        # Allocate device
        success = service.allocate_device("test_dev")
        assert success is True
        
        # Try to allocate again (should fail)
        success = service.allocate_device("test_dev")
        assert success is False
        
        # Release device
        service.release_device("test_dev")
        
        # Should be available again
        success = service.allocate_device("test_dev")
        assert success is True


@pytest.mark.tier1
@pytest.mark.integration
class TestTier1Stack:
    """Full integration test for Tier-1 stack."""
    
    @pytest.mark.asyncio
    async def test_tier1_initialization(self):
        """Test complete Tier-1 stack initialization."""
        config = PolymorphConfig()
        config.daq.backend = "simulator"
        config.raman.provider = "simulator"
        
        service = DeviceDiscoveryService()
        
        # In simulation mode, we won't discover real devices
        # but we can test the service works
        results = service.discover_all()
        
        assert isinstance(results, dict)
        assert DeviceType.NI_DAQ in results
        assert DeviceType.OCEAN_OPTICS in results
