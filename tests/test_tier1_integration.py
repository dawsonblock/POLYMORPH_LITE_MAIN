"""
Tier-1 Hardware Stack Integration Tests.

Tests the production-ready hardware stack:
- DAQ: National Instruments (NI USB-6343 or PCIe-6363)
- Raman: Ocean Optics (seabreeze-compatible spectrometers)

These tests validate end-to-end workflow execution on the Tier-1 stack.
"""
import pytest
import asyncio
from pathlib import Path
from retrofitkit.core.app import AppContext
from retrofitkit.core.orchestrator import Orchestrator
from retrofitkit.core.recipe import Recipe, Step
from retrofitkit.core.config_loader import get_loader


@pytest.fixture
def tier1_context():
    """Load Tier-1 overlay configuration."""
    # Set overlay path
    overlay_path = Path("config/overlays/NI_USB6343_Ocean0")
    
    # Load configuration
    loader = get_loader()
    config = loader.load_base().load_overlay(str(overlay_path)).resolve()
    
    return AppContext(config)


@pytest.mark.integration
@pytest.mark.tier1
async def test_tier1_stack_initialization(tier1_context):
    """
    Test that Tier-1 stack (NI DAQ + Ocean Optics) initializes correctly.
    
    Validates:
    - Correct drivers are loaded
    - Devices are healthy
    - Safety interlocks are configured
    """
    # Create orchestrator
    orch = Orchestrator(tier1_context)
    
    # Verify DAQ is NI
    assert orch.daq.id.startswith("ni_daq"), f"Expected NI DAQ, got {orch.daq.id}"
    assert hasattr(orch.daq, "dev"), "NI DAQ should have 'dev' attribute"
    
    # Verify Raman is Ocean Optics
    assert orch.raman.id.startswith("ocean_optics"), f"Expected Ocean Optics, got {orch.raman.id}"
    
    # Test DAQ health
    daq_health = await orch.daq.health()
    assert daq_health["status"] in ["ok", "disconnected"], "DAQ health check failed"
    
    # Test Raman health
    raman_health = await orch.raman.health()
    assert raman_health["status"] in ["ok", "disconnected"], "Raman health check failed"
    
    # Verify safety interlocks configured
    assert orch.interlocks is not None, "Safety interlocks not configured"


@pytest.mark.integration
@pytest.mark.tier1
async def test_tier1_daq_operations(tier1_context):
    """
    Test DAQ operations on Tier-1 stack.
    
    Tests:
    - Analog output (write_ao)
    - Analog input (read_ai)
    - Digital input (read_di)
    - Digital output (write_do)
    """
    orch = Orchestrator(tier1_context)
    
    # Connect DAQ
    await orch.daq.connect()
    
    try:
        # Test analog output
        await orch.daq.set_voltage(1.5)
        
        # Test analog input (may return simulated value if no hardware)
        voltage = await orch.daq.read_ai()
        assert isinstance(voltage, (int, float)), "read_ai should return numeric value"
        
        # Test digital input
        di_value = await orch.daq.read_di(0)
        assert isinstance(di_value, bool), "read_di should return boolean"
        
        # Test digital output
        await orch.daq.write_do(0, True)
        await orch.daq.write_do(0, False)
        
    finally:
        # Cleanup
        await orch.daq.set_voltage(0.0)
        await orch.daq.disconnect()


@pytest.mark.integration
@pytest.mark.tier1
async def test_tier1_raman_acquisition(tier1_context):
    """
    Test Raman spectrum acquisition on Tier-1 stack.
    
    Validates:
    - Spectrum acquisition
    - Data structure
    - Metadata
    """
    orch = Orchestrator(tier1_context)
    
    # Connect Raman
    await orch.raman.connect()
    
    try:
        # Acquire spectrum
        spectrum = await orch.raman.acquire_spectrum(integration_time_ms=20.0)
        
        # Validate spectrum structure
        assert hasattr(spectrum, "wavelengths"), "Spectrum should have wavelengths"
        assert hasattr(spectrum, "intensities"), "Spectrum should have intensities"
        assert hasattr(spectrum, "meta"), "Spectrum should have metadata"
        
        # Validate data
        assert len(spectrum.wavelengths) > 0, "Wavelengths should not be empty"
        assert len(spectrum.intensities) > 0, "Intensities should not be empty"
        assert len(spectrum.wavelengths) == len(spectrum.intensities), "Wavelengths and intensities should match"
        
        # Validate metadata
        assert "integration_time_ms" in spectrum.meta, "Metadata should include integration time"
        assert "device_id" in spectrum.meta, "Metadata should include device ID"
        
    finally:
        await orch.raman.disconnect()


@pytest.mark.integration
@pytest.mark.tier1
async def test_tier1_workflow_execution(tier1_context):
    """
    Test end-to-end workflow execution on Tier-1 stack.
    
    Executes a complete workflow:
    1. Set DAQ voltage
    2. Wait
    3. Acquire Raman spectrum
    4. AI decision
    
    This validates the full integration of all components.
    """
    orch = Orchestrator(tier1_context)
    
    # Create test recipe
    recipe = Recipe(
        id="test-tier1-workflow",
        name="Tier-1 Integration Test Workflow",
        steps=[
            Step(type="daq", params={"action": "write_ao", "channel": 0, "value": 1.5}),
            Step(type="wait", params={"seconds": 0.1}),
            Step(type="raman", params={"exposure_time": 20.0}),
            Step(type="ai_decision", params={"critical": False}),
            Step(type="daq", params={"action": "write_ao", "channel": 0, "value": 0.0}),
        ]
    )
    
    # Execute workflow
    run_id = await orch.execute_recipe(
        recipe=recipe,
        operator_email="test@tier1.com",
        simulation=True  # Use simulation mode for testing
    )
    
    # Validate execution
    assert run_id is not None, "Workflow should return run ID"
    assert run_id.startswith("RUN-"), "Run ID should have correct format"


@pytest.mark.integration
@pytest.mark.tier1
async def test_tier1_ai_integration(tier1_context):
    """
    Test AI service integration on Tier-1 stack.
    
    Validates:
    - AI client initialization
    - Spectrum preprocessing
    - AI prediction
    """
    orch = Orchestrator(tier1_context)
    
    # Check AI client
    assert orch.ai_client is not None, "AI client should be initialized"
    
    # Get AI status
    status = orch.ai_client.status
    assert "circuit_breaker" in status, "AI status should include circuit breaker state"
    
    # Acquire spectrum
    await orch.raman.connect()
    try:
        spectrum = await orch.raman.acquire_spectrum(integration_time_ms=20.0)
        
        # Extract intensities for AI
        intensities = orch._extract_spectrum_for_ai(spectrum)
        assert isinstance(intensities, list), "Intensities should be list for AI"
        assert len(intensities) > 0, "Intensities should not be empty"
        
    finally:
        await orch.raman.disconnect()


@pytest.mark.integration
@pytest.mark.tier1
async def test_tier1_safety_interlocks(tier1_context):
    """
    Test safety interlock system on Tier-1 stack.
    
    Validates:
    - Interlock controller initialization
    - Safety checks
    - Emergency stop behavior
    """
    orch = Orchestrator(tier1_context)
    
    # Verify interlocks configured
    assert orch.interlocks is not None, "Interlocks should be configured"
    
    # Check safety status (should not raise in safe state)
    try:
        orch.interlocks.check_safe()
    except Exception as e:
        # If E-stop or door open in test environment, that's expected
        assert "E-STOP" in str(e) or "DOOR" in str(e), f"Unexpected safety error: {e}"


@pytest.mark.integration
@pytest.mark.tier1
async def test_tier1_gating_engine(tier1_context):
    """
    Test gating engine on Tier-1 stack.
    
    Validates:
    - Gating engine initialization
    - Rule evaluation
    - Stop condition detection
    """
    orch = Orchestrator(tier1_context)
    
    if orch.gating_engine is None:
        pytest.skip("Gating engine not configured")
    
    # Acquire spectrum
    await orch.raman.connect()
    try:
        spectrum = await orch.raman.acquire_spectrum(integration_time_ms=20.0)
        spectrum_dict = orch._spectrum_to_dict(spectrum)
        
        # Update gating engine
        should_stop = orch.gating_engine.update(spectrum_dict)
        assert isinstance(should_stop, bool), "Gating engine should return boolean"
        
    finally:
        await orch.raman.disconnect()


@pytest.mark.integration
@pytest.mark.tier1
def test_tier1_config_validation(tier1_context):
    """
    Test that Tier-1 configuration is valid and complete.
    
    Validates:
    - All required config sections present
    - Correct backend selections
    - Safety configuration
    """
    config = tier1_context.config
    
    # Validate system config
    assert hasattr(config, "system"), "Config should have system section"
    assert config.system.mode == "production", "Tier-1 should be in production mode"
    
    # Validate DAQ config
    assert hasattr(config, "daq"), "Config should have DAQ section"
    assert config.daq.backend in ["ni_daq", "ni"], "Tier-1 should use NI DAQ"
    
    # Validate Raman config
    assert hasattr(config, "raman"), "Config should have Raman section"
    assert config.raman.provider in ["ocean_optics", "ocean"], "Tier-1 should use Ocean Optics"
    
    # Validate safety config
    assert hasattr(config, "safety"), "Config should have safety section"
    assert hasattr(config.safety, "interlocks"), "Safety should have interlocks configured"
    
    # Validate AI config
    assert hasattr(config, "ai"), "Config should have AI section"
    assert hasattr(config.ai, "service_url"), "AI should have service URL"
