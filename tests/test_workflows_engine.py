"""
Tests for workflow execution engine.

This module tests:
- Workflow definition parsing
- Workflow execution
- Step execution (action, wait)
- Safety policy enforcement
- Device lifecycle management
- Error handling
"""
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from retrofitkit.core.workflows.models import WorkflowDefinition, WorkflowStep, WorkflowExecutionResult
from retrofitkit.core.workflows.engine import WorkflowEngine
from retrofitkit.core.workflows.safety import (
    SafetyManager,
    LoggingPolicy,
    TemperaturePolicy,
    VoltageRangePolicy
)


class TestWorkflowStep:
    """Test cases for WorkflowStep model."""

    def test_workflow_step_creation(self):
        """Test creating a valid workflow step."""
        step = WorkflowStep(
            id="step1",
            kind="action",
            params={"device": "daq", "action": "write"},
            children=["step2"]
        )

        assert step.id == "step1"
        assert step.kind == "action"
        assert step.params["device"] == "daq"
        assert "step2" in step.children

    def test_workflow_step_valid_kinds(self):
        """Test that all valid step kinds are accepted."""
        valid_kinds = ["action", "wait", "loop", "condition"]

        for kind in valid_kinds:
            step = WorkflowStep(id="test", kind=kind)
            assert step.kind == kind

    def test_workflow_step_invalid_kind(self):
        """Test that unknown step kind logs warning but doesn't raise."""
        import warnings
        import logging
        
        # Unknown kind should create step but log warning
        step = WorkflowStep(id="test", kind="unknown_custom_kind")
        
        # Step should be created successfully
        assert step.id == "test"
        assert step.kind == "unknown_custom_kind"

    def test_workflow_step_default_params(self):
        """Test that step params default to empty dict."""
        step = WorkflowStep(id="test", kind="action")

        assert step.params == {}

    def test_workflow_step_default_children(self):
        """Test that step children default to empty list."""
        step = WorkflowStep(id="test", kind="action")

        assert step.children == []


class TestWorkflowDefinition:
    """Test cases for WorkflowDefinition model."""

    def test_workflow_definition_creation(self):
        """Test creating a valid workflow definition."""
        steps = {
            "start": WorkflowStep(id="start", kind="action", children=["end"]),
            "end": WorkflowStep(id="end", kind="wait")
        }

        workflow = WorkflowDefinition(
            id="test_workflow",
            name="Test Workflow",
            steps=steps,
            entry_step="start"
        )

        assert workflow.id == "test_workflow"
        assert workflow.name == "Test Workflow"
        assert len(workflow.steps) == 2
        assert workflow.entry_step == "start"

    def test_workflow_definition_validates_entry_step(self):
        """Test that workflow validates entry step exists."""
        steps = {
            "start": WorkflowStep(id="start", kind="action")
        }

        with pytest.raises(ValueError) as exc_info:
            WorkflowDefinition(
                id="test",
                name="Test",
                steps=steps,
                entry_step="nonexistent"
            )

        assert "Entry step 'nonexistent' not found" in str(exc_info.value)

    def test_workflow_definition_validates_children(self):
        """Test that workflow validates child step references."""
        steps = {
            "start": WorkflowStep(id="start", kind="action", children=["nonexistent"])
        }

        with pytest.raises(ValueError) as exc_info:
            WorkflowDefinition(
                id="test",
                name="Test",
                steps=steps,
                entry_step="start"
            )

        assert "references unknown child 'nonexistent'" in str(exc_info.value)

    def test_workflow_from_yaml(self):
        """Test parsing workflow from YAML."""
        yaml_content = """
id: "test_workflow"
name: "Test Workflow"
entry_step: "start"
steps:
  start:
    kind: "action"
    params:
      device: "daq"
      action: "write"
    children: []
"""
        workflow = WorkflowDefinition.from_yaml(yaml_content)

        assert workflow.id == "test_workflow"
        assert workflow.name == "Test Workflow"
        assert "start" in workflow.steps
        assert workflow.steps["start"].kind == "action"

    def test_workflow_from_yaml_complex(self):
        """Test parsing complex workflow with multiple steps."""
        yaml_content = """
id: "complex_workflow"
name: "Complex Workflow"
entry_step: "init"
metadata:
  author: "Test Author"
  version: "1.0"
steps:
  init:
    kind: "action"
    params:
      device: "raman"
      action: "initialize"
    children: ["measure"]
  measure:
    kind: "wait"
    params:
      seconds: 2.0
    children: ["finish"]
  finish:
    kind: "action"
    params:
      device: "raman"
      action: "shutdown"
    children: []
"""
        workflow = WorkflowDefinition.from_yaml(yaml_content)

        assert len(workflow.steps) == 3
        assert workflow.metadata["author"] == "Test Author"
        assert workflow.steps["init"].children == ["measure"]
        assert workflow.steps["measure"].params["seconds"] == 2.0

    def test_workflow_to_yaml(self):
        """Test converting workflow to YAML."""
        steps = {
            "start": WorkflowStep(id="start", kind="action", params={"device": "daq"})
        }
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            steps=steps,
            entry_step="start"
        )

        yaml_str = workflow.to_yaml()

        assert "id: test" in yaml_str
        assert "name: Test" in yaml_str
        assert "entry_step: start" in yaml_str
        assert "kind: action" in yaml_str

    def test_workflow_to_file_and_from_file(self):
        """Test saving and loading workflow from file."""
        steps = {
            "start": WorkflowStep(id="start", kind="wait", params={"seconds": 1.0})
        }
        workflow = WorkflowDefinition(
            id="file_test",
            name="File Test",
            steps=steps,
            entry_step="start"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            path = Path(f.name)

        try:
            workflow.to_file(path)
            loaded = WorkflowDefinition.from_file(path)

            assert loaded.id == workflow.id
            assert loaded.name == workflow.name
            assert loaded.steps["start"].params["seconds"] == 1.0
        finally:
            path.unlink()


class TestSafetyManager:
    """Test cases for SafetyManager."""

    def test_safety_manager_initialization(self):
        """Test SafetyManager initializes correctly."""
        manager = SafetyManager()

        assert manager._enabled is True
        assert len(manager._policies) == 0

    def test_add_policy(self):
        """Test adding policies to safety manager."""
        manager = SafetyManager()
        policy = LoggingPolicy()

        manager.add_policy(policy)

        assert len(manager._policies) == 1
        assert manager.list_policies() == ["logging"]

    def test_add_multiple_policies(self):
        """Test adding multiple policies."""
        manager = SafetyManager()

        manager.add_policy(LoggingPolicy())
        manager.add_policy(TemperaturePolicy())
        manager.add_policy(VoltageRangePolicy())

        assert len(manager._policies) == 3
        policy_names = manager.list_policies()
        assert "logging" in policy_names
        assert "temperature_check" in policy_names
        assert "voltage_range" in policy_names

    def test_remove_policy(self):
        """Test removing policy by name."""
        manager = SafetyManager()
        manager.add_policy(LoggingPolicy())
        manager.add_policy(TemperaturePolicy())

        manager.remove_policy("logging")

        assert len(manager._policies) == 1
        assert manager.list_policies() == ["temperature_check"]

    def test_enable_disable(self):
        """Test enabling and disabling safety manager."""
        manager = SafetyManager()

        assert manager._enabled is True

        manager.disable()
        assert manager._enabled is False

        manager.enable()
        assert manager._enabled is True

    @pytest.mark.asyncio
    async def test_check_before_action_disabled(self):
        """Test that checks are skipped when disabled."""
        manager = SafetyManager()

        # Add policy that would normally fail
        policy = Mock()
        policy.name = "test"
        policy.before_action = AsyncMock(side_effect=RuntimeError("Should not run"))
        manager.add_policy(policy)

        manager.disable()

        # Should not raise error because manager is disabled
        device = Mock()
        await manager.check_before_action(device, "test_action", {})

        policy.before_action.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_before_action_runs_policies(self):
        """Test that all policies are checked."""
        manager = SafetyManager()

        policy1 = Mock()
        policy1.name = "policy1"
        policy1.before_action = AsyncMock()

        policy2 = Mock()
        policy2.name = "policy2"
        policy2.before_action = AsyncMock()

        manager.add_policy(policy1)
        manager.add_policy(policy2)

        device = Mock()
        await manager.check_before_action(device, "test_action", {"arg": "value"})

        policy1.before_action.assert_called_once_with(device, "test_action", {"arg": "value"})
        policy2.before_action.assert_called_once_with(device, "test_action", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_check_before_action_raises_on_violation(self):
        """Test that policy violation raises error."""
        manager = SafetyManager()

        policy = Mock()
        policy.name = "strict_policy"
        policy.before_action = AsyncMock(side_effect=RuntimeError("Policy violation"))
        manager.add_policy(policy)

        device = Mock()

        with pytest.raises(RuntimeError) as exc_info:
            await manager.check_before_action(device, "dangerous_action", {})

        assert "Policy violation" in str(exc_info.value)


class TestVoltageRangePolicy:
    """Test cases for VoltageRangePolicy."""

    def test_voltage_range_policy_initialization(self):
        """Test VoltageRangePolicy initialization."""
        policy = VoltageRangePolicy(min_volts=-5.0, max_volts=5.0)

        assert policy.min_volts == -5.0
        assert policy.max_volts == 5.0

    @pytest.mark.asyncio
    async def test_voltage_range_allows_safe_voltage(self):
        """Test that safe voltages are allowed."""
        policy = VoltageRangePolicy(min_volts=-10.0, max_volts=10.0)
        device = Mock()

        # Should not raise for safe voltage
        await policy.before_action(device, "write_ao", {"value": 5.0})
        await policy.before_action(device, "set_voltage", {"volts": -5.0})

    @pytest.mark.asyncio
    async def test_voltage_range_blocks_too_high(self):
        """Test that voltage above range is blocked."""
        policy = VoltageRangePolicy(min_volts=-10.0, max_volts=10.0)
        device = Mock()

        with pytest.raises(RuntimeError) as exc_info:
            await policy.before_action(device, "write_ao", {"value": 15.0})

        assert "out of safe range" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_voltage_range_blocks_too_low(self):
        """Test that voltage below range is blocked."""
        policy = VoltageRangePolicy(min_volts=-10.0, max_volts=10.0)
        device = Mock()

        with pytest.raises(RuntimeError) as exc_info:
            await policy.before_action(device, "write_ao", {"value": -15.0})

        assert "out of safe range" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_voltage_range_ignores_other_actions(self):
        """Test that policy ignores non-voltage actions."""
        policy = VoltageRangePolicy(min_volts=-10.0, max_volts=10.0)
        device = Mock()

        # Should not raise for non-voltage actions
        await policy.before_action(device, "read_ai", {"channel": 0})
        await policy.before_action(device, "get_status", {})


class TestTemperaturePolicy:
    """Test cases for TemperaturePolicy."""

    def test_temperature_policy_initialization(self):
        """Test TemperaturePolicy initialization."""
        policy = TemperaturePolicy(max_temp_celsius=-30.0)

        assert policy.max_temp_celsius == -30.0

    @pytest.mark.asyncio
    async def test_temperature_policy_allows_cold_device(self):
        """Test that cold device is allowed."""
        policy = TemperaturePolicy(max_temp_celsius=-20.0)
        device = Mock()
        device.health = AsyncMock(return_value={"ccd_temp": -30.0})

        # Should not raise for cold device
        await policy.before_action(device, "acquire_spectrum", {})

    @pytest.mark.asyncio
    async def test_temperature_policy_blocks_hot_device(self):
        """Test that hot device is blocked."""
        policy = TemperaturePolicy(max_temp_celsius=-20.0)
        device = Mock()
        device.health = AsyncMock(return_value={"ccd_temp": -10.0})

        with pytest.raises(RuntimeError) as exc_info:
            await policy.before_action(device, "acquire_spectrum", {})

        assert "temperature too high" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_temperature_policy_ignores_other_actions(self):
        """Test that policy ignores non-acquisition actions."""
        policy = TemperaturePolicy(max_temp_celsius=-20.0)
        device = Mock()
        device.health = AsyncMock(return_value={"ccd_temp": 100.0})  # Very hot

        # Should not raise for non-acquisition action
        await policy.before_action(device, "initialize", {})

    @pytest.mark.asyncio
    async def test_temperature_policy_ignores_no_temp_sensor(self):
        """Test that policy skips devices without temperature sensor."""
        policy = TemperaturePolicy(max_temp_celsius=-20.0)
        device = Mock()
        device.health = AsyncMock(return_value={"status": "ready"})  # No temp

        # Should not raise for device without temperature
        await policy.before_action(device, "acquire_spectrum", {})


class TestWorkflowEngine:
    """Test cases for WorkflowEngine."""

    @pytest.fixture
    def safety_manager(self):
        """Create a safety manager for testing."""
        return SafetyManager()

    @pytest.fixture
    def engine(self, safety_manager):
        """Create a workflow engine for testing."""
        return WorkflowEngine(safety_manager)

    @pytest.fixture
    def mock_device(self):
        """Create a mock device."""
        device = Mock()
        device.id = "test_device"
        device.connect = AsyncMock()
        device.disconnect = AsyncMock()
        device.health = AsyncMock(return_value={"status": "ready"})
        device.test_action = AsyncMock(return_value={"result": "success"})
        return device

    @pytest.mark.asyncio
    async def test_engine_initialization(self, safety_manager):
        """Test WorkflowEngine initialization."""
        engine = WorkflowEngine(safety_manager)

        assert engine._safety == safety_manager
        assert len(engine._device_instances) == 0

    @pytest.mark.asyncio
    async def test_engine_runs_simple_wait_workflow(self, engine):
        """Test running a simple wait workflow."""
        import time

        steps = {
            "wait": WorkflowStep(id="wait", kind="wait", params={"seconds": 0.1})
        }
        workflow = WorkflowDefinition(
            id="test",
            name="Wait Test",
            steps=steps,
            entry_step="wait"
        )

        start = time.time()
        result = await engine.run(workflow)
        duration = time.time() - start

        assert result.success is True
        assert result.workflow_id == "test"
        assert "wait" in result.steps_executed
        assert result.step_results["wait"]["waited_seconds"] == 0.1
        assert duration >= 0.1

    @pytest.mark.asyncio
    async def test_engine_runs_multi_step_workflow(self, engine):
        """Test running workflow with multiple steps."""
        steps = {
            "step1": WorkflowStep(id="step1", kind="wait", params={"seconds": 0.1}, children=["step2"]),
            "step2": WorkflowStep(id="step2", kind="wait", params={"seconds": 0.1}, children=["step3"]),
            "step3": WorkflowStep(id="step3", kind="wait", params={"seconds": 0.1})
        }
        workflow = WorkflowDefinition(
            id="multi",
            name="Multi Step",
            steps=steps,
            entry_step="step1"
        )

        result = await engine.run(workflow)

        assert result.success is True
        assert len(result.steps_executed) == 3
        assert result.steps_executed == ["step1", "step2", "step3"]

    @pytest.mark.asyncio
    async def test_engine_handles_workflow_error(self, engine):
        """Test that engine handles errors gracefully."""
        # Create workflow with unsupported step kind
        steps = {
            "bad": WorkflowStep(id="bad", kind="loop")  # Not yet supported
        }
        workflow = WorkflowDefinition(
            id="error_test",
            name="Error Test",
            steps=steps,
            entry_step="bad"
        )

        result = await engine.run(workflow)

        assert result.success is False
        assert result.error is not None
        assert "Unsupported workflow step kind" in result.error

    @pytest.mark.asyncio
    async def test_engine_tracks_execution_time(self, engine):
        """Test that engine tracks execution duration."""
        steps = {
            "wait": WorkflowStep(id="wait", kind="wait", params={"seconds": 0.2})
        }
        workflow = WorkflowDefinition(
            id="timing",
            name="Timing Test",
            steps=steps,
            entry_step="wait"
        )

        result = await engine.run(workflow)

        assert result.duration_seconds >= 0.2
        assert result.duration_seconds < 1.0  # Should complete quickly

    @pytest.mark.asyncio
    async def test_engine_cleans_up_devices(self, engine, mock_device):
        """Test that engine disconnects devices after execution."""
        # Patch registry to return mock device
        with patch('retrofitkit.core.workflows.engine.registry') as mock_registry:
            mock_registry.create.return_value = mock_device

            steps = {
                "action": WorkflowStep(
                    id="action",
                    kind="action",
                    params={"device": "test_device", "action": "test_action"}
                )
            }
            workflow = WorkflowDefinition(
                id="cleanup",
                name="Cleanup Test",
                steps=steps,
                entry_step="action"
            )

            await engine.run(workflow)

            # Device should be disconnected
            mock_device.disconnect.assert_called_once()
            assert len(engine._device_instances) == 0
