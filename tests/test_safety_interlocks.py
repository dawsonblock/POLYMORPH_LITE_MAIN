"""
Tests for safety interlock system.

This module tests:
- Emergency stop detection
- Door open detection
- Safety line reading
- DAQ integration
- Fail-safe behavior
"""
import pytest
from unittest.mock import Mock, MagicMock
from retrofitkit.safety.interlocks import Interlocks
from retrofitkit.core.app import AppContext, Config, SafetyCfg, DAQCfg, SystemCfg, SecurityCfg, RamanCfg, GatingCfg


@pytest.fixture
def mock_config():
    """Create a mock configuration with safety interlocks."""
    config = Config(
        system=SystemCfg(
            name="Test System",
            mode="test",
            timezone="UTC",
            data_dir="/tmp/test",
            logs_dir="/tmp/logs"
        ),
        security=SecurityCfg(
            password_policy={},
            two_person_signoff=True,
            jwt_exp_minutes=480,
            rsa_private_key="",
            rsa_public_key=""
        ),
        daq=DAQCfg(
            backend="simulator",
            ni={},
            redpitaya={},
            simulator={}
        ),
        raman=RamanCfg(
            provider="simulator",
            simulator={},
            vendor={}
        ),
        gating=GatingCfg(rules=[]),
        safety=SafetyCfg(
            interlocks={
                "estop_line": 0,
                "door_line": 1
            },
            watchdog_seconds=1.0
        )
    )
    return config


@pytest.fixture
def mock_daq():
    """Create a mock DAQ device."""
    daq = Mock()
    daq.read_di = Mock()
    return daq


@pytest.fixture
def app_context(mock_config):
    """Create a mock AppContext."""
    ctx = AppContext(config=mock_config)
    return ctx


@pytest.fixture
def interlocks(app_context, mock_daq, monkeypatch):
    """Create Interlocks instance with mocked DAQ."""
    def mock_make_daq(config):
        return mock_daq

    monkeypatch.setattr("retrofitkit.safety.interlocks.make_daq", mock_make_daq)
    return Interlocks(app_context)


class TestInterlockInitialization:
    """Test cases for interlock system initialization."""

    def test_interlocks_initializes_with_context(self, app_context, mock_daq, monkeypatch):
        """Test that Interlocks initializes with AppContext."""
        monkeypatch.setattr("retrofitkit.safety.interlocks.make_daq", lambda cfg: mock_daq)

        interlocks = Interlocks(app_context)

        assert interlocks.ctx == app_context
        assert interlocks.daq is not None

    def test_interlocks_creates_daq_instance(self, app_context, mock_daq, monkeypatch):
        """Test that Interlocks creates DAQ instance from config."""
        make_daq_called = []

        def mock_make_daq(config):
            make_daq_called.append(config)
            return mock_daq

        monkeypatch.setattr("retrofitkit.safety.interlocks.make_daq", mock_make_daq)

        interlocks = Interlocks(app_context)

        assert len(make_daq_called) == 1
        assert make_daq_called[0] == app_context.config


class TestEmergencyStopDetection:
    """Test cases for emergency stop (E-STOP) detection."""

    def test_estop_triggered_when_line_high(self, interlocks, mock_daq):
        """Test that E-STOP is detected when line reads high."""
        mock_daq.read_di.return_value = 1

        result = interlocks.estop_triggered()

        assert result is True
        mock_daq.read_di.assert_called_once_with(0)  # estop_line = 0

    def test_estop_not_triggered_when_line_low(self, interlocks, mock_daq):
        """Test that E-STOP is not triggered when line reads low."""
        mock_daq.read_di.return_value = 0

        result = interlocks.estop_triggered()

        assert result is False
        mock_daq.read_di.assert_called_once_with(0)

    def test_estop_reads_correct_line(self, interlocks, mock_daq, app_context):
        """Test that E-STOP reads from configured line number."""
        # Change estop_line in config
        app_context.config.safety.interlocks["estop_line"] = 5
        mock_daq.read_di.return_value = 1

        interlocks.estop_triggered()

        mock_daq.read_di.assert_called_once_with(5)

    def test_estop_converts_to_bool(self, interlocks, mock_daq):
        """Test that E-STOP return value is boolean."""
        # Test various truthy values
        for value in [1, 5, True, "1"]:
            mock_daq.read_di.return_value = value
            result = interlocks.estop_triggered()
            assert isinstance(result, bool)
            assert result is True

        # Test various falsy values
        for value in [0, False, ""]:
            mock_daq.read_di.return_value = value
            result = interlocks.estop_triggered()
            assert isinstance(result, bool)
            assert result is False


class TestDoorOpenDetection:
    """Test cases for door open detection."""

    def test_door_open_when_line_high(self, interlocks, mock_daq):
        """Test that door open is detected when line reads high."""
        mock_daq.read_di.return_value = 1

        result = interlocks.door_open()

        assert result is True
        mock_daq.read_di.assert_called_once_with(1)  # door_line = 1

    def test_door_closed_when_line_low(self, interlocks, mock_daq):
        """Test that door is not open when line reads low."""
        mock_daq.read_di.return_value = 0

        result = interlocks.door_open()

        assert result is False
        mock_daq.read_di.assert_called_once_with(1)

    def test_door_reads_correct_line(self, interlocks, mock_daq, app_context):
        """Test that door sensor reads from configured line number."""
        # Change door_line in config
        app_context.config.safety.interlocks["door_line"] = 7
        mock_daq.read_di.return_value = 1

        interlocks.door_open()

        mock_daq.read_di.assert_called_once_with(7)

    def test_door_converts_to_bool(self, interlocks, mock_daq):
        """Test that door open return value is boolean."""
        # Test truthy values
        for value in [1, 100, True]:
            mock_daq.read_di.return_value = value
            result = interlocks.door_open()
            assert isinstance(result, bool)
            assert result is True

        # Test falsy values
        for value in [0, False]:
            mock_daq.read_di.return_value = value
            result = interlocks.door_open()
            assert isinstance(result, bool)
            assert result is False


class TestInterlockStates:
    """Test cases for combined interlock states."""

    def test_both_interlocks_safe(self, interlocks, mock_daq):
        """Test when both interlocks are in safe state."""
        mock_daq.read_di.return_value = 0

        estop = interlocks.estop_triggered()
        door = interlocks.door_open()

        assert estop is False
        assert door is False

    def test_both_interlocks_triggered(self, interlocks, mock_daq):
        """Test when both interlocks are triggered."""
        mock_daq.read_di.return_value = 1

        estop = interlocks.estop_triggered()
        door = interlocks.door_open()

        assert estop is True
        assert door is True

    def test_only_estop_triggered(self, interlocks, mock_daq):
        """Test when only E-STOP is triggered."""
        def read_di_side_effect(line):
            if line == 0:  # estop_line
                return 1
            return 0

        mock_daq.read_di.side_effect = read_di_side_effect

        estop = interlocks.estop_triggered()
        door = interlocks.door_open()

        assert estop is True
        assert door is False

    def test_only_door_open(self, interlocks, mock_daq):
        """Test when only door is open."""
        def read_di_side_effect(line):
            if line == 1:  # door_line
                return 1
            return 0

        mock_daq.read_di.side_effect = read_di_side_effect

        estop = interlocks.estop_triggered()
        door = interlocks.door_open()

        assert estop is False
        assert door is True

    def test_multiple_sequential_checks(self, interlocks, mock_daq):
        """Test multiple sequential interlock checks."""
        # First check - safe
        mock_daq.read_di.return_value = 0
        assert interlocks.estop_triggered() is False

        # Second check - triggered
        mock_daq.read_di.return_value = 1
        assert interlocks.estop_triggered() is True

        # Third check - safe again
        mock_daq.read_di.return_value = 0
        assert interlocks.estop_triggered() is False


class TestErrorHandling:
    """Test cases for error handling and fail-safe behavior."""

    def test_estop_with_daq_read_error(self, interlocks, mock_daq):
        """Test E-STOP behavior when DAQ read fails."""
        mock_daq.read_di.side_effect = Exception("DAQ communication error")

        # Should raise exception (fail-safe: don't hide errors)
        with pytest.raises(Exception) as exc_info:
            interlocks.estop_triggered()

        assert "DAQ communication error" in str(exc_info.value)

    def test_door_with_daq_read_error(self, interlocks, mock_daq):
        """Test door sensor behavior when DAQ read fails."""
        mock_daq.read_di.side_effect = Exception("DAQ communication error")

        with pytest.raises(Exception) as exc_info:
            interlocks.door_open()

        assert "DAQ communication error" in str(exc_info.value)

    def test_interlocks_with_none_return(self, interlocks, mock_daq):
        """Test interlock behavior when DAQ returns None."""
        mock_daq.read_di.return_value = None

        # bool(None) is False, should not trigger interlock
        assert interlocks.estop_triggered() is False
        assert interlocks.door_open() is False


class TestConfigurationVariations:
    """Test cases for different configuration scenarios."""

    def test_interlocks_with_same_line_number(self, app_context, mock_daq, monkeypatch):
        """Test when estop and door use same line (unusual but valid)."""
        app_context.config.safety.interlocks = {
            "estop_line": 0,
            "door_line": 0  # Same line
        }

        monkeypatch.setattr("retrofitkit.safety.interlocks.make_daq", lambda cfg: mock_daq)
        interlocks = Interlocks(app_context)

        mock_daq.read_di.return_value = 1

        # Both should read same value
        assert interlocks.estop_triggered() is True
        assert interlocks.door_open() is True

    def test_interlocks_with_high_line_numbers(self, app_context, mock_daq, monkeypatch):
        """Test interlocks with high line numbers."""
        app_context.config.safety.interlocks = {
            "estop_line": 100,
            "door_line": 200
        }

        monkeypatch.setattr("retrofitkit.safety.interlocks.make_daq", lambda cfg: mock_daq)
        interlocks = Interlocks(app_context)

        mock_daq.read_di.return_value = 1

        interlocks.estop_triggered()
        mock_daq.read_di.assert_called_with(100)

        interlocks.door_open()
        mock_daq.read_di.assert_called_with(200)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_startup_safety_check(self, interlocks, mock_daq):
        """Test typical startup safety check sequence."""
        mock_daq.read_di.return_value = 0

        # Check all interlocks at startup
        estop_safe = not interlocks.estop_triggered()
        door_safe = not interlocks.door_open()

        assert estop_safe is True
        assert door_safe is True

    def test_emergency_shutdown_detection(self, interlocks, mock_daq):
        """Test detection of emergency shutdown condition."""
        # Simulate E-STOP being pressed during operation
        mock_daq.read_di.return_value = 1

        if interlocks.estop_triggered():
            # Emergency shutdown should be initiated
            shutdown_needed = True
        else:
            shutdown_needed = False

        assert shutdown_needed is True

    def test_door_interlock_during_operation(self, interlocks, mock_daq):
        """Test door opening during operation."""
        def read_di_side_effect(line):
            if line == 1:  # door_line
                return 1  # Door opened
            return 0

        mock_daq.read_di.side_effect = read_di_side_effect

        # E-STOP is safe
        assert interlocks.estop_triggered() is False

        # But door is open - should pause operation
        if interlocks.door_open():
            operation_paused = True
        else:
            operation_paused = False

        assert operation_paused is True

    def test_recovery_from_interlock(self, interlocks, mock_daq):
        """Test recovery sequence after interlock clears."""
        # Initially triggered
        mock_daq.read_di.return_value = 1
        assert interlocks.estop_triggered() is True

        # Cleared
        mock_daq.read_di.return_value = 0
        assert interlocks.estop_triggered() is False

        # Safe to resume
        can_resume = not interlocks.estop_triggered() and not interlocks.door_open()
        assert can_resume is True
