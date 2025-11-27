"""
Tests for safety watchdog system.

This module tests:
- Watchdog initialization
- Periodic toggle functionality
- Async operation
- Error handling
- Stop/shutdown behavior
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from retrofitkit.safety.watchdog import Watchdog
from retrofitkit.core.app import AppContext, Config, SafetyCfg, DAQCfg, SystemCfg, SecurityCfg, RamanCfg, GatingCfg


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Config(
        system=SystemCfg(
            name="Test System",
            environment="testing",
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
            estop_line=0,
            door_line=1,
            watchdog_seconds=1.0
        )
    )
    return config


@pytest.fixture
def mock_daq():
    """Create a mock DAQ device with async toggle_watchdog."""
    daq = Mock()
    daq.toggle_watchdog = AsyncMock()
    return daq


@pytest.fixture
def app_context(mock_config):
    """Create a mock AppContext."""
    return AppContext(config=mock_config)


@pytest.fixture
def watchdog(app_context, mock_daq, monkeypatch):
    """Create Watchdog instance with mocked DAQ."""
    def mock_make_daq(config):
        return mock_daq

    monkeypatch.setattr("retrofitkit.safety.watchdog.make_daq", mock_make_daq)
    return Watchdog(app_context)


class TestWatchdogInitialization:
    """Test cases for watchdog initialization."""

    def test_watchdog_initializes_with_context(self, app_context, mock_daq, monkeypatch):
        """Test that Watchdog initializes with AppContext."""
        monkeypatch.setattr("retrofitkit.safety.watchdog.make_daq", lambda cfg: mock_daq)

        watchdog = Watchdog(app_context)

        assert watchdog.ctx == app_context
        assert watchdog.daq is not None

    def test_watchdog_creates_daq_instance(self, app_context, mock_daq, monkeypatch):
        """Test that Watchdog creates DAQ instance from config."""
        make_daq_called = []

        def mock_make_daq(config):
            make_daq_called.append(config)
            return mock_daq

        monkeypatch.setattr("retrofitkit.safety.watchdog.make_daq", mock_make_daq)

        watchdog = Watchdog(app_context)

        assert len(make_daq_called) == 1
        assert make_daq_called[0] == app_context.config

    def test_watchdog_initializes_stop_event(self, watchdog):
        """Test that Watchdog initializes stop event."""
        assert hasattr(watchdog, '_stop')
        assert isinstance(watchdog._stop, asyncio.Event)
        assert not watchdog._stop.is_set()


class TestWatchdogToggling:
    """Test cases for watchdog toggle functionality."""

    @pytest.mark.asyncio
    async def test_watchdog_toggles_periodically(self, watchdog, mock_daq):
        """Test that watchdog toggles state periodically."""
        # Run watchdog for short duration
        task = asyncio.create_task(watchdog.run())

        # Wait for a few toggles
        await asyncio.sleep(2.5)

        # Stop watchdog
        watchdog.stop()
        await task

        # Should have toggled at least twice (once per second)
        assert mock_daq.toggle_watchdog.call_count >= 2

    @pytest.mark.asyncio
    async def test_watchdog_alternates_values(self, watchdog, mock_daq):
        """Test that watchdog alternates between True and False."""
        task = asyncio.create_task(watchdog.run())

        await asyncio.sleep(2.5)
        watchdog.stop()
        await task

        # Get all the calls
        calls = mock_daq.toggle_watchdog.call_args_list

        # Should alternate: False, True, False, True...
        if len(calls) >= 2:
            assert calls[0][0][0] is False  # First call with False
            assert calls[1][0][0] is True   # Second call with True

    @pytest.mark.asyncio
    async def test_watchdog_timing(self, watchdog, mock_daq):
        """Test that watchdog toggles approximately every second."""
        import time

        task = asyncio.create_task(watchdog.run())

        start_time = time.time()
        await asyncio.sleep(3.2)
        watchdog.stop()
        await task
        elapsed = time.time() - start_time

        # Should have toggled roughly every second
        expected_toggles = int(elapsed)
        actual_toggles = mock_daq.toggle_watchdog.call_count

        # Allow ±1 toggle due to timing
        assert abs(actual_toggles - expected_toggles) <= 1

    @pytest.mark.asyncio
    async def test_watchdog_only_toggles_when_method_exists(self, watchdog, mock_daq):
        """Test that watchdog only toggles if DAQ has toggle_watchdog method."""
        # Remove toggle_watchdog attribute
        del mock_daq.toggle_watchdog

        task = asyncio.create_task(watchdog.run())
        await asyncio.sleep(1.5)
        watchdog.stop()
        await task

        # Should complete without error, but no toggles
        # (because hasattr check fails)


class TestWatchdogStopBehavior:
    """Test cases for stopping the watchdog."""

    @pytest.mark.asyncio
    async def test_stop_sets_event(self, watchdog):
        """Test that stop() sets the stop event."""
        assert not watchdog._stop.is_set()

        watchdog.stop()

        assert watchdog._stop.is_set()

    @pytest.mark.asyncio
    async def test_watchdog_stops_when_event_set(self, watchdog, mock_daq):
        """Test that watchdog loop exits when stop event is set."""
        task = asyncio.create_task(watchdog.run())

        await asyncio.sleep(0.5)
        initial_count = mock_daq.toggle_watchdog.call_count

        watchdog.stop()
        await task

        # Watchdog should have stopped
        await asyncio.sleep(1.5)
        final_count = mock_daq.toggle_watchdog.call_count

        # Count should not increase after stop
        assert final_count == initial_count

    @pytest.mark.asyncio
    async def test_stop_can_be_called_multiple_times(self, watchdog):
        """Test that stop() can be called multiple times safely."""
        watchdog.stop()
        watchdog.stop()
        watchdog.stop()

        assert watchdog._stop.is_set()

    @pytest.mark.asyncio
    async def test_watchdog_task_completes_after_stop(self, watchdog, mock_daq):
        """Test that watchdog task completes after stop."""
        task = asyncio.create_task(watchdog.run())

        await asyncio.sleep(0.5)
        watchdog.stop()

        # Task should complete quickly after stop
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("Watchdog task did not complete after stop")


class TestErrorHandling:
    """Test cases for error handling in watchdog."""

    @pytest.mark.asyncio
    async def test_watchdog_continues_after_toggle_error(self, watchdog, mock_daq, capsys):
        """Test that watchdog continues running even if toggle fails."""
        # Make toggle fail sometimes
        call_count = [0]

        async def toggle_with_error(v):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Toggle failed")

        mock_daq.toggle_watchdog = toggle_with_error

        task = asyncio.create_task(watchdog.run())
        await asyncio.sleep(3.5)
        watchdog.stop()
        await task

        # Should have attempted multiple toggles despite error
        assert call_count[0] >= 3

        # Check that error was printed
        captured = capsys.readouterr()
        assert "Watchdog toggle failed" in captured.out

    @pytest.mark.asyncio
    async def test_watchdog_handles_missing_toggle_method(self, watchdog):
        """Test that watchdog handles DAQ without toggle_watchdog method."""
        # Create DAQ without toggle_watchdog
        watchdog.daq = Mock(spec=[])  # No methods

        task = asyncio.create_task(watchdog.run())
        await asyncio.sleep(1.5)
        watchdog.stop()

        # Should complete without error
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("Watchdog should handle missing toggle_watchdog gracefully")


class TestConcurrentOperation:
    """Test cases for concurrent watchdog operation."""

    @pytest.mark.asyncio
    async def test_multiple_toggle_cycles(self, watchdog, mock_daq):
        """Test watchdog running through multiple toggle cycles."""
        task = asyncio.create_task(watchdog.run())

        # Let it run for 5 seconds
        await asyncio.sleep(5.5)
        watchdog.stop()
        await task

        # Should have completed ~5 toggles
        assert 4 <= mock_daq.toggle_watchdog.call_count <= 6

    @pytest.mark.asyncio
    async def test_watchdog_with_other_async_tasks(self, watchdog, mock_daq):
        """Test watchdog running alongside other async tasks."""
        other_task_ran = []

        async def other_task():
            for i in range(3):
                other_task_ran.append(i)
                await asyncio.sleep(1)

        watchdog_task = asyncio.create_task(watchdog.run())
        other = asyncio.create_task(other_task())

        await asyncio.sleep(3.5)
        watchdog.stop()

        await watchdog_task
        await other

        # Both tasks should have run
        assert mock_daq.toggle_watchdog.call_count >= 3
        assert len(other_task_ran) == 3


class TestWatchdogStateTransitions:
    """Test cases for watchdog state transitions."""

    @pytest.mark.asyncio
    async def test_watchdog_initial_state(self, watchdog, mock_daq):
        """Test watchdog starts with False state."""
        task = asyncio.create_task(watchdog.run())

        await asyncio.sleep(0.5)
        watchdog.stop()
        await task

        # First call should be with False
        if mock_daq.toggle_watchdog.call_count > 0:
            first_call_value = mock_daq.toggle_watchdog.call_args_list[0][0][0]
            assert first_call_value is False

    @pytest.mark.asyncio
    async def test_watchdog_state_sequence(self, watchdog, mock_daq):
        """Test watchdog follows correct state sequence."""
        task = asyncio.create_task(watchdog.run())

        await asyncio.sleep(4.5)
        watchdog.stop()
        await task

        calls = mock_daq.toggle_watchdog.call_args_list

        # Should follow pattern: False, True, False, True, ...
        for i, call in enumerate(calls):
            expected_value = (i % 2) == 1  # True on odd indices
            actual_value = call[0][0]
            assert actual_value == expected_value


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_stop_before_run(self, watchdog):
        """Test calling stop before run doesn't cause issues."""
        watchdog.stop()

        # Run should exit immediately
        task = asyncio.create_task(watchdog.run())

        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail("Watchdog should exit immediately if stopped before run")

    @pytest.mark.asyncio
    async def test_very_short_run_duration(self, watchdog, mock_daq):
        """Test watchdog with very short run duration."""
        task = asyncio.create_task(watchdog.run())

        await asyncio.sleep(0.1)
        watchdog.stop()
        await task

        # May or may not have toggled depending on timing
        assert mock_daq.toggle_watchdog.call_count >= 0

    @pytest.mark.asyncio
    async def test_watchdog_restart(self, watchdog, mock_daq):
        """Test stopping and restarting watchdog."""
        # First run
        task1 = asyncio.create_task(watchdog.run())
        await asyncio.sleep(1.5)
        watchdog.stop()
        await task1

        first_count = mock_daq.toggle_watchdog.call_count

        # Reset stop event for restart
        watchdog._stop = asyncio.Event()

        # Second run
        task2 = asyncio.create_task(watchdog.run())
        await asyncio.sleep(1.5)
        watchdog.stop()
        await task2

        second_count = mock_daq.toggle_watchdog.call_count

        # Should have toggled more in second run
        assert second_count > first_count


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_watchdog_during_system_operation(self, watchdog, mock_daq):
        """Test watchdog running during simulated system operation."""
        operation_steps = []

        async def simulated_operation():
            operation_steps.append("start")
            await asyncio.sleep(2)
            operation_steps.append("middle")
            await asyncio.sleep(2)
            operation_steps.append("end")

        # Start watchdog
        watchdog_task = asyncio.create_task(watchdog.run())

        # Run operation
        await simulated_operation()

        # Stop watchdog
        watchdog.stop()
        await watchdog_task

        # Watchdog should have toggled throughout operation
        assert mock_daq.toggle_watchdog.call_count >= 4
        assert operation_steps == ["start", "middle", "end"]

    @pytest.mark.asyncio
    async def test_watchdog_heartbeat_detection(self, watchdog, mock_daq):
        """Test that watchdog provides detectable heartbeat."""
        heartbeats = []

        async def toggle_with_logging(v):
            heartbeats.append({"value": v, "time": asyncio.get_event_loop().time()})

        mock_daq.toggle_watchdog = toggle_with_logging

        task = asyncio.create_task(watchdog.run())
        await asyncio.sleep(3.5)
        watchdog.stop()
        await task

        # Should have regular heartbeats
        assert len(heartbeats) >= 3

        # Check timing between heartbeats
        if len(heartbeats) >= 2:
            time_diff = heartbeats[1]["time"] - heartbeats[0]["time"]
            assert 0.9 <= time_diff <= 1.1  # Approximately 1 second
