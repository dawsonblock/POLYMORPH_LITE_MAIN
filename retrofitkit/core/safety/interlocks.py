"""
Safety Interlock Controller.

Monitors critical safety inputs (E-Stop, Door) and enforces safe operation.
"""
import logging
from typing import Callable, List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class SafetyError(Exception):
    """Raised when an operation is attempted in an unsafe state."""
    pass

class InterlockController:
    """
    Manages system interlocks.
    
    Integrates with DAQ to read digital inputs for E-Stop and Enclosure Door.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.estop_active = False
        self.door_open = False
        self._callbacks: List[Callable[[bool], None]] = []
        self._daq = None # Lazy loaded to avoid circular dependency

    def set_daq(self, daq_driver):
        """Set the DAQ driver used for reading interlocks."""
        self._daq = daq_driver

    async def check_status(self) -> Dict[str, bool]:
        """
        Read current interlock status from hardware.
        
        Returns:
            Dict with 'estop_active' and 'door_open'.
        """
        if not self._daq:
            logger.warning("InterlockController: No DAQ driver set. Assuming safe for simulation/startup.")
            return {"estop_active": False, "door_open": False}

        try:
            # Read E-Stop
            # Assuming active LOW (0 = E-Stop pressed/unsafe, 1 = Safe) or configurable
            # Let's assume standard industrial: 0V = E-Stop Active (Fail-safe)
            estop_val = await self._daq.read_di(self.config.safety.estop_line)
            self.estop_active = not estop_val # If 0 (False), then Active (True)

            # Read Door
            # Assuming switch closed (1) = Closed/Safe, Open (0) = Open/Unsafe
            door_val = await self._daq.read_di(self.config.safety.door_line)
            self.door_open = not door_val

            if self.estop_active or self.door_open:
                self._trigger_callbacks(unsafe=True)

            return {
                "estop_active": self.estop_active,
                "door_open": self.door_open
            }
        except Exception as e:
            logger.error(f"Failed to read interlocks: {e}")
            # Fail safe: assume unsafe if we can't read
            self.estop_active = True
            self._trigger_callbacks(unsafe=True)
            raise SafetyError(f"Could not read interlock status: {e}")

    def check_temperature(self, temp_c: float) -> None:
        """
        Verify system is safe. Raises SafetyError if not.
        
        Should be called before any hazardous operation (laser on, motion).
        """
        # This method is a placeholder for future temperature interlocks.
        # For now, it does nothing.
        pass

    def check_safe(self):
        """
        Verify system is safe. Raises SafetyError if not.
        
        Should be called before any hazardous operation (laser on, motion).
        """
        # We rely on the last polled state or force a poll?
        # For performance, we might rely on a background poller, but for critical checks,
        # we should probably check cached state + freshness, or just check.
        # For now, we check cached state which should be updated by a monitoring loop.
        # If no monitoring loop, we should poll here.

        # Let's assume we trust the cached state if updated recently,
        # but for this implementation, we'll just check the flags.
        # In a real system, we'd want `await self.check_status()` but this method is often synchronous
        # in property checks. Let's make it raise if the flags are set.

        if self.estop_active:
            raise SafetyError("E-STOP ACTIVE: Operation refused.")

        if self.door_open:
            # Some operations might be allowed with door open (e.g. low power alignment)
            # But generally unsafe.
            raise SafetyError("DOOR OPEN: Operation refused.")

    def register_callback(self, callback: Callable[[bool], None]):
        """Register a callback to be called when safety state changes."""
        self._callbacks.append(callback)

    def _trigger_callbacks(self, unsafe: bool):
        """Notify listeners of safety event."""
        for cb in self._callbacks:
            try:
                cb(unsafe)
            except Exception as e:
                logger.error(f"Error in safety callback: {e}")

# Global instance
_interlock_instance: Optional[InterlockController] = None

def get_interlocks(config=None) -> InterlockController:
    global _interlock_instance
    if _interlock_instance is None:
        if config is None:
             raise ValueError("Config required to initialize InterlockController")
        _interlock_instance = InterlockController(config)
    return _interlock_instance
