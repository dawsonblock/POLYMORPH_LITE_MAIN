import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

class HardwareTimeoutError(Exception):
    """Raised when a hardware operation exceeds its timeout."""
    pass

class ProductionHardwareDriver:
    """
    Base class for all production hardware drivers.
    Enforces thread safety and timeouts for blocking SDK calls.
    """
    def __init__(self, max_workers: int = 1):
        # Initialize logger if not present (BaseHardwareDriver usually handles this if inherited)
        # But ProductionHardwareDriver doesn't seem to inherit from BaseHardwareDriver in the file I viewed?
        # Let's check imports. It imports DeviceKind, DeviceCapabilities from base.
        # But definition is `class ProductionHardwareDriver:`. It does NOT inherit BaseHardwareDriver!
        # This is a problem if AndorRaman expects it to be a driver.
        # AndorRaman inherits ProductionHardwareDriver.
        # If ProductionHardwareDriver is the base, it should provide logger.
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()
        
        # Dry Run Configuration
        # If config object is passed in subclass, it should set this.
        # Here we default to False, but allow enabling.
        self.dry_run = False

    def set_dry_run(self, enabled: bool):
        """Enable or disable dry-run mode."""
        self.dry_run = enabled
        if enabled:
            self.logger.warning(f"DRY RUN MODE ENABLED for {self.__class__.__name__} - No real hardware commands will be sent.")

    def log_dry_run(self, command: str, params: dict = None):
        """Log a command in dry-run mode."""
        if self.dry_run:
            self.logger.info(f"[DRY-RUN] {command} | Params: {params or {}}")
            return True
        return False

    async def _run_blocking(self, func: Callable, *args, timeout: float = 30.0) -> Any:
        """
        Executes a blocking function in a separate thread with a timeout.
        
        Args:
            func: The blocking function to execute.
            *args: Arguments to pass to the function.
            timeout: Maximum time to wait for the function to complete (in seconds).
            
        Returns:
            The result of the function call.
            
        Raises:
            HardwareTimeoutError: If the function execution exceeds the timeout.
            Exception: Any exception raised by the blocking function.
        """
        async with self._lock:
            try:
                loop = asyncio.get_running_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(self._executor, func, *args),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                raise HardwareTimeoutError(f"{func.__name__} exceeded {timeout}s timeout")
            except Exception as e:
                # Log the error here if a logger is available
                raise e

    async def connect(self):
        """Connect to the hardware."""
        self.connected = True
        self.logger.info(f"Connected to {self.__class__.__name__}")

    async def disconnect(self):
        """Disconnect from the hardware."""
        self.connected = False
        self.logger.info(f"Disconnected from {self.__class__.__name__}")

    async def shutdown(self):
        """Clean up resources."""
        await self.disconnect()
        self._executor.shutdown(wait=True)
