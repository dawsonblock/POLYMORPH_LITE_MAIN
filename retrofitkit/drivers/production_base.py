import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

class HardwareTimeoutError(Exception):
    """Raised when a hardware operation exceeds its timeout."""
    pass

class ProductionHardwareDriver:
    """
    Base class for all production hardware drivers.
    Enforces thread safety and timeouts for blocking SDK calls.
    """
    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()

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

    async def shutdown(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
