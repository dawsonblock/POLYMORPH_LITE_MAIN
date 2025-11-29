import asyncio
import logging
import functools
from typing import Callable, Any, TypeVar, Optional
from retrofitkit.core.error_codes import ErrorCode

logger = logging.getLogger(__name__)

T = TypeVar("T")

def hardware_call(
    timeout: float = 5.0,
    retries: int = 0,
    retry_delay: float = 1.0,
    error_code: ErrorCode = ErrorCode.INTERNAL_SERVER_ERROR
):
    """
    Decorator to wrap hardware calls with safety, timeout, and error handling.
    
    Args:
        timeout: Max execution time in seconds.
        retries: Number of retries on failure.
        retry_delay: Delay between retries.
        error_code: Error code to map exceptions to.
    """
    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    # Execute with timeout
                    # Note: This assumes the wrapped function is async. 
                    # If sync, we'd need run_in_executor.
                    if asyncio.iscoroutinefunction(func):
                        return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                    else:
                        # For sync functions, we can't easily enforce timeout without threads/signals
                        # So we just run it. Ideally, drivers should be async or run in threadpool.
                        # For now, we assume async for timeout enforcement or just run sync.
                        return func(*args, **kwargs)
                        
                except asyncio.TimeoutError:
                    logger.error(f"Hardware Timeout in {func.__name__} (Attempt {attempt+1}/{retries+1})", 
                                 extra={"error_code": ErrorCode.HARDWARE_TIMEOUT})
                    last_exception = TimeoutError(f"Hardware call timed out after {timeout}s")
                    
                except Exception as e:
                    logger.error(f"Hardware Error in {func.__name__}: {e} (Attempt {attempt+1}/{retries+1})", 
                                 extra={"error_code": error_code})
                    last_exception = e
                
                if attempt < retries:
                    await asyncio.sleep(retry_delay)
            
            # If we get here, all retries failed
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator
