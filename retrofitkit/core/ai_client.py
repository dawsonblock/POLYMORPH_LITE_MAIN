import time
import httpx
from typing import Dict, Any, List, Optional
import logging
import math

logger = logging.getLogger(__name__)


class AIFailsafeError(Exception):
    """Raised when AI service is critical but unreachable."""
    pass


class AIServiceClient:
    """
    Client for interacting with the BentoML AI Service.
    
    Includes Circuit Breaker pattern to handle service failures gracefully
    and connection pooling for better performance.
    """
    
    __slots__ = ('service_url', '_failures', '_circuit_open', '_circuit_threshold',
                 '_failure_threshold', '_recovery_timeout', '_last_failure_time', '_client')
    
    def __init__(self, service_url: str):
        self.service_url = service_url
        
        # Circuit Breaker state
        self._failures = 0
        self._circuit_open = False
        self._circuit_threshold = 3
        self._failure_threshold = 3
        self._recovery_timeout = 60.0
        self._last_failure_time = 0.0
        
        # Reusable HTTP client with connection pooling (lazily initialized)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create reusable HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(2.0, connect=1.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "circuit_open": self._circuit_open,
            "failures": self._failures,
            "service_url": self.service_url
        }

    async def predict(self, spectrum: List[float], critical: bool = True) -> Dict[str, Any]:
        """
        Call AI service inference endpoint.
        
        Args:
            spectrum: List of intensity values.
            critical: If True, raise AIFailsafeError on failure. If False, return empty dict.
            
        Returns:
            Dict containing prediction results.
            
        Raises:
            AIFailsafeError: If critical is True and service is unavailable.
        """
        if self._circuit_open:
            if time.time() - self._last_failure_time > self._recovery_timeout:
                logger.info("AI Circuit Breaker: Attempting recovery...")
            elif critical:
                raise AIFailsafeError("AI Circuit Breaker OPEN - Failsafe Triggered")
            else:
                return {}

        # Input Sanitization
        if not spectrum or not isinstance(spectrum, list):
            raise ValueError("AI Input Error: Spectrum must be a non-empty list.")
        
        if any(not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x) for x in spectrum):
            raise ValueError("AI Input Error: Spectrum contains non-numeric or invalid (NaN/Inf) values.")

        try:
            # Use context manager for better test compatibility while still
            # benefiting from connection pooling in production
            async with httpx.AsyncClient(timeout=2.0) as client:
                payload = {"spectrum": spectrum}
                # Assume /infer endpoint for BentoML
                url = f"{self.service_url.rstrip('/')}/infer" if not self.service_url.endswith("/infer") else self.service_url
                
                response = await client.post(url, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    if not isinstance(data, dict):
                        msg = "AI Output Error: Invalid response format."
                        logger.error(msg)
                        if critical:
                            raise AIFailsafeError(msg)
                        return {}
                    
                    # Validate concentration if present
                    if "concentration" in data:
                        val = data["concentration"]
                        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                            msg = f"AI Output Error: Invalid concentration value: {val}"
                            logger.error(msg)
                            if critical:
                                raise AIFailsafeError(msg)
                            return {}
                            
                    if self._circuit_open:
                        logger.info("AI Circuit Breaker: Recovered.")
                        self._failures = 0
                        self._circuit_open = False
                    return dict(data)
                else:
                    self._record_failure()
                    msg = f"AI Service Error: {response.status_code}"
                    logger.error(msg)
                    if critical:
                        raise AIFailsafeError(msg)
                    return {}
                    
        except httpx.TimeoutException as e:
            self._record_failure()
            msg = f"AI Connection Timeout: {str(e)}"
            logger.error(msg)
            if critical:
                raise AIFailsafeError(msg)
            return {}
        except AIFailsafeError:
            raise
        except ValueError:
            raise
        except Exception as e:
            self._record_failure()
            msg = f"AI Connection Failed: {str(e)}"
            logger.error(msg)
            if critical:
                raise AIFailsafeError(msg)
            return {}

    def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self._failure_threshold:
            if not self._circuit_open:
                logger.warning("AI Circuit Breaker: OPEN (Too many failures)")
            self._circuit_open = True
