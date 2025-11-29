import time
import httpx
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class AIFailsafeError(Exception):
    """Raised when AI service is critical but unreachable."""
    pass

class AIServiceClient:
    """
    Client for interacting with the BentoML AI Service.
    Includes Circuit Breaker pattern to handle service failures gracefully.
    """
    def __init__(self, service_url: str):
        self.service_url = service_url
        
        # Circuit Breaker state
        self._failures = 0
        self._circuit_open = False
        self._circuit_threshold = 3
        self._failure_threshold = 3
        self._recovery_timeout = 60.0
        self._last_failure_time = 0.0

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

        try:
            async with httpx.AsyncClient() as client:
                payload = {"spectrum": spectrum}
                # Assume /infer endpoint for BentoML
                url = f"{self.service_url.rstrip('/')}/infer" if not self.service_url.endswith("/infer") else self.service_url
                
                response = await client.post(url, json=payload, timeout=2.0)

                if response.status_code == 200:
                    if self._circuit_open:
                        logger.info("AI Circuit Breaker: Recovered.")
                        self._failures = 0
                        self._circuit_open = False
                    return dict(response.json())
                else:
                    self._record_failure()
                    msg = f"AI Service Error: {response.status_code}"
                    logger.error(msg)
                    if critical: raise AIFailsafeError(msg)
                    return {}
                    
        except httpx.TimeoutException as e:
            self._record_failure()
            msg = f"AI Connection Timeout: {str(e)}"
            logger.error(msg)
            if critical: raise AIFailsafeError(msg)
            return {}
        except Exception as e:
            self._record_failure()
            msg = f"AI Connection Failed: {str(e)}"
            logger.error(msg)
            if critical: raise AIFailsafeError(msg)
            return {}

    def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self._failure_threshold:
            if not self._circuit_open:
                logger.warning("AI Circuit Breaker: OPEN (Too many failures)")
            self._circuit_open = True
