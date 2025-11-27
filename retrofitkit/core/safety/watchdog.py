"""
System Watchdog.

Monitors software health and toggles a hardware pin to prevent hardware timeouts.
"""
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

class WatchdogError(Exception):
    pass

class SystemWatchdog:
    """
    Software watchdog that pets a hardware watchdog.
    """
    def __init__(self, config, daq_driver):
        self.config = config
        self.daq = daq_driver
        self.running = False
        self._task = None
        self.last_pet = time.time()
        self.timeout = config.safety.watchdog_seconds

    async def start(self):
        """Start the watchdog loop."""
        if self.running:
            return

        if not self.config.safety.watchdog_enabled:
            logger.info("Watchdog disabled in config.")
            return

        self.running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("System Watchdog started.")

    async def stop(self):
        """Stop the watchdog loop."""
        self.running = False
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("System Watchdog stopped.")

    async def pet(self):
        """Update the last pet time."""
        self.last_pet = time.time()

    async def _loop(self):
        """Main watchdog loop."""
        toggle = False
        watchdog_line = self.config.daq.ni_do_watchdog_line

        while self.running:
            try:
                # Check if we've been petted recently
                if time.time() - self.last_pet > self.timeout:
                    logger.critical("WATCHDOG TIMEOUT: Main thread unresponsive!")
                    # In a real system, we might trigger a shutdown or let the hardware watchdog trip
                    # We stop toggling the pin, which lets hardware safety take over
                    break

                # Toggle hardware pin
                if self.daq:
                    await self.daq.write_do(watchdog_line, toggle)
                    toggle = not toggle

                # Sleep half the timeout or a fixed fast interval
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Watchdog loop error: {e}")
                await asyncio.sleep(1.0)
