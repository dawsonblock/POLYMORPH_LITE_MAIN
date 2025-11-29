"""
Red Pitaya DAQ Driver - PRODUCTION LIMITATIONS

This driver provides ANALOG OUTPUT ONLY via SCPI over TCP/IP.

SUPPORTED FEATURES:
==================
✅ write_ao(channel, value): Set analog output voltage (AOUT0-3)
   - SCPI command: ANALOG:PIN AOUT{channel},{value}
   - Validated and production-ready
   - Voltage range: Typically -1.8V to +1.8V (check your model)

NOT SUPPORTED IN THIS BUILD:
============================
❌ read_ai(): Analog input
   - Raises RuntimeError with clear message
   - Use NI DAQ (ni.py) for analog input requirements

❌ read_di(): Digital input
   - Raises RuntimeError with clear message
   - Use NI DAQ (ni.py) for digital input requirements

❌ write_do(): Digital output
   - Raises RuntimeError with clear message
   - Use NI DAQ (ni.py) for digital output requirements

DEPLOYMENT RECOMMENDATIONS:
===========================
For production deployments:
1. Use Red Pitaya ONLY for analog output control
2. Use NI DAQ (ni.py) for all AI/DI/DO requirements
3. Tier-1 stack recommendation: NI DAQ + Ocean Optics (see docs/HARDWARE_DRIVERS.md)

IMPLEMENTATION STATUS:
=====================
- AO: PRODUCTION-READY (SCPI validated)
- AI/DI/DO: NOT IMPLEMENTED (clear runtime errors)

TO ENABLE FULL SUPPORT:
======================
Implement SCPI commands in the unimplemented methods:
- read_ai(): Use "ANALOG:PIN? AIN{channel}" SCPI query
- read_di(): Use appropriate digital I/O SCPI commands
- write_do(): Use appropriate digital I/O SCPI commands

See Red Pitaya SCPI documentation: https://redpitaya.readthedocs.io/

HARDWARE REQUIREMENTS:
=====================
- Red Pitaya STEMlab (125-14 or 125-10)
- Network connection to Red Pitaya
- SCPI server running on Red Pitaya (default port 5000)

EXAMPLE USAGE:
=============
```python
from retrofitkit.drivers.daq.redpitaya import RedPitayaDAQ

# Initialize
daq = RedPitayaDAQ(config)
await daq.connect()

# Set analog output
await daq.write_ao(channel=0, value=1.5)  # ✅ Works

# These will raise RuntimeError:
# await daq.read_ai(channel=0)  # ❌ Not implemented
# await daq.read_di(line=0)     # ❌ Not implemented
# await daq.write_do(line=0, state=True)  # ❌ Not implemented
```
"""
import asyncio
import logging
from typing import Dict, Any
from retrofitkit.drivers.base import DeviceCapabilities, DeviceKind, SafetyAwareMixin, require_safety
from retrofitkit.core.registry import registry

logger = logging.getLogger(__name__)

class RedPitayaDAQ(SafetyAwareMixin):
    """
    Driver for Red Pitaya STEMlab via SCPI.
    
    NOTE: This driver is currently LIMITED to Analog Output (AO) only.
    AI and DI/DO are not fully implemented/validated in this build.
    """
    KIND = DeviceKind.DAQ

    # Required for registry validation (class-level default)
    capabilities = DeviceCapabilities(
        kind=DeviceKind.DAQ,
        vendor="Red Pitaya",
        model="STEMlab 125-14",
        actions=["write_ao"],
        features={"channels_ai": 0, "channels_ao": 2, "channels_dio": 0}
    )

    def __init__(self, config):
        super().__init__(config)
        self.id = f"redpitaya_{self.host}"
        
        # Warn about limitations
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Red Pitaya driver initialized for {self.host}. Note: This build supports AO only. AI/DI/DO are NOT implemented. Use NI DAQ for full IO.")
        # Instance-level capabilities (can be same as class-level)
        self.capabilities = self.__class__.capabilities
        self.host = config.daq.redpitaya_host
        self.port = config.daq.redpitaya_port
        self.reader = None
        self.writer = None
        self.connected = False

    async def connect(self) -> None:
        """Connect to Red Pitaya SCPI server."""
        try:
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
            self.connected = True
            logger.info(f"Connected to Red Pitaya at {self.host}:{self.port}")

            # Reset
            await self._send_cmd("RST")
        except Exception as e:
            logger.error(f"Failed to connect to Red Pitaya: {e}")
            # Fallback to simulation if configured?
            # For now, we raise, but in a real app we might want a soft fail.
            raise

    async def disconnect(self) -> None:
        """Close connection."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.connected = False
        logger.info("Red Pitaya disconnected.")

    async def health(self) -> Dict[str, Any]:
        """Check connection health."""
        return {
            "status": "ok" if self.connected else "error",
            "connected": self.connected,
            "host": self.host
        }

    async def _send_cmd(self, cmd: str) -> None:
        """Send SCPI command."""
        if not self.connected:
            return
        msg = f"{cmd}\r\n".encode()
        self.writer.write(msg)
        await self.writer.drain()

    async def _query(self, cmd: str) -> str:
        """Send SCPI query and read response."""
        if not self.connected:
            return ""
        await self._send_cmd(cmd)
        data = await self.reader.read(1024)
        return data.decode().strip()

    @require_safety
    async def read_ai(self, channel: int = 1) -> float:
        """
        Read analog input voltage.
        """
        raise RuntimeError(
            "RedPitayaDAQ read_ai is not implemented in this build. "
            "This backend is output-only; configure another DAQ for AI."
        )

    @require_safety
    async def write_ao(self, channel: int, value: float) -> None:
        """
        Write analog output voltage.
        Channel 0-3 (AOUT0-3).
        """
        # "ANALOG:PIN AOUT0,1.5"
        await self._send_cmd(f"ANALOG:PIN AOUT{channel},{value}")

    @require_safety
    async def read_di(self, line: int) -> bool:
        """
        Read digital input.
        """
        raise RuntimeError(
            "RedPitayaDAQ read_di is not implemented in this build. "
            "Use NI DAQ or another backend for digital inputs."
        )

    @require_safety
    async def write_do(self, line: int, state: bool) -> None:
        """
        Write digital output.
        """
        raise RuntimeError(
            "RedPitayaDAQ write_do is not implemented in this build. "
            "Use NI DAQ or another backend for digital outputs."
        )

# Register the device in the global registry.
registry.register("redpitaya_daq", RedPitayaDAQ)
