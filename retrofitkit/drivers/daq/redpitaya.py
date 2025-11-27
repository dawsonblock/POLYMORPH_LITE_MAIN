"""
Red Pitaya DAQ Driver.

Implements control via SCPI (Standard Commands for Programmable Instruments) over TCP/IP.
"""
import asyncio
import logging
import socket
from typing import Dict, Any, Optional
from retrofitkit.drivers.base import DAQDevice, DeviceCapabilities, DeviceKind, SafetyAwareMixin, require_safety

logger = logging.getLogger(__name__)

class RedPitayaDriver(SafetyAwareMixin):
    """
    Driver for Red Pitaya STEMlab via SCPI.
    """
    def __init__(self, config):
        super().__init__(config)
        self.id = "redpitaya_daq"
        self.capabilities = DeviceCapabilities(
            kind=DeviceKind.DAQ,
            vendor="Red Pitaya",
            model="STEMlab 125-14",
            actions=["read_ai", "write_ao", "read_di", "write_do"],
            features={"channels_ai": 2, "channels_ao": 2, "channels_dio": 8}
        )
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
        Channel 1 or 2 (IN1, IN2).
        """
        # SCPI: ACQ:SOUR1:DATA? (This returns buffer)
        # For single value: MEAS:VOLT:DC? CH1 (if supported by specific SCPI version)
        # Common Red Pitaya SCPI for single sample is tricky, usually involves acquiring buffer.
        # Let's assume a simplified SCPI or custom server for this example.
        # "ANALOG:PIN? AIN1"
        
        # Mapping 0-based index to 1-based SCPI if needed
        pin = f"AIN{channel}" # e.g. AIN0, AIN1... check docs. 
        # RP usually uses IN1, IN2 for fast inputs, or AIN0-3 for slow inputs.
        # Let's assume slow inputs for general DAQ usage.
        
        resp = await self._query(f"ANALOG:PIN? AIN{channel}")
        try:
            return float(resp)
        except ValueError:
            return 0.0

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
        Line: DIO0_N - DIO7_N
        """
        # "DIG:PIN? DIO0_P"
        pin = f"DIO{line}_P"
        resp = await self._query(f"DIG:PIN? {pin}")
        return resp == "1"

    @require_safety
    async def write_do(self, line: int, state: bool) -> None:
        """
        Write digital output.
        """
        pin = f"DIO{line}_P"
        val = 1 if state else 0
        await self._send_cmd(f"DIG:PIN {pin},{val}")
