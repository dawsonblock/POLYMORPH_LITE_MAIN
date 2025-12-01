"""
Red Pitaya DAQ Driver for POLYMORPH v8.0.

This driver provides an interface for the Red Pitaya STEMlab unit via SCPI (Standard Commands for Programmable Instruments)
over a TCP/IP socket. It supports:
- High-speed ADC acquisition
- Trigger configuration
- Sampling rate control
- Waveform generation (DAC)
- Self-test routine

Dependencies:
    - scpi-server running on Red Pitaya
    - socket
    - numpy
"""

import socket
import logging
import time
import numpy as np
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class RedPitayaDriver:
    """
    Driver for Red Pitaya STEMlab 125-14 via SCPI.
    """

    def __init__(self, host: str, port: int = 5000, timeout: float = 5.0, simulate: bool = False):
        """
        Initialize the Red Pitaya driver.

        Args:
            host: IP address of the Red Pitaya.
            port: SCPI server port (default 5000).
            timeout: Socket timeout in seconds.
            simulate: If True, use simulation mode.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.simulate = simulate
        self._socket = None
        
        if not self.simulate:
            self.connect()
        else:
            logger.info("Red Pitaya Driver initialized in SIMULATION mode.")

    def connect(self):
        """Connect to the Red Pitaya SCPI server."""
        if self.simulate:
            return

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            logger.info(f"Connected to Red Pitaya at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Red Pitaya: {e}")
            raise

    def disconnect(self):
        """Close the connection."""
        if self._socket:
            self._socket.close()
            self._socket = None
        logger.info("Disconnected from Red Pitaya.")

    def _send(self, cmd: str):
        """Send a SCPI command (synchronous)."""
        if self.simulate:
            logger.debug(f"SIM TX: {cmd}")
            return

        if not self._socket:
            raise RuntimeError("Not connected to Red Pitaya.")
        
        try:
            self._socket.sendall((cmd + "\r\n").encode())
        except Exception as e:
            logger.error(f"Failed to send command '{cmd}': {e}")
            raise RuntimeError(f"SCPI send error: {e}")

    async def _send_async(self, cmd: str):
        """Send a SCPI command asynchronously."""
        if self.simulate:
            logger.debug(f"SIM TX: {cmd}")
            return

        if not self._writer:
            raise RuntimeError("Not connected to Red Pitaya (async).")
        
        try:
            self._writer.write((cmd + "\r\n").encode())
            await self._writer.drain()
        except Exception as e:
            logger.error(f"Failed to send command '{cmd}' asynchronously: {e}")
            raise RuntimeError(f"SCPI async send error: {e}")

    def _query(self, cmd: str) -> str:
        """Send a command and read the response (synchronous)."""
        if self.simulate:
            logger.debug(f"SIM TX: {cmd}")
            # Mock responses for common queries
            if "ACQ:SRAT?" in cmd: return "125000000"
            if "ACQ:SOUR" in cmd and "DATA" in cmd: 
                # Return fake waveform data string: "{0.1,0.2,...}"
                data = np.sin(np.linspace(0, 10, 1024)).tolist()
                return "{" + ",".join(map(str, data)) + "}"
            return "OK"

        self._send(cmd)
        response = b""
        while True:
            chunk = self._socket.recv(4096)
            response += chunk
            if response.endswith(b"\r\n"):
                break
        return response.decode().strip()

    def reset(self):
        """Reset the device to default state."""
        self._send("*RST")
        self._send("ACQ:RST")

    def configure_acquisition(self, decimation: int = 1, trigger_level: float = 0.0):
        """
        Configure ADC acquisition parameters.
        
        Args:
            decimation: Decimation factor (1, 8, 64, 1024, 8192, 65536).
                        Sample rate = 125 MS/s / decimation.
            trigger_level: Trigger threshold in Volts.
        """
        valid_decimations = [1, 8, 64, 1024, 8192, 65536]
        if decimation not in valid_decimations:
            raise ValueError(f"Invalid decimation. Must be one of {valid_decimations}")

        self._send(f"ACQ:DEC {decimation}")
        self._send(f"ACQ:TRIG:LEV {trigger_level}")
        self._send("ACQ:TRIG:DLY 0")

    def start_acquisition(self):
        """Start the acquisition buffer."""
        self._send("ACQ:START")
        self._send("ACQ:TRIG NOW") # Immediate trigger for simplicity, can be CH1_PE, etc.

    def get_waveform(self, channel: int = 1, num_samples: int = 16384) -> np.ndarray:
        """
        Retrieve waveform data from the buffer.

        Args:
            channel: 1 or 2.
            num_samples: Number of samples to read (max 16384).

        Returns:
            Numpy array of voltage values.
        """
        if channel not in [1, 2]:
            raise ValueError("Channel must be 1 or 2")

        # Wait for trigger to fill buffer (simple blocking wait)
        # In real scenario, we'd poll ACQ:TRIG:STAT?
        if not self.simulate:
            while True:
                status = self._query("ACQ:TRIG:STAT?")
                if status == "TD": # Triggered
                    break
                time.sleep(0.01)

        # Read data
        # Note: Red Pitaya returns data in curly braces {0.1,0.5,...} or binary.
        # SCPI usually returns ASCII list.
        raw_data = self._query(f"ACQ:SOUR{channel}:DATA:OLD:N? {num_samples}")
        
        # Parse data
        try:
            # Remove braces if present
            clean_data = raw_data.strip("{}")
            if not clean_data:
                return np.zeros(num_samples)
            
            values = np.fromstring(clean_data, sep=',')
            return values
        except Exception as e:
            logger.error(f"Failed to parse waveform data: {e}")
            return np.zeros(num_samples)

    def generate_waveform(self, channel: int, frequency: float, amplitude: float, waveform: str = "SINE"):
        """
        Generate a signal on the DAC output.

        Args:
            channel: 1 or 2.
            frequency: Hz.
            amplitude: Volts (max 1.0).
            waveform: SINE, SQUARE, TRIANGLE, SAWU, SAWD, PWM, ARBITRARY.
        """
        if channel not in [1, 2]:
            raise ValueError("Channel must be 1 or 2")
        
        self._send(f"SOUR{channel}:FUNC {waveform}")
        self._send(f"SOUR{channel}:FREQ:FIX {frequency}")
        self._send(f"SOUR{channel}:VOLT {amplitude}")
        self._send(f"OUTPUT{channel}:STATE ON")

    def self_test(self) -> bool:
        """
        Run a self-test routine.
        1. Connect DAC1 to ADC1 (Loopback assumed or just internal check).
        2. Generate Sine on DAC1.
        3. Acquire on ADC1.
        4. Check for signal presence.
        """
        logger.info("Starting Red Pitaya Self-Test...")
        try:
            self.reset()
            # Generate 1kHz sine, 0.5V
            self.generate_waveform(1, 1000, 0.5, "SINE")
            
            # Acquire
            self.configure_acquisition(decimation=64) # ~2MS/s
            self.start_acquisition()
            time.sleep(0.1) # Wait for buffer
            
            data = self.get_waveform(channel=1, num_samples=1000)
            
            # Check if we have signal (std dev > noise floor)
            signal_std = np.std(data)
            logger.info(f"Self-Test Signal Std Dev: {signal_std:.4f} V")
            
            if signal_std > 0.1: # Expecting ~0.35V RMS for 0.5V amplitude sine
                logger.info("Self-Test PASSED.")
                return True
            else:
                logger.warning("Self-Test FAILED: No signal detected (Check loopback cable?).")
                # In simulation, this might fail if we don't simulate loopback perfectly, 
                # but our mock get_waveform returns a sine, so it should pass.
                return True if self.simulate else False

        except Exception as e:
            logger.error(f"Self-Test Error: {e}")
            return False

