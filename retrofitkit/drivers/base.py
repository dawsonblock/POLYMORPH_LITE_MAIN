"""
Base abstractions for device drivers in POLYMORPH-4 Lite.

This module defines the core interfaces and capability model for all hardware devices.
Devices register themselves with the DeviceRegistry for unified discovery and control.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from enum import Enum


class DeviceKind(str, Enum):
    """Device categories for capability-based discovery."""
    SPECTROMETER = "spectrometer"
    DAQ = "daq"
    MOTION = "motion"
    LASER = "laser"
    CAMERA = "camera"
    TEMPERATURE = "temperature"
    ELECTROCHEMISTRY = "electrochemistry"


@dataclass
class DeviceCapabilities:
    """
    Describes what a device can do.
    
    Used for capability-based device discovery and workflow validation.
    """
    kind: DeviceKind
    vendor: str
    model: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)

    def supports_action(self, action: str) -> bool:
        """Check if device supports a specific action."""
        return action in self.actions


@runtime_checkable
class DeviceBase(Protocol):
    """
    Base protocol for all hardware devices.
    
    All drivers must implement this interface to participate in the
    DeviceRegistry and workflow system.
    """
    id: str
    capabilities: DeviceCapabilities

    async def connect(self) -> None:
        """Establish connection to hardware device."""
        ...

    async def disconnect(self) -> None:
        """Close connection to hardware device."""
        ...

    async def health(self) -> Dict[str, Any]:
        """
        Get device health status.
        
        Returns:
            Dict with at least {"status": "ok"|"warning"|"error"}
        """
        ...


@runtime_checkable
class SpectrometerDevice(DeviceBase, Protocol):
    """
    Protocol for spectrometer devices (Raman, UV-Vis, etc.).
    
    Returns unified Spectrum data model.
    """
    async def acquire_spectrum(self, **kwargs) -> "Spectrum":  # type: ignore[name-defined]
        """Acquire a spectrum with device-specific parameters."""
        ...


@runtime_checkable
class DAQDevice(DeviceBase, Protocol):
    """
    Protocol for Data Acquisition devices.
    
    Provides analog/digital I/O capabilities.
    """
    async def read_ai(self, channel: int = 0) -> float:
        """Read analog input voltage from channel."""
        ...

    async def write_ao(self, channel: int, value: float) -> None:
        """Write analog output voltage to channel."""
        ...

    async def read_di(self, line: int) -> bool:
        """Read digital input state."""
        ...

    async def write_do(self, line: int, state: bool) -> None:
        """Write digital output state."""
        ...


@runtime_checkable
class MotionDevice(DeviceBase, Protocol):
    """
    Protocol for motion control devices (stages, actuators).
    
    Provides position control and homing.
    """
    async def move_to(self, position: float, **kwargs) -> None:
        """Move to absolute position."""
        ...

    async def move_relative(self, distance: float, **kwargs) -> None:
        """Move relative to current position."""
        ...

    async def home(self) -> None:
        """Home the stage to reference position."""
        ...

    async def get_position(self) -> float:
        """Get current position."""
        ...

    async def stop(self) -> None:
        """Emergency stop."""
        ...


@runtime_checkable
class LaserDevice(DeviceBase, Protocol):
    """
    Protocol for laser control devices.
    
    Provides power control and safety interlocks.
    """
    async def set_power(self, power_mw: float) -> None:
        """Set laser power in milliwatts."""
        ...

    async def get_power(self) -> float:
        """Get current laser power."""
        ...

    async def enable(self) -> None:
        """Enable laser output (open shutter)."""
        ...

    async def disable(self) -> None:
        """Disable laser output (close shutter)."""
        ...

    async def is_enabled(self) -> bool:
        """Check if laser output is enabled."""
        ...

# --- Safety Integration ---

from functools import wraps
from retrofitkit.core.safety.interlocks import get_interlocks

def require_safety(func):
    """Decorator to enforce safety checks before hardware actions."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if hasattr(self, "interlocks") and self.interlocks:
            self.interlocks.check_safe()
        return await func(self, *args, **kwargs)
    return wrapper

class SafetyAwareMixin:
    """Mixin for devices that need to respect system interlocks."""
    def __init__(self, config):
        self.config = config
        try:
            self.interlocks = get_interlocks(config)
        except Exception:
            self.interlocks = None

    async def ensure_safe(self):
        """Explicitly check safety."""
        if self.interlocks:
            self.interlocks.check_safe()

