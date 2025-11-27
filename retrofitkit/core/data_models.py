"""
Unified data models for laboratory devices.

All devices return structured data using these models instead of raw arrays or dicts.
This enables:
- Type safety
- Consistent metadata handling
- Easy serialization
- Workflow compatibility
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import numpy as np
from datetime import datetime, timezone


@dataclass
class Spectrum:
    """
    Spectral data from any spectrometer (Raman, UV-Vis, Fluorescence, etc.).
    
    Attributes:
        wavelengths: Wavelength or wavenumber array
        intensities: Intensity values at each wavelength
        meta: Arbitrary metadata (integration time, temperature, etc.)
    """
    wavelengths: np.ndarray
    intensities: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate and set timestamp."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        
        # Ensure numpy arrays
        if not isinstance(self.wavelengths, np.ndarray):
            self.wavelengths = np.array(self.wavelengths)
        if not isinstance(self.intensities, np.ndarray):
            self.intensities = np.array(self.intensities)
        
        # Validate shapes match
        if self.wavelengths.shape != self.intensities.shape:
            raise ValueError(
                f"Wavelength and intensity shapes must match: "
                f"{self.wavelengths.shape} vs {self.intensities.shape}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "wavelengths": self.wavelengths.tolist(),
            "intensities": self.intensities.tolist(),
            "meta": self.meta,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Spectrum":
        """Create from dict (e.g., loaded from JSON)."""
        return cls(
            wavelengths=np.array(data["wavelengths"]),
            intensities=np.array(data["intensities"]),
            meta=data.get("meta", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
        )


@dataclass
class DAQTrace:
    """
    Time-series data from DAQ device.
    
    Attributes:
        time: Time array (seconds)
        values: Measured values at each time point
        channel: DAQ channel number
        meta: Arbitrary metadata (sample rate, voltage range, etc.)
    """
    time: np.ndarray
    values: np.ndarray
    channel: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate and set timestamp."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        
        # Ensure numpy arrays
        if not isinstance(self.time, np.ndarray):
            self.time = np.array(self.time)
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values)
        
        # Validate shapes match
        if self.time.shape != self.values.shape:
            raise ValueError(
                f"Time and value shapes must match: "
                f"{self.time.shape} vs {self.values.shape}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "time": self.time.tolist(),
            "values": self.values.tolist(),
            "channel": self.channel,
            "meta": self.meta,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAQTrace":
        """Create from dict (e.g., loaded from JSON)."""
        return cls(
            time=np.array(data["time"]),
            values=np.array(data["values"]),
            channel=data.get("channel", 0),
            meta=data.get("meta", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
        )


@dataclass
class ImageFrame:
    """
    Image data from cameras or imaging spectrometers.
    
    Attributes:
        data: 2D or 3D numpy array (height, width) or (height, width, channels)
        meta: Arbitrary metadata (exposure, binning, etc.)
    """
    data: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate and set timestamp."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        
        # Ensure numpy array
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
        
        # Validate dimensionality
        if self.data.ndim not in (2, 3):
            raise ValueError(
                f"Image data must be 2D or 3D, got {self.data.ndim}D"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "data": self.data.tolist(),
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype),
            "meta": self.meta,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageFrame":
        """Create from dict (e.g., loaded from JSON)."""
        array_data = np.array(data["data"])
        if "dtype" in data:
            array_data = array_data.astype(data["dtype"])
        
        return cls(
            data=array_data,
            meta=data.get("meta", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
        )


@dataclass
class MotionPosition:
    """
    Position data from motion stages.
    
    Attributes:
        position: Current position (units device-specific)
        velocity: Current velocity (if available)
        meta: Arbitrary metadata (units, axis name, etc.)
    """
    position: float
    velocity: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "position": self.position,
            "velocity": self.velocity,
            "meta": self.meta,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
