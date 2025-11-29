"""
Device Discovery Module for Tier-1 Hardware Stack.

Auto-discovers and registers NI DAQ devices and Ocean Optics spectrometers.
Maintains device registry with real-time status tracking.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Supported device types."""
    NI_DAQ = "ni_daq"
    OCEAN_OPTICS = "ocean_optics"
    UNKNOWN = "unknown"


class DeviceStatus(str, Enum):
    """Device operational status."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class DiscoveredDevice:
    """Represents a discovered hardware device."""
    device_id: str
    device_type: DeviceType
    name: str
    status: DeviceStatus
    capabilities: Dict[str, Any]
    metadata: Dict[str, Any]
    discovered_at: datetime


class DeviceRegistry:
    """
    Central registry for discovered devices.
    
    Maintains device inventory and provides lookup capabilities.
    """
    
    def __init__(self):
        self._devices: Dict[str, DiscoveredDevice] = {}
        
    def register(self, device: DiscoveredDevice) -> None:
        """Register a discovered device."""
        self._devices[device.device_id] = device
        logger.info(f"Registered device: {device.device_id} ({device.device_type})")
        
    def get(self, device_id: str) -> Optional[DiscoveredDevice]:
        """Retrieve device by ID."""
        return self._devices.get(device_id)
        
    def list_by_type(self, device_type: DeviceType) -> List[DiscoveredDevice]:
        """List all devices of a specific type."""
        return [d for d in self._devices.values() if d.device_type == device_type]
        
    def list_available(self, device_type: Optional[DeviceType] = None) -> List[DiscoveredDevice]:
        """List available devices, optionally filtered by type."""
        devices = self._devices.values()
        if device_type:
            devices = [d for d in devices if d.device_type == device_type]
        return [d for d in devices if d.status == DeviceStatus.AVAILABLE]
        
    def update_status(self, device_id: str, status: DeviceStatus) -> None:
        """Update device status."""
        if device_id in self._devices:
            self._devices[device_id].status = status
            logger.info(f"Updated device {device_id} status to {status}")
            
    def all_devices(self) -> List[DiscoveredDevice]:
        """Return all registered devices."""
        return list(self._devices.values())


class NIDAQDiscovery:
    """Discovery for NI DAQ devices."""
    
    def discover(self) -> List[DiscoveredDevice]:
        """
        Discover all available NI DAQ devices.
        
        Returns:
            List of discovered NI DAQ devices
        """
        devices = []
        
        try:
            # Try to import NI DAQmx
            import nidaqmx
            system = nidaqmx.system.System.local()
            
            for device in system.devices:
                device_obj = DiscoveredDevice(
                    device_id=device.name,
                    device_type=DeviceType.NI_DAQ,
                    name=f"NI {device.product_type}",
                    status=DeviceStatus.AVAILABLE,
                    capabilities={
                        "ai_channels": len(device.ai_physical_chans),
                        "ao_channels": len(device.ao_physical_chans),
                        "di_lines": len(device.di_lines),
                        "do_lines": len(device.do_lines),
                        "product_type": device.product_type,
                    },
                    metadata={
                        "serial_number": getattr(device, 'serial_num', 'unknown'),
                        "driver_version": system.driver_version.major_version,
                    },
                    discovered_at=datetime.now()
                )
                devices.append(device_obj)
                logger.info(f"Discovered NI DAQ: {device.name}")
                
        except ImportError:
            logger.warning("NI DAQmx not available - skipping NI DAQ discovery")
        except Exception as e:
            logger.error(f"Error during NI DAQ discovery: {e}")
            
        return devices


class OceanOpticsDiscovery:
    """Discovery for Ocean Optics spectrometers."""
    
    def discover(self) -> List[DiscoveredDevice]:
        """
        Discover all available Ocean Optics spectrometers.
        
        Returns:
            List of discovered Ocean Optics devices
        """
        devices = []
        
        try:
            # Try to import seabreeze
            from seabreeze.spectrometers import list_devices, Spectrometer
            
            device_list = list_devices()
            
            for idx, device_info in enumerate(device_list):
                # Create temporary spectrometer instance to get capabilities
                spec = Spectrometer(device_info)
                
                device_obj = DiscoveredDevice(
                    device_id=f"ocean_{idx}",
                    device_type=DeviceType.OCEAN_OPTICS,
                    name=spec.model,
                    status=DeviceStatus.AVAILABLE,
                    capabilities={
                        "wavelengths": spec.wavelengths().tolist() if hasattr(spec, 'wavelengths') else [],
                        "min_integration_time": spec.minimum_integration_time_micros,
                        "max_intensity": getattr(spec, 'max_intensity', None),
                        "pixels": len(spec.wavelengths()) if hasattr(spec, 'wavelengths') else 0,
                    },
                    metadata={
                        "serial_number": spec.serial_number,
                        "model": spec.model,
                    },
                    discovered_at=datetime.now()
                )
                
                spec.close()
                devices.append(device_obj)
                logger.info(f"Discovered Ocean Optics: {spec.model} ({spec.serial_number})")
                
        except ImportError:
            logger.warning("seabreeze not available - skipping Ocean Optics discovery")
        except Exception as e:
            logger.error(f"Error during Ocean Optics discovery: {e}")
            
        return devices


class DeviceDiscoveryService:
    """
    Main service for device discovery and registry management.
    
    Usage:
        service = DeviceDiscoveryService()
        service.discover_all()
        devices = service.get_available_devices()
    """
    
    def __init__(self):
        self.registry = DeviceRegistry()
        self.discoverers = {
            DeviceType.NI_DAQ: NIDAQDiscovery(),
            DeviceType.OCEAN_OPTICS: OceanOpticsDiscovery(),
        }
        
    def discover_all(self) -> Dict[DeviceType, List[DiscoveredDevice]]:
        """
        Run discovery for all supported device types.
        
        Returns:
            Dictionary mapping device type to list of discovered devices
        """
        results = {}
        
        for device_type, discoverer in self.discoverers.items():
            try:
                devices = discoverer.discover()
                for device in devices:
                    self.registry.register(device)
                results[device_type] = devices
                logger.info(f"Discovered {len(devices)} {device_type} device(s)")
            except Exception as e:
                logger.error(f"Discovery failed for {device_type}: {e}")
                results[device_type] = []
                
        return results
        
    def get_available_devices(
        self, 
        device_type: Optional[DeviceType] = None
    ) -> List[DiscoveredDevice]:
        """Get all available devices, optionally filtered by type."""
        return self.registry.list_available(device_type)
        
    def get_tier1_devices(self) -> Dict[str, Optional[DiscoveredDevice]]:
        """
        Get Tier-1 hardware stack (NI DAQ + Ocean Optics).
        
        Returns:
            Dictionary with 'daq' and 'raman' devices (None if not found)
        """
        daq_devices = self.registry.list_available(DeviceType.NI_DAQ)
        raman_devices = self.registry.list_available(DeviceType.OCEAN_OPTICS)
        
        return {
            "daq": daq_devices[0] if daq_devices else None,
            "raman": raman_devices[0] if raman_devices else None,
        }
        
    def allocate_device(self, device_id: str) -> bool:
        """
        Mark device as in-use.
        
        Returns:
            True if device was available and allocated, False otherwise
        """
        device = self.registry.get(device_id)
        if device and device.status == DeviceStatus.AVAILABLE:
            self.registry.update_status(device_id, DeviceStatus.IN_USE)
            return True
        return False
        
    def release_device(self, device_id: str) -> None:
        """Mark device as available."""
        self.registry.update_status(device_id, DeviceStatus.AVAILABLE)


# Global discovery service instance
_discovery_service: Optional[DeviceDiscoveryService] = None


def get_discovery_service() -> DeviceDiscoveryService:
    """Get or create global discovery service instance."""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = DeviceDiscoveryService()
    return _discovery_service
