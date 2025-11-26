"""
Device Registry for unified device discovery and management.

The registry provides:
- Capability-based device discovery
- Factory pattern for device instantiation
- Runtime device inventory
"""
from typing import Dict, Type, Any, List, Optional
from retrofitkit.drivers.base import DeviceBase, DeviceCapabilities, DeviceKind


class DeviceRegistry:
    """
    Central registry for all hardware device drivers.
    
    Drivers register themselves at import time, enabling:
    - Dynamic device discovery
    - Capability-based queries
    - Consistent factory pattern
    
    Example:
        # In driver file
        registry.register("ocean_optics", OceanOpticsSpectrometer)
        
        # In application code
        devices = registry.list_drivers()
        device = registry.create("ocean_optics", device_index=0)
    """
    
    def __init__(self) -> None:
        self._drivers: Dict[str, Type[DeviceBase]] = {}
        self._instances: Dict[str, DeviceBase] = {}
    
    def register(self, name: str, driver_cls: Type[DeviceBase]) -> None:
        """
        Register a device driver class.
        
        Args:
            name: Unique driver name (e.g., "ocean_optics", "andor", "ni_daq")
            driver_cls: Driver class implementing DeviceBase protocol
            
        Raises:
            ValueError: If driver name already registered
        """
        if name in self._drivers:
            raise ValueError(f"Driver '{name}' already registered")
        
        # Validate driver has capabilities
        if not hasattr(driver_cls, "capabilities"):
            raise TypeError(
                f"Driver class {driver_cls.__name__} must have 'capabilities' class attribute"
            )
        
        self._drivers[name] = driver_cls
    
    def unregister(self, name: str) -> None:
        """
        Remove a driver from registry.
        
        Args:
            name: Driver name to remove
        """
        self._drivers.pop(name, None)
    
    def list_drivers(self) -> Dict[str, DeviceCapabilities]:
        """
        Get all registered drivers and their capabilities.
        
        Returns:
            Dict mapping driver name to DeviceCapabilities
        """
        return {
            name: driver_cls.capabilities  # type: ignore[attr-defined]
            for name, driver_cls in self._drivers.items()
        }
    
    def find_by_kind(self, kind: DeviceKind) -> List[str]:
        """
        Find all drivers of a specific kind.
        
        Args:
            kind: Device kind to search for
            
        Returns:
            List of driver names matching the kind
        """
        return [
            name
            for name, driver_cls in self._drivers.items()
            if driver_cls.capabilities.kind == kind  # type: ignore[attr-defined]
        ]
    
    def find_by_action(self, action: str) -> List[str]:
        """
        Find all drivers supporting a specific action.
        
        Args:
            action: Action name (e.g., "acquire_spectrum", "move_to")
            
        Returns:
            List of driver names supporting the action
        """
        return [
            name
            for name, driver_cls in self._drivers.items()
            if driver_cls.capabilities.supports_action(action)  # type: ignore[attr-defined]
        ]
    
    def create(self, name: str, instance_id: Optional[str] = None, **kwargs: Any) -> DeviceBase:
        """
        Create a device instance using registered driver.
        
        Args:
            name: Registered driver name
            instance_id: Optional unique instance ID for tracking
            **kwargs: Arguments passed to driver constructor
            
        Returns:
            Device instance
            
        Raises:
            KeyError: If driver not found
        """
        driver_cls = self._drivers.get(name)
        if driver_cls is None:
            available = ", ".join(self._drivers.keys())
            raise KeyError(
                f"No driver registered with name '{name}'. "
                f"Available drivers: {available}"
            )
        
        instance = driver_cls(**kwargs)  # type: ignore[call-arg]
        
        # Track instance if ID provided
        if instance_id is not None:
            self._instances[instance_id] = instance
        
        return instance
    
    def get_instance(self, instance_id: str) -> Optional[DeviceBase]:
        """
        Get a tracked device instance by ID.
        
        Args:
            instance_id: Instance ID from create()
            
        Returns:
            Device instance or None if not found
        """
        return self._instances.get(instance_id)
    
    def list_instances(self) -> Dict[str, DeviceBase]:
        """
        Get all tracked device instances.
        
        Returns:
            Dict mapping instance ID to device instance
        """
        return dict(self._instances)
    
    def remove_instance(self, instance_id: str) -> None:
        """
        Remove a tracked instance.
        
        Args:
            instance_id: Instance ID to remove
        """
        self._instances.pop(instance_id, None)
    
    def clear(self) -> None:
        """Clear all registered drivers and instances."""
        self._drivers.clear()
        self._instances.clear()


# Global registry instance
registry = DeviceRegistry()
