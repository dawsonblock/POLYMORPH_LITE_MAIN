"""
Unified Hardware Router.

Responsible for:
1. Mapping abstract device types (e.g., "daq", "raman") to concrete driver classes.
2. Instantiating drivers based on configuration.
3. Managing the lifecycle of hardware connections.
"""
import logging
from typing import Dict, Type, Any, Optional

from retrofitkit.core.config import PolymorphConfig

logger = logging.getLogger(__name__)

class DriverRouter:
    """
    Registry and factory for hardware drivers.
    """
    def __init__(self):
        self._registry: Dict[str, Dict[str, Type]] = {
            "daq": {},
            "raman": {},
            "stage": {},
            "laser": {}
        }
        self._active_drivers: Dict[str, Any] = {}

    def register_driver(self, device_type: str, driver_name: str, driver_cls: Type):
        """
        Register a driver class.
        
        Args:
            device_type: "daq", "raman", etc.
            driver_name: "ni", "simulator", "ocean", etc.
            driver_cls: The class implementing the driver.
        """
        if device_type not in self._registry:
            self._registry[device_type] = {}

        self._registry[device_type][driver_name] = driver_cls
        logger.debug(f"Registered driver: {device_type}/{driver_name} -> {driver_cls.__name__}")

    def get_driver(self, device_type: str, config: PolymorphConfig) -> Any:
        """
        Get (or create) a driver instance for the given device type based on config.
        
        Args:
            device_type: "daq", "raman", etc.
            config: The full system configuration.
            
        Returns:
            An instance of the requested driver.
        """
        # Check if already instantiated
        if device_type in self._active_drivers:
            return self._active_drivers[device_type]

        # Determine driver name from config
        driver_name = self._resolve_driver_name(device_type, config)

        if not driver_name:
            raise ValueError(f"No driver configured for device type '{device_type}'")

        # Look up class
        driver_cls = self._registry.get(device_type, {}).get(driver_name)
        if not driver_cls:
            raise ValueError(f"Driver '{driver_name}' not registered for device type '{device_type}'")

        # Instantiate
        try:
            # We pass the relevant config section to the driver
            # This assumes drivers accept a config object or dict
            # For now, we'll pass the whole config and let the driver pick what it needs
            # OR we can pass specific sections. Let's pass the whole config for maximum flexibility
            # but drivers should ideally be loosely coupled.

            # Better approach: Pass the specific config section if possible, or the whole thing.
            # Let's pass the whole config for now as drivers might need cross-cutting concerns (safety, logging)
            driver_instance = driver_cls(config)
            self._active_drivers[device_type] = driver_instance
            logger.info(f"Instantiated driver {device_type}/{driver_name}")
            return driver_instance

        except Exception as e:
            logger.error(f"Failed to instantiate driver {device_type}/{driver_name}: {e}")
            raise RuntimeError(f"Failed to instantiate driver {device_type}/{driver_name}: {e}")

    def _resolve_driver_name(self, device_type: str, config: PolymorphConfig) -> Optional[str]:
        """Resolve the driver name from configuration."""
        if device_type == "daq":
            return config.daq.backend
        elif device_type == "raman":
            return config.raman.provider
        # Add more mappings as needed
        return None

    def initialize_all(self, config: PolymorphConfig):
        """Initialize all configured drivers."""
        # Pre-load standard drivers
        self._register_standard_drivers()

        # Instantiate core devices
        self.get_driver("daq", config)
        self.get_driver("raman", config)

    def _register_standard_drivers(self):
        """Register built-in drivers."""
        # Lazy import to avoid circular dependencies
        try:
            from retrofitkit.drivers.daq.simulator import DAQSimulator
            self.register_driver("daq", "simulator", DAQSimulator)
        except ImportError:
            logger.warning("Could not import DAQSimulator")

        try:
            from retrofitkit.drivers.daq.ni import NIDAQDriver
            self.register_driver("daq", "ni", NIDAQDriver)
        except ImportError:
            logger.warning("Could not import NIDAQDriver")

        try:
            from retrofitkit.drivers.daq.redpitaya import RedPitayaDriver
            self.register_driver("daq", "redpitaya", RedPitayaDriver)
        except ImportError:
            logger.warning("Could not import RedPitayaDriver")

        try:
            from retrofitkit.drivers.raman.simulator import RamanSimulator
            self.register_driver("raman", "simulator", RamanSimulator)
        except ImportError:
            logger.warning("Could not import RamanSimulator")

        try:
            from retrofitkit.drivers.raman.vendor_ocean_optics import OceanOpticsDriver
            self.register_driver("raman", "ocean", OceanOpticsDriver)
        except ImportError:
            logger.warning("Could not import OceanOpticsDriver")

        try:
            from retrofitkit.drivers.raman.vendor_horiba import HoribaDriver
            self.register_driver("raman", "horiba", HoribaDriver)
        except ImportError:
            logger.warning("Could not import HoribaDriver")

        try:
            from retrofitkit.drivers.raman.vendor_andor import AndorDriver
            self.register_driver("raman", "andor", AndorDriver)
        except ImportError:
            logger.warning("Could not import AndorDriver")

    def shutdown_all(self):
        """Shutdown all active drivers."""
        for device_type, driver in self._active_drivers.items():
            try:
                if hasattr(driver, "shutdown"):
                    driver.shutdown()
                elif hasattr(driver, "close"):
                    driver.close()
                logger.info(f"Shutdown driver for {device_type}")
            except Exception as e:
                logger.error(f"Error shutting down {device_type}: {e}")
        self._active_drivers.clear()

# Global router instance
_router_instance: Optional[DriverRouter] = None

def get_router() -> DriverRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = DriverRouter()
    return _router_instance
