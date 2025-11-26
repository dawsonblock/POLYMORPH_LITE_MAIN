"""
Raman driver factory using DeviceRegistry.

Provides backward-compatible factory function that creates Raman drivers
via the DeviceRegistry based on configuration.
"""
from retrofitkit.core.registry import registry
from retrofitkit.drivers.raman.simulator import SimRaman
from retrofitkit.drivers.raman.vendor_stub import VendorRaman


def make_raman(cfg):
    """
    Create a Raman driver based on configuration.
    
    Now uses DeviceRegistry for extensibility. Falls back to direct
    instantiation if registry lookup fails (for gradual migration).
    
    Args:
        cfg: Configuration object with cfg.raman.provider
        
    Returns:
        Raman driver instance
    """
    provider = cfg.raman.provider
    
    # Try DeviceRegistry first (Option C path)
    try:
        driver = registry.create(provider, cfg=cfg)
        return driver
    except KeyError:
        pass  # Fall back to legacy path
    
    # Legacy fallback (will be removed once all drivers registered)
    if provider == "simulator":
        return SimRaman(cfg)
    
    # Default to stub for unknown providers
    return VendorRaman(cfg)
