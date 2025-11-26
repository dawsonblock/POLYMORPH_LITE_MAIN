"""
DAQ driver factory using DeviceRegistry.

Provides backward-compatible factory function that creates DAQ drivers
via the DeviceRegistry based on configuration.
"""
from retrofitkit.core.registry import registry
from retrofitkit.drivers.daq.simulator import SimDAQ
from retrofitkit.drivers.daq.redpitaya import RedPitayaDAQ
from retrofitkit.drivers.daq.ni import NIDAQ
from retrofitkit.drivers.daq.vendor_gamry import GamryPotentiostat


def make_daq(cfg):
    """
    Create a DAQ driver based on configuration.
    
    Now uses DeviceRegistry for extensibility. Falls back to direct
    instantiation if registry lookup fails (for gradual migration).
    
    Args:
        cfg: Configuration object with cfg.daq.backend
        
    Returns:
        DAQ driver instance
    """
    backend = cfg.daq.backend
    
    # Try Dev iceRegistry first (Option C path)
    try:
        driver = registry.create(backend, cfg=cfg)
        return driver
    except KeyError:
        pass  # Fall back to legacy path
    
    # Legacy fallback (will be removed once all drivers registered)
    if backend == "simulator":
        return SimDAQ(cfg)
    if backend == "redpitaya":
        return RedPitayaDAQ(cfg)
    if backend == "ni":
        return NIDAQ(cfg)
    if backend == "gamry":
        return GamryPotentiostat(cfg)
    
    # Default to simulator for unknown backends
    return SimDAQ(cfg)
