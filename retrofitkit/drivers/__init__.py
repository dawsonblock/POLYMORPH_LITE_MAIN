"""
Central driver initialization.

Importing this module guarantees that all core production drivers
register themselves with the global registry.
"""

# DAQ backends
# from .daq import ni  # noqa: F401 # NI driver not fully implemented in this context yet, skipping to avoid import error if file missing
from .daq import redpitaya  # noqa: F401

# Raman backends
from .raman import vendor_ocean_optics  # noqa: F401
from .raman import vendor_horiba  # noqa: F401
from .raman import vendor_andor  # noqa: F401
