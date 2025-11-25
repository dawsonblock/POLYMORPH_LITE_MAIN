from retrofitkit.drivers.raman.simulator import SimRaman
from retrofitkit.drivers.raman.vendor_stub import VendorRaman

def make_raman(cfg):
    if cfg.raman.provider == "simulator":
        return SimRaman(cfg)
    return VendorRaman(cfg)
