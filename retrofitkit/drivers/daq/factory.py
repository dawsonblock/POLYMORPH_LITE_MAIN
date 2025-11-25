from retrofitkit.drivers.daq.simulator import SimDAQ
from retrofitkit.drivers.daq.redpitaya import RedPitayaDAQ
from retrofitkit.drivers.daq.ni import NIDAQ
from retrofitkit.drivers.daq.vendor_gamry import GamryPotentiostat

def make_daq(cfg):
    be = cfg.daq.backend
    if be == "simulator":
        return SimDAQ(cfg)
    if be == "redpitaya":
        return RedPitayaDAQ(cfg)
    if be == "ni":
        return NIDAQ(cfg)
    if be == "gamry":
        return GamryPotentiostat(cfg)
    return SimDAQ(cfg)
