from pydantic import BaseModel
from dataclasses import dataclass

from retrofitkit.core.config import (
    PolymorphConfig as Config,
    # GatingCfg is not in config.py, we might need to keep it or remove it if unused
)

# GatingCfg seems to be missing from config.py.
# If it's used, we should define it or find where it belongs.
# For now, I'll keep a local definition if it's not in config.py,
# but looking at config.py, there is no GatingConfig.
# I'll define it here for now to avoid breaking imports,
# but ideally it should be in config.py.

class GatingCfg(BaseModel):
    rules: list = []

@dataclass
class AppContext:
    config: Config
    _instance = None

    @staticmethod
    def load():
        from retrofitkit.core.config_loader import get_loader

        # Use ConfigLoader to resolve configuration
        config = get_loader().load_base().resolve()

        instance = AppContext(config)
        AppContext._instance = instance
        return instance

def get_app_instance():
    return AppContext._instance
