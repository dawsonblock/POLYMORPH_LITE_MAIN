from pydantic import BaseModel
from dataclasses import dataclass

from retrofitkit.core.config import (
    PolymorphConfig as Config,
    SystemConfig,
    SecurityConfig,
    DAQConfig,
    RamanConfig,
    SafetyConfig,
)

# Backwards-compatible aliases for legacy tests
SystemCfg = SystemConfig
SecurityCfg = SecurityConfig
DAQCfg = DAQConfig
RamanCfg = RamanConfig
SafetyCfg = SafetyConfig


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
