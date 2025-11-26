from pydantic_settings import BaseSettings
from pydantic import BaseModel
import yaml, os
from dataclasses import dataclass

class SystemCfg(BaseModel):
    name: str
    mode: str
    timezone: str
    data_dir: str
    logs_dir: str

class SecurityCfg(BaseModel):
    password_policy: dict
    two_person_signoff: bool
    jwt_exp_minutes: int
    rsa_private_key: str
    rsa_public_key: str

class DAQCfg(BaseModel):
    backend: str
    ni: dict
    redpitaya: dict
    simulator: dict

class RamanCfg(BaseModel):
    provider: str
    simulator: dict
    vendor: dict

class GatingCfg(BaseModel):
    rules: list

class SafetyCfg(BaseModel):
    interlocks: dict
    watchdog_seconds: float

class Config(BaseModel):
    system: SystemCfg
    security: SecurityCfg
    daq: DAQCfg
    raman: RamanCfg
    gating: GatingCfg
    safety: SafetyCfg

@dataclass
class AppContext:
    config: Config
    _instance = None

    @staticmethod
    def load():
        path = os.environ.get("P4_CONFIG", "config/config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = Config(**raw)
        instance = AppContext(cfg)
        AppContext._instance = instance
        return instance

def get_app_instance():
    return AppContext._instance

def create_app_instance():
    return AppContext.load()
