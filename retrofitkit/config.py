from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
import os

class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "POLYMORPH-LITE"
    APP_VERSION: str = "8.0.0"
    ENV: str = "development"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/polymorph.db"
    
    # Security
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Hardware Simulation
    SIMULATE_HARDWARE: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
