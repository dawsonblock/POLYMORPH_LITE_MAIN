"""
Enhanced configuration management with environment variable support
"""
import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class SystemConfig(BaseSettings):
    """System configuration with environment variable support."""

    # Core system settings
    name: str = Field(default="POLYMORPH-4 Lite", validation_alias="P4_SYSTEM_NAME")
    environment: Environment = Field(default=Environment.DEVELOPMENT, validation_alias="P4_ENVIRONMENT")
    debug: bool = Field(default=False, validation_alias="P4_DEBUG")
    timezone: str = Field(default="UTC", validation_alias="P4_TIMEZONE")

    # Directories
    data_dir: str = Field(default="data", validation_alias="P4_DATA_DIR")
    logs_dir: str = Field(default="logs", validation_alias="P4_LOGS_DIR")
    config_dir: str = Field(default="config", validation_alias="P4_CONFIG_DIR")

    # Server settings
    host: str = Field(default="0.0.0.0", validation_alias="P4_HOST")
    port: int = Field(default=8000, validation_alias="P4_PORT")
    reload: bool = Field(default=False, validation_alias="P4_RELOAD")

    # Logging
    log_level: str = Field(default="INFO", validation_alias="P4_LOG_LEVEL")
    log_format: str = Field(default="json", validation_alias="P4_LOG_FORMAT")  # json or text

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v):
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                return Environment.DEVELOPMENT
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if isinstance(v, str):
            return v.upper() if v.upper() in valid_levels else "INFO"
        return v

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, populate_by_name=True, extra="ignore")


class SecurityConfig(BaseSettings):
    """Security configuration with environment variable support."""

    # Authentication
    jwt_secret_key: str = Field(default="your-secret-key-change-this", validation_alias="P4_JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", validation_alias="P4_JWT_ALGORITHM")
    jwt_exp_minutes: int = Field(default=480, validation_alias="P4_JWT_EXP_MINUTES")

    # Password policy
    password_min_length: int = Field(default=12, validation_alias="P4_PASSWORD_MIN_LENGTH")
    password_require_upper: bool = Field(default=True, validation_alias="P4_PASSWORD_REQUIRE_UPPER")
    password_require_digit: bool = Field(default=True, validation_alias="P4_PASSWORD_REQUIRE_DIGIT")
    password_require_symbol: bool = Field(default=True, validation_alias="P4_PASSWORD_REQUIRE_SYMBOL")

    # Compliance
    two_person_signoff: bool = Field(default=True, validation_alias="P4_TWO_PERSON_SIGNOFF")
    audit_enabled: bool = Field(default=True, validation_alias="P4_AUDIT_ENABLED")

    # RSA keys
    rsa_private_key_path: str = Field(default="config/keys/private.pem", validation_alias="P4_RSA_PRIVATE_KEY")
    rsa_public_key_path: str = Field(default="config/keys/public.pem", validation_alias="P4_RSA_PUBLIC_KEY")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, populate_by_name=True, extra="ignore")


class DAQConfig(BaseSettings):
    """DAQ configuration with environment variable support."""

    backend: str = Field(default="simulator", validation_alias="P4_DAQ_BACKEND")

    # NI DAQ settings
    ni_device_name: str = Field(default="Dev1", validation_alias="P4_NI_DEVICE_NAME")
    ni_ao_channel: str = Field(default="ao0", validation_alias="P4_NI_AO_CHANNEL")
    ni_ai_channel: str = Field(default="ai0", validation_alias="P4_NI_AI_CHANNEL")
    ni_di_lines: Union[str, list] = Field(default="port0/line0,port0/line1", validation_alias="P4_NI_DI_LINES")
    ni_do_watchdog_line: str = Field(default="port0/line2", validation_alias="P4_NI_DO_WATCHDOG")

    # Red Pitaya settings
    redpitaya_host: str = Field(default="192.168.1.100", validation_alias="P4_REDPITAYA_HOST")
    redpitaya_port: int = Field(default=5000, validation_alias="P4_REDPITAYA_PORT")
    redpitaya_ai_channel: int = Field(default=1, validation_alias="P4_REDPITAYA_AI_CHANNEL")
    redpitaya_ao_channel: int = Field(default=1, validation_alias="P4_REDPITAYA_AO_CHANNEL")

    # Simulator settings
    simulator_noise_std: float = Field(default=0.003, validation_alias="P4_SIMULATOR_NOISE_STD")
    simulator_estop: bool = Field(default=False, validation_alias="P4_SIMULATOR_ESTOP")
    simulator_door_open: bool = Field(default=False, validation_alias="P4_SIMULATOR_DOOR_OPEN")

    @field_validator("ni_di_lines", mode="before")
    @classmethod
    def parse_di_lines(cls, v):
        if isinstance(v, str):
            return v.split(",")
        return v

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, populate_by_name=True, extra="ignore")


class RamanConfig(BaseSettings):
    """Raman configuration with environment variable support."""

    provider: str = Field(default="simulator", validation_alias="P4_RAMAN_PROVIDER")

    # Simulator settings
    simulator_peak_nm: float = Field(default=532.0, validation_alias="P4_RAMAN_PEAK_NM")
    simulator_base_intensity: float = Field(default=1200.0, validation_alias="P4_RAMAN_BASE_INTENSITY")
    simulator_noise_std: float = Field(default=3.0, validation_alias="P4_RAMAN_NOISE_STD")
    simulator_drift_per_s: float = Field(default=0.8, validation_alias="P4_RAMAN_DRIFT_PER_S")

    # Ocean Optics settings
    ocean_device_index: int = Field(default=0, validation_alias="P4_OCEAN_DEVICE_INDEX")
    ocean_integration_time_us: int = Field(default=20000, validation_alias="P4_OCEAN_INTEGRATION_TIME")

    # Horiba settings
    horiba_device_id: str = Field(default="", validation_alias="P4_HORIBA_DEVICE_ID")
    horiba_grating: str = Field(default="600", validation_alias="P4_HORIBA_GRATING")

    # Andor settings
    andor_camera_serial: str = Field(default="", validation_alias="P4_ANDOR_CAMERA_SERIAL")
    andor_exposure_time_ms: int = Field(default=1000, validation_alias="P4_ANDOR_EXPOSURE_TIME")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, populate_by_name=True, extra="ignore")


class SafetyConfig(BaseSettings):
    """Safety configuration with environment variable support."""

    # Interlock settings
    estop_line: int = Field(default=0, validation_alias="P4_SAFETY_ESTOP_LINE")
    door_line: int = Field(default=1, validation_alias="P4_SAFETY_DOOR_LINE")

    # Watchdog settings
    watchdog_seconds: float = Field(default=2.0, validation_alias="P4_SAFETY_WATCHDOG_SECONDS")
    watchdog_enabled: bool = Field(default=True, validation_alias="P4_SAFETY_WATCHDOG_ENABLED")

    # Temperature monitoring
    max_temperature: float = Field(default=80.0, validation_alias="P4_SAFETY_MAX_TEMPERATURE")
    temperature_monitoring: bool = Field(default=False, validation_alias="P4_SAFETY_TEMPERATURE_MONITORING")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, populate_by_name=True, extra="ignore")


class DatabaseConfig(BaseSettings):
    """Database configuration with environment variable support."""

    # Database URL (SQLite by default)
    url: str = Field(default="sqlite:///data/audit.db", validation_alias="P4_DATABASE_URL")

    # Connection settings
    echo: bool = Field(default=False, validation_alias="P4_DATABASE_ECHO")
    pool_size: int = Field(default=5, validation_alias="P4_DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=10, validation_alias="P4_DATABASE_MAX_OVERFLOW")

    # Backup settings
    backup_enabled: bool = Field(default=True, validation_alias="P4_DATABASE_BACKUP_ENABLED")
    backup_interval_hours: int = Field(default=24, validation_alias="P4_DATABASE_BACKUP_INTERVAL")
    backup_retention_days: int = Field(default=30, validation_alias="P4_DATABASE_BACKUP_RETENTION")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, populate_by_name=True, extra="ignore")


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""

    # Metrics
    metrics_enabled: bool = Field(default=True, validation_alias="P4_METRICS_ENABLED")
    metrics_port: int = Field(default=9090, validation_alias="P4_METRICS_PORT")
    metrics_path: str = Field(default="/metrics", validation_alias="P4_METRICS_PATH")

    # Health checks
    health_check_enabled: bool = Field(default=True, validation_alias="P4_HEALTH_CHECK_ENABLED")
    health_check_interval: int = Field(default=30, validation_alias="P4_HEALTH_CHECK_INTERVAL")

    # Prometheus
    prometheus_enabled: bool = Field(default=False, validation_alias="P4_PROMETHEUS_ENABLED")
    prometheus_pushgateway_url: str = Field(default="", validation_alias="P4_PROMETHEUS_PUSHGATEWAY_URL")

    # Grafana
    grafana_enabled: bool = Field(default=False, validation_alias="P4_GRAFANA_ENABLED")
    grafana_port: int = Field(default=3000, validation_alias="P4_GRAFANA_PORT")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, populate_by_name=True, extra="ignore")


class AIConfig(BaseSettings):
    """AI/ML configuration."""

    service_url: str = Field(default="http://localhost:3000/infer", validation_alias="P4_AI_SERVICE_URL")
    enabled: bool = Field(default=True, validation_alias="P4_AI_ENABLED")
    timeout_seconds: float = Field(default=2.0, validation_alias="P4_AI_TIMEOUT")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, populate_by_name=True, extra="ignore")


class PolymorphConfig:
    """Main configuration class that combines all configuration sections."""

    def __init__(self, config_file: Optional[str] = None, **kwargs):
        self.config_file = config_file or os.getenv("P4_CONFIG", "config/config.yaml")

        # Load configurations
        self.system = kwargs.get('system') or SystemConfig()
        self.security = kwargs.get('security') or SecurityConfig()
        self.daq = kwargs.get('daq') or DAQConfig()
        self.raman = kwargs.get('raman') or RamanConfig()
        self.safety = kwargs.get('safety') or SafetyConfig()
        self.database = kwargs.get('database') or DatabaseConfig()
        self.monitoring = kwargs.get('monitoring') or MonitoringConfig()
        self.ai = kwargs.get('ai') or AIConfig()

        # Handle extra kwargs (like gating)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

        # Load YAML config if exists
        if Path(self.config_file).exists():
            self._load_yaml_config()

    def _load_yaml_config(self):
        """Load configuration from YAML file, merging with environment variables."""
        try:
            with open(self.config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)

            if not yaml_config:
                return

            # Update each configuration section
            if 'system' in yaml_config:
                self._update_config(self.system, yaml_config['system'])

            if 'security' in yaml_config:
                self._update_config(self.security, yaml_config['security'])

            if 'daq' in yaml_config:
                self._update_config(self.daq, yaml_config['daq'])

            if 'raman' in yaml_config:
                self._update_config(self.raman, yaml_config['raman'])

            if 'safety' in yaml_config:
                self._update_config(self.safety, yaml_config['safety'])

            if 'database' in yaml_config:
                self._update_config(self.database, yaml_config['database'])

            if 'monitoring' in yaml_config:
                self._update_config(self.monitoring, yaml_config['monitoring'])

            if 'ai' in yaml_config:
                self._update_config(self.ai, yaml_config['ai'])

        except Exception as e:
            print(f"Warning: Could not load YAML config from {self.config_file}: {e}")

    def _update_config(self, config_obj: BaseSettings, yaml_data: Dict[str, Any]):
        """Update a configuration object with YAML data (environment variables take precedence)."""
        for key, value in yaml_data.items():
            if hasattr(config_obj, key):
                # Only update if environment variable is not set
                # In Pydantic V2, we check model_fields[key].validation_alias
                field_info = type(config_obj).model_fields.get(key)
                env_var_name = field_info.validation_alias if field_info else None

                if env_var_name and env_var_name not in os.environ:
                    setattr(config_obj, key, value)
                elif not env_var_name:
                    setattr(config_obj, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "system": self.system.model_dump(),
            "security": self.security.model_dump(),
            "daq": self.daq.model_dump(),
            "raman": self.raman.model_dump(),
            "safety": self.safety.model_dump(),
            "database": self.database.model_dump(),
            "monitoring": self.monitoring.model_dump(),
            "ai": self.ai.model_dump()
        }

    def save_to_file(self, filename: str):
        """Save current configuration to YAML file."""
        with open(filename, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return any issues."""
        issues = {}

        # Check required directories exist
        for dir_path in [self.system.data_dir, self.system.logs_dir]:
            if not Path(dir_path).exists():
                issues.setdefault('directories', []).append(f"Directory does not exist: {dir_path}")

        # Check RSA key files exist if not in development
        if self.system.environment != Environment.DEVELOPMENT:
            for key_path in [self.security.rsa_private_key_path, self.security.rsa_public_key_path]:
                if not Path(key_path).exists():
                    issues.setdefault('security', []).append(f"RSA key file missing: {key_path}")

        # Validate DAQ configuration
        if self.daq.backend == "ni" and not self.daq.ni_device_name:
            issues.setdefault('daq', []).append("NI device name is required when using NI backend")

        if self.daq.backend == "redpitaya" and not self.daq.redpitaya_host:
            issues.setdefault('daq', []).append("Red Pitaya host is required when using Red Pitaya backend")

        return issues


# Global configuration instance
_config_instance: Optional[PolymorphConfig] = None


def get_config(config_file: Optional[str] = None) -> PolymorphConfig:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None or config_file:
        _config_instance = PolymorphConfig(config_file)
    return _config_instance


def reload_config(config_file: Optional[str] = None):
    """Reload configuration from file."""
    global _config_instance
    _config_instance = PolymorphConfig(config_file)
