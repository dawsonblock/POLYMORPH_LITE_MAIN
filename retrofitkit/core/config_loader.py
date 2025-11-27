"""
Unified Configuration Loader.

Responsible for loading, merging, and resolving configuration from multiple sources:
1. Base configuration (config.yaml)
2. Hardware profiles (e.g., profiles/simulated_lab.yaml)
3. Runtime overlays (e.g., overlays/experiment_1.yaml)
4. Environment variables (via Pydantic Settings)
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from retrofitkit.core.config import PolymorphConfig, SystemConfig, SecurityConfig, DAQConfig, RamanConfig, SafetyConfig, DatabaseConfig, MonitoringConfig, AIConfig

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Loads and merges configuration from multiple layers.
    """
    def __init__(self, base_config_path: str = "config/config.yaml"):
        self.base_config_path = base_config_path
        self.config_data: Dict[str, Any] = {}
        self.loaded_files = []

    def load_base(self) -> "ConfigLoader":
        """Load the base configuration file."""
        if not os.path.exists(self.base_config_path):
            logger.warning(f"Base config not found at {self.base_config_path}. Using defaults.")
            return self

        with open(self.base_config_path, "r") as f:
            base_data = yaml.safe_load(f) or {}
            self._merge(base_data)
            self.loaded_files.append(self.base_config_path)
            logger.info(f"Loaded base config from {self.base_config_path}")
        return self

    def apply_hardware_profile(self, profile_name: str) -> "ConfigLoader":
        """
        Apply a hardware profile.
        
        Profiles are looked up in:
        1. config/profiles/{profile_name}.yaml
        2. config/profiles/{profile_name}
        """
        # Try to find the profile file
        candidates = [
            f"config/profiles/{profile_name}.yaml",
            f"config/profiles/{profile_name}",
            profile_name # Allow absolute path
        ]
        
        profile_path = None
        for path in candidates:
            if os.path.exists(path):
                profile_path = path
                break
        
        if not profile_path:
            logger.error(f"Hardware profile '{profile_name}' not found.")
            raise FileNotFoundError(f"Hardware profile '{profile_name}' not found.")

        with open(profile_path, "r") as f:
            profile_data = yaml.safe_load(f) or {}
            self._merge(profile_data)
            self.loaded_files.append(profile_path)
            logger.info(f"Applied hardware profile from {profile_path}")
            
        return self

    def apply_overlay(self, overlay_path: str) -> "ConfigLoader":
        """Apply a runtime configuration overlay."""
        if not os.path.exists(overlay_path):
            logger.warning(f"Overlay config not found at {overlay_path}. Skipping.")
            return self

        with open(overlay_path, "r") as f:
            overlay_data = yaml.safe_load(f) or {}
            self._merge(overlay_data)
            self.loaded_files.append(overlay_path)
            logger.info(f"Applied overlay from {overlay_path}")
            
        return self

    def _merge(self, source: Dict[str, Any]):
        """Deep merge source dict into self.config_data."""
        for key, value in source.items():
            if isinstance(value, dict) and key in self.config_data and isinstance(self.config_data[key], dict):
                self._deep_merge(self.config_data[key], value)
            else:
                self.config_data[key] = value

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Recursive deep merge."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def resolve(self) -> PolymorphConfig:
        """
        Resolve the final configuration into a PolymorphConfig object.
        
        This creates a PolymorphConfig and then manually injects the merged dictionary values,
        allowing Pydantic's environment variable logic to still take final precedence 
        (because PolymorphConfig uses BaseSettings).
        """
        # Create a fresh config instance (which loads env vars by default)
        # We pass None to avoid loading the default config file again logic inside __init__
        # We will manually update it.
        config = PolymorphConfig(config_file=None)
        
        # We need to bypass the _load_yaml_config logic in PolymorphConfig.__init__
        # and instead inject our merged data.
        
        # Helper to update a Pydantic model from a dict
        def update_model(model, data):
            for k, v in data.items():
                if hasattr(model, k):
                    # Check if env var is set (env vars win)
                    field = model.__fields__.get(k)
                    env_var = field.field_info.extra.get('env') if field else None
                    # In Pydantic v2 this is different, but for now assuming v1 style or compatible
                    # Actually, PolymorphConfig._update_config logic is good to reuse if possible,
                    # but it's instance method.
                    
                    # Let's use the same logic as PolymorphConfig._update_config
                    # "Only update if environment variable is not set"
                    
                    # Check if env var exists
                    if env_var and os.environ.get(env_var) is not None:
                        continue
                        
                    setattr(model, k, v)

        # Apply merged data to the config sections
        if 'system' in self.config_data:
            config._update_config(config.system, self.config_data['system'])
        if 'security' in self.config_data:
            config._update_config(config.security, self.config_data['security'])
        if 'daq' in self.config_data:
            config._update_config(config.daq, self.config_data['daq'])
        if 'raman' in self.config_data:
            config._update_config(config.raman, self.config_data['raman'])
        if 'safety' in self.config_data:
            config._update_config(config.safety, self.config_data['safety'])
        if 'database' in self.config_data:
            config._update_config(config.database, self.config_data['database'])
        if 'monitoring' in self.config_data:
            config._update_config(config.monitoring, self.config_data['monitoring'])
        if 'ai' in self.config_data:
            config._update_config(config.ai, self.config_data['ai'])
            
        return config

# Global loader instance
_loader_instance: Optional[ConfigLoader] = None

def get_loader() -> ConfigLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ConfigLoader()
    return _loader_instance
