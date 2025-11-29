import pytest
from unittest.mock import patch, MagicMock
from retrofitkit.core.config import PolymorphConfig, Environment, SystemConfig, SecurityConfig

def test_production_default_secrets():
    """Verify validation catches default secrets in production."""
    # Create config with production env and default secrets
    config = PolymorphConfig()
    config.system.environment = Environment.PRODUCTION
    config.security.jwt_secret_key = "your-secret-key-change-this"
    
    issues = config.validate()
    
    assert "security" in issues
    assert any("Default JWT secret key" in i for i in issues["security"])

def test_production_short_secret():
    """Verify validation catches short secrets in production."""
    config = PolymorphConfig()
    config.system.environment = Environment.PRODUCTION
    config.security.jwt_secret_key = "short"
    
    issues = config.validate()
    
    assert "security" in issues
    assert any("too short" in i for i in issues["security"])

def test_production_secure():
    """Verify validation passes with secure secrets."""
    config = PolymorphConfig()
    config.system.environment = Environment.PRODUCTION
    config.security.jwt_secret_key = "a" * 32 # 32 chars
    
    # Mock directories existence to avoid directory errors
    with patch("pathlib.Path.exists", return_value=True):
        issues = config.validate()
        
    # Should not have security issues related to secrets
    if "security" in issues:
        assert not any("Default JWT secret key" in i for i in issues["security"])
        assert not any("too short" in i for i in issues["security"])
