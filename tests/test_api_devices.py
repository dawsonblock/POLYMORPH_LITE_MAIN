"""
Tests for device API endpoints.

This module tests:
- Device driver listing
- Device capabilities queries
- Device instance management
- Device discovery by kind and action
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from retrofitkit.api.devices import router
from retrofitkit.drivers.base import DeviceKind, DeviceCapabilities


@pytest.fixture
def app():
    """Create test app with devices router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_registry():
    """Create mock device registry."""
    with patch('retrofitkit.api.devices.registry') as mock_reg:
        # Setup mock capabilities
        mock_cap = DeviceCapabilities(
            kind=DeviceKind.SPECTROMETER,
            vendor="MockVendor",
            model="TestModel",
            actions=["acquire_spectrum", "calibrate"],
            features={"wavelength_range": [200, 1000]}
        )

        mock_reg.list_drivers.return_value = {
            "mock_driver": mock_cap
        }
        mock_reg.find_by_kind.return_value = ["mock_driver"]
        mock_reg.find_by_action.return_value = ["mock_driver"]

        yield mock_reg


class TestListDrivers:
    """Test cases for GET /devices/drivers endpoint."""

    def test_list_drivers_returns_200(self, client, mock_registry):
        """Test that list drivers returns 200 OK."""
        response = client.get("/devices/drivers")
        assert response.status_code == 200

    def test_list_drivers_returns_array(self, client, mock_registry):
        """Test that list drivers returns array."""
        response = client.get("/devices/drivers")
        assert isinstance(response.json(), list)

    def test_list_drivers_includes_capabilities(self, client, mock_registry):
        """Test that drivers include capability information."""
        response = client.get("/devices/drivers")
        drivers = response.json()

        assert len(drivers) > 0
        driver = drivers[0]
        assert "name" in driver
        assert "capabilities" in driver
        assert "kind" in driver["capabilities"]
        assert "vendor" in driver["capabilities"]
        assert "actions" in driver["capabilities"]


class TestListDriversByKind:
    """Test cases for GET /devices/drivers/by-kind/{kind}."""

    def test_find_drivers_by_kind(self, client, mock_registry):
        """Test finding drivers by kind."""
        response = client.get("/devices/drivers/by-kind/spectrometer")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestListDriversByAction:
    """Test cases for GET /devices/drivers/by-action/{action}."""

    def test_find_drivers_by_action(self, client, mock_registry):
        """Test finding drivers by action."""
        response = client.get("/devices/drivers/by-action/acquire_spectrum")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestGetDriverCapabilities:
    """Test cases for GET /devices/drivers/{name}/capabilities."""

    def test_get_existing_driver_capabilities(self, client, mock_registry):
        """Test getting capabilities for existing driver."""
        response = client.get("/devices/drivers/mock_driver/capabilities")
        assert response.status_code == 200
        caps = response.json()
        assert caps["vendor"] == "MockVendor"
        assert "acquire_spectrum" in caps["actions"]

    def test_get_nonexistent_driver_returns_404(self, client, mock_registry):
        """Test that nonexistent driver returns 404."""
        response = client.get("/devices/drivers/nonexistent/capabilities")
        assert response.status_code == 404


class TestListInstances:
    """Test cases for GET /devices/instances."""

    def test_list_instances_empty(self, client, mock_registry):
        """Test listing instances when none exist."""
        mock_registry.list_instances.return_value = {}

        response = client.get("/devices/instances")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_instances_with_devices(self, client, mock_registry):
        """Test listing active device instances."""
        mock_device = Mock()
        mock_device.health = AsyncMock(return_value={"status": "ready"})
        mock_device.capabilities = DeviceCapabilities(
            kind=DeviceKind.SPECTROMETER,
            vendor="TestVendor",
            model=None,
            actions=[],
            features={}
        )

        mock_registry.list_instances.return_value = {
            "device1": mock_device
        }

        response = client.get("/devices/instances")
        assert response.status_code == 200
        instances = response.json()
        assert len(instances) == 1
        assert instances[0]["instance_id"] == "device1"


class TestGetInstance:
    """Test cases for GET /devices/instances/{id}."""

    def test_get_existing_instance(self, client, mock_registry):
        """Test getting details for existing instance."""
        mock_device = Mock()
        mock_device.health = AsyncMock(return_value={"status": "ready"})
        mock_device.capabilities = DeviceCapabilities(
            kind=DeviceKind.DAQ,
            vendor="TestDAQ",
            model=None,
            actions=[],
            features={}
        )

        mock_registry.get_instance.return_value = mock_device

        response = client.get("/devices/instances/test_device")
        assert response.status_code == 200
        data = response.json()
        assert data["instance_id"] == "test_device"
        assert data["connected"] is True

    def test_get_nonexistent_instance_returns_404(self, client, mock_registry):
        """Test that nonexistent instance returns 404."""
        mock_registry.get_instance.return_value = None

        response = client.get("/devices/instances/nonexistent")
        assert response.status_code == 404


class TestRemoveInstance:
    """Test cases for DELETE /devices/instances/{id}."""

    def test_remove_existing_instance(self, client, mock_registry):
        """Test removing an existing instance."""
        mock_device = Mock()
        mock_device.disconnect = AsyncMock()

        mock_registry.get_instance.return_value = mock_device

        response = client.delete("/devices/instances/test_device")
        assert response.status_code == 200
        assert "removed" in response.json()["message"]

        mock_device.disconnect.assert_called_once()
        mock_registry.remove_instance.assert_called_once_with("test_device")

    def test_remove_nonexistent_instance_returns_404(self, client, mock_registry):
        """Test that removing nonexistent instance returns 404."""
        mock_registry.get_instance.return_value = None

        response = client.delete("/devices/instances/nonexistent")
        assert response.status_code == 404
