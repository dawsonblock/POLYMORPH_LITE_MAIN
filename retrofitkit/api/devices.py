"""
Device API endpoints for unified device discovery and control.

Provides REST interface to:
- List available device drivers
- Query device capabilities
- Create/manage device instances
- Execute device actions
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from retrofitkit.core.registry import registry
from retrofitkit.drivers.base import DeviceKind


router = APIRouter(prefix="/devices", tags=["devices"])


class DeviceCapabilitiesResponse(BaseModel):
    """Response model for device capabilities."""
    kind: str
    vendor: str
    model: Optional[str] = None
    actions: List[str]
    features: Dict[str, Any]


class DeviceDriverResponse(BaseModel):
    """Response model for device driver listing."""
    name: str
    capabilities: DeviceCapabilitiesResponse


class DeviceInstanceResponse(BaseModel):
    """Response model for device instance."""
    instance_id: str
    driver_name: str
    connected: bool
    health: Dict[str, Any]


@router.get("/drivers", response_model=List[DeviceDriverResponse])
async def list_drivers():
    """
    List all registered device drivers and their capabilities.
    
    Returns:
        List of available drivers with capability information
    """
    drivers_caps = registry.list_drivers()

    return [
        DeviceDriverResponse(
            name=name,
            capabilities=DeviceCapabilitiesResponse(
                kind=cap.kind.value,
                vendor=cap.vendor,
                model=cap.model,
                actions=cap.actions,
                features=cap.features,
            )
        )
        for name, cap in drivers_caps.items()
    ]


@router.get("/drivers/by-kind/{kind}", response_model=List[str])
async def list_drivers_by_kind(kind: DeviceKind):
    """
    Find drivers by device kind.
    
    Args:
        kind: Device kind (spectrometer, daq, motion, etc.)
        
    Returns:
        List of driver names matching the kind
    """
    return registry.find_by_kind(kind)


@router.get("/drivers/by-action/{action}", response_model=List[str])
async def list_drivers_by_action(action: str):
    """
    Find drivers supporting a specific action.
    
    Args:
        action: Action name (e.g., "acquire_spectrum", "read_ai")
        
    Returns:
        List of driver names supporting the action
    """
    return registry.find_by_action(action)


@router.get("/drivers/{driver_name}/capabilities")
async def get_driver_capabilities(driver_name: str):
    """
    Get capabilities for a specific driver.
    
    Args:
        driver_name: Name of registered driver
        
    Returns:
        Driver capabilities
        
    Raises:
        404: If driver not found
    """
    drivers = registry.list_drivers()

    if driver_name not in drivers:
        raise HTTPException(
            status_code=404,
            detail=f"Driver '{driver_name}' not found"
        )

    cap = drivers[driver_name]
    return DeviceCapabilitiesResponse(
        kind=cap.kind.value,
        vendor=cap.vendor,
        model=cap.model,
        actions=cap.actions,
        features=cap.features,
    )


@router.get("/instances", response_model=List[DeviceInstanceResponse])
async def list_instances():
    """
    List all active device instances.
    
    Returns:
        List of device instances with status
    """
    instances = registry.list_instances()

    results = []
    for instance_id, device in instances.items():
        try:
            health = await device.health()
        except Exception as e:
            health = {"status": "error", "error": str(e)}

        # Infer driver name from capabilities
        driver_name = device.capabilities.vendor

        results.append(
            DeviceInstanceResponse(
                instance_id=instance_id,
                driver_name=driver_name,
                connected=True,  # If instance exists, assume connected
                health=health,
            )
        )

    return results


@router.get("/instances/{instance_id}")
async def get_instance(instance_id: str):
    """
    Get details for a specific device instance.
    
    Args:
        instance_id: Instance ID
        
    Returns:
        Device instance details
        
    Raises:
        404: If instance not found
    """
    device = registry.get_instance(instance_id)

    if device is None:
        raise HTTPException(
            status_code=404,
            detail=f"Instance '{instance_id}' not found"
        )

    try:
        health = await device.health()
    except Exception as e:
        health = {"status": "error", "error": str(e)}

    driver_name = device.capabilities.vendor

    return DeviceInstanceResponse(
        instance_id=instance_id,
        driver_name=driver_name,
        connected=True,
        health=health,
    )


@router.delete("/instances/{instance_id}")
async def remove_instance(instance_id: str):
    """
    Remove a device instance.
    
    Args:
        instance_id: Instance ID to remove
        
    Returns:
        Success message
        
    Raises:
        404: If instance not found
    """
    device = registry.get_instance(instance_id)

    if device is None:
        raise HTTPException(
            status_code=404,
            detail=f"Instance '{instance_id}' not found"
        )

    # Disconnect device
    try:
        await device.disconnect()
    except Exception:
        # Log but don't fail on disconnect error
        pass

    # Remove from registry
    registry.remove_instance(instance_id)

    return {"message": f"Instance '{instance_id}' removed"}
