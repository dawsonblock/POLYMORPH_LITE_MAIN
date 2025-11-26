"""
Safety Manager for workflow execution.

Enforces safety policies before device actions to prevent dangerous operations.
"""
from typing import Dict, Any, List, Protocol
from retrofitkit.drivers.base import DeviceBase


class PolicyBase(Protocol):
    """
    Base protocol for safety policies.
    
    Policies can inspect device state and action parameters
    before execution to prevent unsafe operations.
    """
    name: str
    
    async def before_action(
        self,
        device: DeviceBase,
        action: str,
        args: Dict[str, Any]
    ) -> None:
        """
        Check policy before action execution.
        
        Args:
            device: Device instance
            action: Action name to execute
            args: Action arguments
            
        Raises:
            RuntimeError: If policy violation detected
        """
        ...


class LoggingPolicy:
    """
    Example policy that logs all device actions.
    
    Useful for audit trails and debugging.
    """
    name = "logging"
    
    async def before_action(
        self,
        device: DeviceBase,
        action: str,
        args: Dict[str, Any]
    ) -> None:
        """Log action for audit trail."""
        print(f"[AUDIT] Device {device.id} executing {action} with args: {args}")


class TemperaturePolicy:
    """
    Example policy requiring temperature checks.
    
    Useful for CCD cameras (Andor) that must be cooled before long exposures.
    """
    name = "temperature_check"
    
    def __init__(self, max_temp_celsius: float = -20.0):
        """
        Initialize temperature policy.
        
        Args:
            max_temp_celsius: Maximum allowed temperature
        """
        self.max_temp_celsius = max_temp_celsius
    
    async def before_action(
        self,
        device: DeviceBase,
        action: str,
        args: Dict[str, Any]
    ) -> None:
        """
        Check temperature before spectroscopy actions.
        
        Raises:
            RuntimeError: If temperature too high
        """
        # Only check for spectrometer acquisitions
        if action != "acquire_spectrum":
            return
        
        # Only check for devices that report temperature
        health = await device.health()
        if "ccd_temp" not in health and "temperature" not in health:
            return
        
        temp = health.get("ccd_temp") or health.get("temperature")
        if temp is None:
            return
        
        if temp > self.max_temp_celsius:
            raise RuntimeError(
                f"Device {device.id} temperature too high: {temp}°C > "
                f"{self.max_temp_celsius}°C. Wait for cooling before acquisition."
            )


class VoltageRangePolicy:
    """
    Example policy enforcing DAQ voltage limits.
    
    Prevents damage to sensitive equipment.
    """
    name = "voltage_range"
    
    def __init__(self, min_volts: float = -10.0, max_volts: float = 10.0):
        """
        Initialize voltage range policy.
        
        Args:
            min_volts: Minimum safe voltage
            max_volts: Maximum safe voltage
        """
        self.min_volts = min_volts
        self.max_volts = max_volts
    
    async def before_action(
        self,
        device: DeviceBase,
        action: str,
        args: Dict[str, Any]
    ) -> None:
        """
        Check voltage is within safe range.
        
        Raises:
            RuntimeError: If voltage out of range
        """
        # Only check for voltage write actions
        if action not in ["write_ao", "set_voltage"]:
            return
        
        # Check voltage/volts parameter
        voltage = args.get("value") or args.get("volts") or args.get("voltage")
        if voltage is None:
            return
        
        if voltage < self.min_volts or voltage > self.max_volts:
            raise RuntimeError(
                f"Voltage {voltage}V out of safe range "
                f"[{self.min_volts}V, {self.max_volts}V]"
            )


class SafetyManager:
    """
    Manages safety policies for workflow execution.
    
    Checks all policies before device actions and blocks
    unsafe operations.
    
    Example:
        safety = SafetyManager()
        safety.add_policy(LoggingPolicy())
        safety.add_policy(TemperaturePolicy(max_temp_celsius=-30.0))
        
        await safety.check_before_action(device, "acquire_spectrum", {})
    """
    
    def __init__(self):
        """Initialize safety manager with empty policy list."""
        self._policies: List[PolicyBase] = []
        self._enabled = True
    
    def add_policy(self, policy: PolicyBase) -> None:
        """
        Add a safety policy.
        
        Args:
            policy: Policy implementing PolicyBase protocol
        """
        self._policies.append(policy)
    
    def remove_policy(self, policy_name: str) -> None:
        """
        Remove a policy by name.
        
        Args:
            policy_name: Name of policy to remove
        """
        self._policies = [p for p in self._policies if p.name != policy_name]
    
    def list_policies(self) -> List[str]:
        """
        Get names of all active policies.
        
        Returns:
            List of policy names
        """
        return [p.name for p in self._policies]
    
    def enable(self) -> None:
        """Enable safety checks."""
        self._enabled = True
    
    def disable(self) -> None:
        """
        Disable safety checks.
        
        WARNING: Should only be used for testing or emergency situations.
        """
        self._enabled = False
    
    async def check_before_action(
        self,
        device: DeviceBase,
        action: str,
        args: Dict[str, Any]
    ) -> None:
        """
        Run all safety policies before device action.
        
        Args:
            device: Device instance
            action: Action to execute
            args: Action arguments
            
        Raises:
            RuntimeError: If any policy blocks the action
        """
        if not self._enabled:
            return
        
        for policy in self._policies:
            await policy.before_action(device, action, args)
