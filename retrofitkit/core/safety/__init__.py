"""
Safety module for POLYMORPH-LITE.

Provides safety guardrails, interlocks, and watchdog functionality.
"""

from .guardrails import SafetyGuardrails
from .interlocks import InterlockController, get_interlocks, SafetyError
from .watchdog import SystemWatchdog, WatchdogError

__all__ = [
    "SafetyGuardrails",
    "InterlockController",
    "get_interlocks",
    "SafetyError",
    "SystemWatchdog",
    "WatchdogError",
]
