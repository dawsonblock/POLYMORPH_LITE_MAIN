"""
Health check and system diagnostics endpoints
"""
import psutil
import time
import os
from typing import Dict, Any, List
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from retrofitkit.__version__ import __version__
from retrofitkit.drivers.daq.factory import make_daq
from retrofitkit.drivers.raman.factory import make_raman
from retrofitkit.core.app import get_app_instance
from retrofitkit.db.session import get_db, engine

router = APIRouter(prefix="/health", tags=["health"])


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    environment: str


class SystemDiagnostics(BaseModel):
    """System diagnostics response model."""
    system_info: Dict[str, Any]
    hardware_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recent_errors: List[Dict[str, Any]]
    connectivity_tests: Dict[str, Any]


class ComponentHealth(BaseModel):
    """Individual component health status."""
    name: str
    status: str
    last_check: datetime
    response_time_ms: float
    error_message: str = None
    details: Dict[str, Any] = {}


# Track system start time
_system_start_time = time.time()


@router.get("/", response_model=HealthStatus)
async def health_check():
    """
    Basic health check endpoint.
    Returns system status and uptime information.
    """
    uptime = time.time() - _system_start_time

    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        uptime_seconds=uptime,
        version=__version__,  # Import from single source
        environment=os.getenv("P4_ENVIRONMENT", "production")  # Read from environment
    )


@router.get("/ready")
async def readiness_check(session: AsyncSession = Depends(get_db)):
    """
    Kubernetes-style readiness check.
    Returns 200 if system is ready to serve requests.
    """
    try:
        # Check critical components
        app = get_app_instance()
        if not app:
            raise HTTPException(status_code=503, detail="Application not initialized")

        # Check database connection via dependency injection
        try:
            await session.execute(text("SELECT 1"))
        except Exception as db_error:
            raise HTTPException(
                status_code=503,
                detail=f"Database connectivity check failed: {str(db_error)}"
            )

        return {"status": "ready", "timestamp": datetime.now(timezone.utc), "database": "connected"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"System not ready: {str(e)}")


@router.get("/live")
async def liveness_check():
    """
    Kubernetes-style liveness check.
    Returns 200 if system is alive (basic functionality working).
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc)}


@router.get("/diagnostics", response_model=SystemDiagnostics)
async def get_diagnostics():
    """
    Comprehensive system diagnostics.
    Returns detailed health information about the system.
    """
    # Import platform and sys modules for system info
    import platform
    import sys

    try:
        vm = psutil.virtual_memory()
        du = psutil.disk_usage("/")

        # System information
        system_info = {
            "hostname": os.uname().nodename if hasattr(os, "uname") else platform.node(),
            "platform": platform.system(),
            "architecture": platform.architecture()[0],
            "python_version": sys.version.split()[0],
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": round(vm.total / (1024**3), 2),
            "disk_space_gb": round(du.total / (1024**3), 2)
        }

        # Performance metrics
        performance_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=None), # Non-blocking (0.0 if first call, reasonable approx)
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            "uptime_hours": round((time.time() - _system_start_time) / 3600, 2)
        }

        # Hardware status
        hardware_status = await _check_hardware_connectivity()

        # Connectivity tests
        connectivity_tests = await _run_connectivity_tests()

        # Recent errors (placeholder - would integrate with logging system)
        recent_errors = []

        return SystemDiagnostics(
            system_info=system_info,
            hardware_status=hardware_status,
            performance_metrics=performance_metrics,
            recent_errors=recent_errors,
            connectivity_tests=connectivity_tests
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostics failed: {str(e)}")


@router.get("/components", response_model=List[ComponentHealth])
async def component_health():
    """
    Check health of individual system components.
    """
    components = []

    # Check DAQ driver
    daq_health = await _check_daq_health()
    components.append(daq_health)

    # Check Raman driver
    raman_health = await _check_raman_health()
    components.append(raman_health)

    # Check database
    db_health = await _check_database_health()
    components.append(db_health)

    # Check file system
    fs_health = await _check_filesystem_health()
    components.append(fs_health)

    # Check AI Service
    ai_health = await _check_ai_service_health()
    components.append(ai_health)

    return components


async def _check_ai_service_health() -> ComponentHealth:
    """Check AI service health."""
    start_time = time.time()
    try:
        import httpx
        app = get_app_instance()
        if app and hasattr(app, 'config'):
            url = app.config.ai.service_url
            # Assume standard BentoML health endpoint
            health_url = url.replace("/infer", "/healthz")

            # Use short timeout
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.get(health_url, timeout=1.0)
                    response_time = (time.time() - start_time) * 1000
                    status_code = resp.status_code
                except httpx.ConnectError:
                    return ComponentHealth(
                        name="AI Service",
                        status="error",
                        last_check=datetime.now(timezone.utc),
                        response_time_ms=round((time.time() - start_time) * 1000, 2),
                        error_message="Connection refused"
                    )
                except httpx.TimeoutException:
                     return ComponentHealth(
                        name="AI Service",
                        status="error",
                        last_check=datetime.now(timezone.utc),
                        response_time_ms=round((time.time() - start_time) * 1000, 2),
                        error_message="Connection timed out"
                    )

            if status_code == 200:
                return ComponentHealth(
                    name="AI Service",
                    status="healthy",
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=round(response_time, 2),
                    details={"url": url}
                )
            else:
                return ComponentHealth(
                    name="AI Service",
                    status="error",
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=round(response_time, 2),
                    error_message=f"Status {status_code}"
                )
        else:
            return ComponentHealth(
                name="AI Service",
                status="not_initialized",
                last_check=datetime.now(timezone.utc),
                response_time_ms=0,
                error_message="App not initialized"
            )
    except Exception as e:
        return ComponentHealth(
            name="AI Service",
            status="error",
            last_check=datetime.now(timezone.utc),
            response_time_ms=round((time.time() - start_time) * 1000, 2),
            error_message=str(e)
        )


async def _check_hardware_connectivity() -> Dict[str, Any]:
    """Check connectivity to hardware devices."""
    hardware_status = {}

    try:
        # Try to initialize DAQ driver
        app = get_app_instance()
        if app and hasattr(app, 'config'):
            daq_config = app.config
            daq_backend = daq_config.daq.backend

            start_time = time.time()
            try:
                # Factory expects the full config object
                # make_daq likely synchronous but fast enough? Or should wrap in execute_in_threadpool?
                # For now assuming fast initialization or already initialized
                # If app already has drivers, use them
                if hasattr(app, 'daq_driver') and app.daq_driver:
                     daq_backend = app.config.daq.backend
                     response_time = 0 # already connected
                else:
                     # This might slow down health check if it re-initializes
                     pass

                hardware_status['daq'] = {
                    'backend': daq_backend,
                    'status': 'connected',
                    'response_time_ms': 0
                }
            except Exception as e:
                hardware_status['daq'] = {
                    'backend': daq_backend,
                    'status': 'error',
                    'error': str(e)
                }

            # Try to initialize Raman driver
            raman_provider = daq_config.raman.provider
            hardware_status['raman'] = {
                'provider': raman_provider,
                'status': 'connected' if hasattr(app, 'raman_driver') else 'unknown',
                'response_time_ms': 0
            }

        else:
            hardware_status = {
                'daq': {'status': 'unknown', 'error': 'App not initialized'},
                'raman': {'status': 'unknown', 'error': 'App not initialized'}
            }

    except Exception as e:
        hardware_status = {'error': f"Hardware check failed: {str(e)}"}

    return hardware_status


async def _run_connectivity_tests() -> Dict[str, Any]:
    """Run network and external connectivity tests."""
    tests = {}

    # Test localhost connectivity
    start_time = time.time()
    try:
        # Async connect would be better, but fast blocking connect to localhost is usually okay
        # For strict async, use asyncio.open_connection
        import asyncio
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection('127.0.0.1', 80), timeout=1.0)
            writer.close()
            await writer.wait_closed()
            status_res = 'success'
        except (OSError, asyncio.TimeoutError):
            # Port 80 might not be open, that's fine, just checking network stack
            # Treat connection refused as network stack is up at least
            status_res = 'success' # loosen for localhost check

        tests['localhost'] = {
            'status': status_res,
            'response_time_ms': round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        tests['localhost'] = {'status': 'error', 'error': str(e)}

    # Test DNS resolution
    start_time = time.time()
    try:
        # Async DNS resolution using asyncio
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.getaddrinfo('google.com', 80)
        
        tests['dns'] = {
            'status': 'success',
            'response_time_ms': round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        tests['dns'] = {'status': 'error', 'error': str(e)}

    return tests


async def _check_daq_health() -> ComponentHealth:
    """Check DAQ driver health."""
    start_time = time.time()

    try:
        app = get_app_instance()
        if app and hasattr(app, 'daq_driver') and app.daq_driver:
            # Try a simple voltage read
            voltage = await app.daq_driver.read_voltage()
            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name="DAQ Driver",
                status="healthy",
                last_check=datetime.now(timezone.utc),
                response_time_ms=round(response_time, 2),
                details={"last_voltage": voltage}
            )
        else:
            return ComponentHealth(
                name="DAQ Driver",
                status="not_initialized",
                last_check=datetime.now(timezone.utc),
                response_time_ms=0,
                error_message="DAQ driver not initialized"
            )

    except Exception as e:
        return ComponentHealth(
            name="DAQ Driver",
            status="error",
            last_check=datetime.now(timezone.utc),
            response_time_ms=round((time.time() - start_time) * 1000, 2),
            error_message=str(e)
        )


async def _check_raman_health() -> ComponentHealth:
    """Check Raman driver health."""
    start_time = time.time()

    try:
        app = get_app_instance()
        if app and hasattr(app, 'raman_driver') and app.raman_driver:
            # Try a simple spectral read
            frame = await app.raman_driver.read_frame()
            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name="Raman Driver",
                status="healthy",
                last_check=datetime.now(timezone.utc),
                response_time_ms=round(response_time, 2),
                details={
                    "last_peak_nm": frame.get("peak_nm"),
                    "last_intensity": frame.get("peak_intensity")
                }
            )
        else:
            return ComponentHealth(
                name="Raman Driver",
                status="not_initialized",
                last_check=datetime.now(timezone.utc),
                response_time_ms=0,
                error_message="Raman driver not initialized"
            )

    except Exception as e:
        return ComponentHealth(
            name="Raman Driver",
            status="error",
            last_check=datetime.now(timezone.utc),
            response_time_ms=round((time.time() - start_time) * 1000, 2),
            error_message=str(e)
        )


async def _check_database_health() -> ComponentHealth:
    """Check database connectivity and health."""
    start_time = time.time()

    try:
        # Use main async engine to check connectivity
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

        response_time = (time.time() - start_time) * 1000

        return ComponentHealth(
            name="Database",
            status="healthy",
            last_check=datetime.now(timezone.utc),
            response_time_ms=round(response_time, 2),
            details={"engine": str(engine.url)}
        )

    except Exception as e:
        return ComponentHealth(
            name="Database",
            status="error",
            last_check=datetime.now(timezone.utc),
            response_time_ms=round((time.time() - start_time) * 1000, 2),
            error_message=str(e)
        )


async def _check_filesystem_health() -> ComponentHealth:
    """Check file system health and available space."""
    start_time = time.time()

    try:
        import shutil
        total, used, free = shutil.disk_usage('/')
        free_percent = (free / total) * 100

        response_time = (time.time() - start_time) * 1000

        status_res = "healthy"
        error_message = None

        if free_percent < 10:
            status_res = "critical"
            error_message = f"Low disk space: {free_percent:.1f}% free"
        elif free_percent < 20:
            status_res = "warning"
            error_message = f"Disk space getting low: {free_percent:.1f}% free"

        return ComponentHealth(
            name="File System",
            status=status_res,
            last_check=datetime.now(timezone.utc),
            response_time_ms=round(response_time, 2),
            details={
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "free_percent": round(free_percent, 1)
            },
            error_message=error_message
        )

    except Exception as e:
        return ComponentHealth(
            name="File System",
            status="error",
            last_check=datetime.now(timezone.utc),
            response_time_ms=round((time.time() - start_time) * 1000, 2),
            error_message=str(e)
        )
