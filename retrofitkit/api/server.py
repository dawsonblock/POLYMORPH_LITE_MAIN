from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import socketio
from typing import Dict, List

from retrofitkit.core.app import AppContext
from retrofitkit.core.orchestrator import Orchestrator
from retrofitkit.api.auth import router as auth_router
from retrofitkit.api.routes import router as api_router
from retrofitkit.core.database import init_db
from retrofitkit.core.workflow.runner import workflow_runner
from retrofitkit.core.events import EventBus
from retrofitkit.__version__ import __version__
from retrofitkit.metrics.exporter import Metrics
from retrofitkit.security.headers import SecurityHeadersMiddleware, RateLimitMiddleware
import time
from datetime import datetime, timezone
from retrofitkit.config import settings
from retrofitkit.logging import logger
import numpy as np

# Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    async_mode='asgi'
)


class ConnectionManager:
    """Optimized WebSocket connection manager with batch operations."""
    
    __slots__ = ('active_connections',)
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients using concurrent sends."""
        if not self.active_connections:
            return
            
        # Use asyncio.gather for concurrent sends
        async def send_to_client(client_id: str, websocket: WebSocket) -> str | None:
            try:
                await websocket.send_json(message)
                return None
            except Exception:
                return client_id
        
        results = await asyncio.gather(
            *(send_to_client(cid, ws) for cid, ws in self.active_connections.items()),
            return_exceptions=True
        )
        
        # Remove disconnected clients
        for result in results:
            if isinstance(result, str):
                self.disconnect(result)


manager = ConnectionManager()


# Pre-compute static wavelength array for spectra simulation (optimization)
_WAVELENGTHS = np.array([400 + i * 0.5 for i in range(800)])
_WAVELENGTHS_LIST = _WAVELENGTHS.tolist()
_PEAK_CENTER = 532.0
_PEAK_FACTOR = -1.0 / 10.0


async def system_monitor_task():
    """Generate system status updates."""
    while True:
        try:
            await manager.broadcast({
                "type": "status",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "ok"
            })
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            await asyncio.sleep(5)


async def broadcast_spectra_task():
    """Broadcast Raman spectra to frontend using vectorized numpy operations."""
    while True:
        try:
            t = time.time()
            
            # Vectorized computation (much faster than list comprehension)
            peak_diff = _WAVELENGTHS - _PEAK_CENTER
            gaussian = np.exp(peak_diff * peak_diff * _PEAK_FACTOR)
            amplitude = 0.8 + 0.4 * np.sin(t)
            intensities = 100.0 + 1000.0 * gaussian * amplitude + np.random.normal(0, 5, len(_WAVELENGTHS))
            
            data = {
                "wavelengths": _WAVELENGTHS_LIST,
                "intensities": intensities.tolist(),
                "t": t
            }

            await sio.emit('spectral_data', data)
            await asyncio.sleep(0.1)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Spectra broadcast error: {e}")
            await asyncio.sleep(1)


async def data_generation_task(orc: Orchestrator):
    """Generate process data simulation."""
    while True:
        try:
            active_run_id = orc.status.get("active_run_id")

            if active_run_id:
                progress = orc.status.get("progress", {"current": 0, "total": 1})
                processes = [{
                    "id": str(active_run_id),
                    "recipeId": "active-recipe",
                    "recipeName": "Active Run",
                    "status": "running",
                    "progress": int(100 * progress["current"] / max(progress["total"], 1)),
                    "data": []
                }]
            else:
                processes = []

            await sio.emit('processes_update', processes)
            await manager.broadcast({"type": "processes_update", "data": processes})
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("Starting POLYMORPH-LITE v8.0...")
    await init_db()
    
    Metrics.start()

    # Start background tasks
    monitor_task = asyncio.create_task(system_monitor_task())
    spectra_task = asyncio.create_task(broadcast_spectra_task())

    yield

    # Shutdown: cancel background tasks gracefully
    logger.info("Shutting down...")
    monitor_task.cancel()
    spectra_task.cancel()
    
    # Wait for tasks to complete cancellation
    await asyncio.gather(monitor_task, spectra_task, return_exceptions=True)


# Create FastAPI application (single instance)
app = FastAPI(
    title="POLYMORPH-4 Lite",
    version=__version__,
    description="Unified Retrofit Control + Raman-Gating Kit for Analytical Instrument Automation",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Register error handlers
from retrofitkit.api.errors import register_exception_handlers
register_exception_handlers(app)

# Mount Socket.IO
socket_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_app)

# Add security headers middleware (order matters!)
if settings.ENV != "testing":
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, requests=100)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    max_age=3600,
)

ctx = AppContext.load()
orc = Orchestrator(ctx)
bus = EventBus()
# raman_streamer = RamanStreamer(ctx, bus, device=orc.raman)
_start_time = time.time()

from retrofitkit.api.health import router as health_router
from retrofitkit.api.devices import router as devices_router
from retrofitkit.api.workflows import router as workflows_router
from retrofitkit.api.samples import router as samples_router
from retrofitkit.api.inventory import router as inventory_router
from retrofitkit.api.calibration import router as calibration_router
from retrofitkit.api.workflow_builder import router as workflow_builder_router
from retrofitkit.api.compliance import router as compliance_router
from retrofitkit.api.polymorph import router as polymorph_router  # v4.0: Polymorph Discovery

# Import drivers to trigger registry auto-registration
from retrofitkit.drivers.raman import vendor_ocean_optics  # noqa: F401

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(api_router, prefix="/api", tags=["api"])
app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(devices_router, prefix="/api", tags=["devices"])
app.include_router(workflows_router, prefix="/api", tags=["workflows"])
app.include_router(samples_router, tags=["samples"])
app.include_router(inventory_router, tags=["inventory"])
app.include_router(calibration_router, tags=["calibration"])
app.include_router(workflow_builder_router, tags=["workflow-builder"])
app.include_router(compliance_router, tags=["compliance"])
app.include_router(polymorph_router, tags=["polymorph"])

app.mount(
    "/static",
    StaticFiles(directory="retrofitkit/api/static"),
    name="static",
)

# Socket.IO Events
@sio.event
async def connect(sid, environ, auth):
    print(f"Socket.IO client {sid} connected")
    await sio.emit('connection_established', {'status': 'connected'}, room=sid)

@sio.event
async def disconnect(sid):
    print(f"Socket.IO client {sid} disconnected")

# WebSocket Endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("retrofitkit/api/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(Metrics.get().render_prom())

# Legacy root health endpoints
@app.get("/health")
async def root_health():
    from retrofitkit.api.health import health_check
    return await health_check()

@app.get("/health/ready")
async def root_ready():
    from retrofitkit.api.health import readiness_check
    return await readiness_check()

@app.get("/health/live")
async def root_live():
    from retrofitkit.api.health import liveness_check
    return await liveness_check()
