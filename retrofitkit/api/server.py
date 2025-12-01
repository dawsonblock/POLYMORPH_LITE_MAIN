from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import socketio
from typing import Dict

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

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if self.active_connections:
            disconnected = []
            for client_id, websocket in self.active_connections.items():
                try:
                    await websocket.send_json(message)
                except Exception:
                    disconnected.append(client_id)
            for client_id in disconnected:
                self.disconnect(client_id)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting POLYMORPH-LITE v8.0...")
    await init_db()
    
    # Start metrics exporter
    Metrics.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)

# Socket.IO
app_sio = socketio.ASGIApp(sio, app)

# Routers
app.include_router(auth_router, prefix="/auth", tags=["Auth"])
app.include_router(api_router, prefix="/api/v1", tags=["API"])

@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo for now
            await manager.broadcast({"client": client_id, "message": data})
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# Background tasks
async def system_monitor_task():
    """Generate system status updates"""
    while True:
        try:
            # Broadcast status every 5 seconds
            await manager.broadcast({
                "type": "status",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "ok"
            })
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            await asyncio.sleep(5)

@app.on_event("startup")
async def start_monitor():
    asyncio.create_task(system_monitor_task())
    while True:
        try:
            # Get real status from Orchestrator
            orc_status = orc.status

            # Get hardware health
            daq_health = await orc.daq.health()
            raman_health = await orc.raman.health()

            system_status = {
                "overall": "healthy" if not orc_status["ai_circuit_open"] else "warning",
                "components": {
                    "daq": {
                        "status": daq_health.get("status", "unknown"),
                        "temperature": 23.5, # Placeholder until driver supports temp
                        "lastUpdate": datetime.now().isoformat()
                    },
                    "raman": {
                        "status": raman_health.get("status", "unknown"),
                        "temperature": 22.1,
                        "lastUpdate": datetime.now().isoformat()
                    },
                    "ai": {
                        "status": "offline" if orc_status["ai_circuit_open"] else "online",
                        "circuit_open": orc_status["ai_circuit_open"],
                        "failures": orc_status["ai_failures"],
                        "lastUpdate": datetime.now().isoformat()
                    },
                    "safety": {
                        "status": "online",
                        "lastUpdate": datetime.now().isoformat()
                    }
                },
                "uptime": int(time.time() - _start_time),
                "lastUpdate": datetime.now().isoformat()
            }

            await sio.emit('system_status', system_status)
            await manager.broadcast({"type": "system_status", "data": system_status})
            await asyncio.sleep(2) # Faster updates for responsiveness
        except Exception as e:
            print(f"Monitor error: {e}")
            await asyncio.sleep(5)

async def broadcast_spectra_task():
    """Broadcast Raman spectra to frontend"""
    # Simulate spectra stream if no real device
    while True:
        try:
            # In a real scenario, this would iterate over raman_streamer.frames()
            # For now, we generate a synthetic frame if streamer isn't active
            
            # Synthetic frame
            t = time.time()
            wavelengths = [400 + i*0.5 for i in range(800)]
            # Dynamic peak at 532nm
            intensities = [
                100.0 + 1000.0 * np.exp(-((wl - 532.0)**2) / 10.0) * (0.8 + 0.4 * np.sin(t)) + np.random.normal(0, 5)
                for wl in wavelengths
            ]
            
            data = {
                "wavelengths": wavelengths,
                "intensities": intensities,
                "t": t
            }

            # Emit to Socket.IO
            await sio.emit('spectral_data', data)
            
            # Throttle to ~10Hz for UI performance
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Spectra broadcast error: {e}")
            await asyncio.sleep(1)

async def data_generation_task():
    """Generate process data simulation"""
    while True:
        try:
            # Get active run from Orchestrator
            active_run_id = orc.status.get("active_run_id")

            if active_run_id:
                processes = [{
                    "id": str(active_run_id),
                    "recipeId": "active-recipe",
                    "recipeName": "Active Run",
                    "status": "running",
                    "recipeName": "Active Run",
                    "status": "running",
                    "progress": int(100 * orc.status["progress"]["current"] / max(orc.status["progress"]["total"], 1)),
                    "data": []
                }]
            else:
                processes = []

            await sio.emit('processes_update', processes)
            await manager.broadcast({"type": "processes_update", "data": processes})
            await asyncio.sleep(2)
        except Exception:
            await asyncio.sleep(5)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting POLYMORPH-LITE v8.0...")
    await init_db()
    
    # Start metrics exporter
    Metrics.start()
    
    # await raman_streamer.start()

    # Start background tasks
    monitor_task = asyncio.create_task(system_monitor_task())
    # data_task = asyncio.create_task(data_generation_task())
    spectra_task = asyncio.create_task(broadcast_spectra_task())

    yield

    # Shutdown
    logger.info("Shutting down...")
    monitor_task.cancel()
    # data_task.cancel()
    spectra_task.cancel()
    # await raman_streamer.stop()

# Create FastAPI application
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
