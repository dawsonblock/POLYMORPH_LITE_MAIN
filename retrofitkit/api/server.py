from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # type: ignore[import]
from fastapi.responses import HTMLResponse, PlainTextResponse  # type: ignore[import]
from fastapi.staticfiles import StaticFiles  # type: ignore[import]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import]
from contextlib import asynccontextmanager
import asyncio
import socketio
from typing import Dict, List, Any

from retrofitkit.core.app import AppContext
from retrofitkit.api.auth import router as auth_router
from retrofitkit.api.routes import router as api_router
from retrofitkit.core.raman_stream import RamanStreamer
from retrofitkit.core.events import EventBus
from retrofitkit.drivers.raman.factory import make_raman
from retrofitkit.core.app import get_app_instance, create_app_instance
from retrofitkit.core.config import PolymorphConfig
from retrofitkit.__version__ import __version__
from retrofitkit.metrics.exporter import Metrics
from retrofitkit.security.headers import SecurityHeadersMiddleware, RateLimitMiddleware
import time
from datetime import datetime, timezone

from retrofitkit.api import auth, system

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

# Background tasks
async def system_monitor_task():
    """Generate system status updates"""
    while True:
        try:
            # Mock system status (to be replaced with real Orchestrator data)
            system_status = {
                "overall": "healthy",
                "components": {
                    "daq": {"status": "online", "temperature": 23.5, "lastUpdate": datetime.now().isoformat()},
                    "raman": {"status": "online", "temperature": 22.1, "lastUpdate": datetime.now().isoformat()},
                    "safety": {"status": "online", "lastUpdate": datetime.now().isoformat()},
                },
                "uptime": int(time.time() - _start_time),
                "lastUpdate": datetime.now().isoformat()
            }
            
            await sio.emit('system_status', system_status)
            await manager.broadcast({"type": "system_status", "data": system_status})
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Monitor error: {e}")
            await asyncio.sleep(5)

async def data_generation_task():
    """Generate process data simulation"""
    while True:
        try:
            # Mock process data
            processes = [{
                "id": "proc-001",
                "recipeId": "recipe-demo",
                "recipeName": "Demo Synthesis",
                "status": "running",
                "progress": (int(time.time()) % 100),
                "data": []
            }]
            
            await sio.emit('processes_update', processes)
            await manager.broadcast({"type": "processes_update", "data": processes})
            await asyncio.sleep(2)
        except Exception:
            await asyncio.sleep(5)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await raman_streamer.start()
    
    # Start background tasks
    monitor_task = asyncio.create_task(system_monitor_task())
    data_task = asyncio.create_task(data_generation_task())
    
    yield
    
    # Shutdown
    monitor_task.cancel()
    data_task.cancel()
    await raman_streamer.stop()

# Create FastAPI application
app = FastAPI(
    title="POLYMORPH-4 Lite",
    version=__version__,
    description="Unified Retrofit Control + Raman-Gating Kit for Analytical Instrument Automation",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Mount Socket.IO
socket_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_app)

# Add security middleware (order matters!)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    max_age=3600,
)

ctx = AppContext.load()
bus = EventBus()
raman_streamer = RamanStreamer(ctx, bus)
_start_time = time.time()

from retrofitkit.api.health import router as health_router
from retrofitkit.api.devices import router as devices_router
from retrofitkit.api.workflows import router as workflows_router

# Import drivers to trigger registry auto-registration
from retrofitkit.drivers.raman import vendor_ocean_optics  # noqa: F401

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(api_router, prefix="/api", tags=["api"])
app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(devices_router, prefix="/api", tags=["devices"])
app.include_router(workflows_router, prefix="/api", tags=["workflows"])

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
