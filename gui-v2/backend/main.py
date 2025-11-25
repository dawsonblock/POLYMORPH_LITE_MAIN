#!/usr/bin/env python3
"""
POLYMORPH-4 Lite Modern GUI Server v2.0
Ultra-modern FastAPI backend with structured concurrency and real-time features
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import socketio

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Import POLYMORPH-4 core modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from retrofitkit.core.app import PolymorphApp
    from retrofitkit.api.routes import router as api_router
    from retrofitkit.api.health import router as health_router
    from retrofitkit.core.config import get_config
except ImportError:
    logger.warning("POLYMORPH-4 core modules not found, running in standalone mode")
    api_router = None
    health_router = None

# Modern FastAPI with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan context manager"""
    logger.info("🚀 Starting POLYMORPH-4 Lite GUI Server v2.0")

    try:
        async with asyncio.TaskGroup() as tg:
            # Start background tasks under structured concurrency
            tg.create_task(system_monitor_task())
            tg.create_task(data_generation_task())

            # Yield control while tasks run for the lifetime of the app
            yield

    except* Exception as exc_group:
        logger.error("Background tasks error", exc_info=exc_group)
    finally:
        logger.info("🛑 Shutting down POLYMORPH-4 Lite GUI Server")


# FastAPI app with modern configuration
app = FastAPI(
    title="POLYMORPH-4 Lite GUI Server v2.0",
    description="Ultra-modern WebSocket and HTTP server with React 19 frontend support",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Modern CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://polymorph4-lite.local"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO server with modern configuration
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    async_mode='asgi'
)

# Mount Socket.IO
socket_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_app)

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected via WebSocket")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected from WebSocket")
            
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if self.active_connections:
            disconnected = []
            for client_id, websocket in self.active_connections.items():
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to {client_id}: {e}")
                    disconnected.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected:
                self.disconnect(client_id)

manager = ConnectionManager()

# Modern background task with structured concurrency
async def system_monitor_task():
    """Modern system monitoring with structured concurrency"""
    logger.info("📊 Starting system monitor task")
    
    while True:
        try:
            # Generate mock system data
            system_status = {
                "overall": "healthy",
                "components": {
                    "daq": {
                        "status": "online",
                        "temperature": 23.5,
                        "lastUpdate": datetime.now().isoformat()
                    },
                    "raman": {
                        "status": "online", 
                        "temperature": 22.1,
                        "lastUpdate": datetime.now().isoformat()
                    },
                    "safety": {
                        "status": "online",
                        "lastUpdate": datetime.now().isoformat()
                    },
                    "pumps": {
                        "status": "online",
                        "pressure": 1.75,
                        "flowRate": 5.2,
                        "lastUpdate": datetime.now().isoformat()
                    },
                    "valves": {
                        "status": "online",
                        "lastUpdate": datetime.now().isoformat()
                    }
                },
                "uptime": 3600,
                "lastUpdate": datetime.now().isoformat()
            }
            
            # Emit to Socket.IO clients
            await sio.emit('system_status', system_status)
            
            # Broadcast to WebSocket clients
            await manager.broadcast({
                "type": "system_status",
                "data": system_status
            })
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error("System monitor task error", exc_info=e)
            await asyncio.sleep(10)

async def data_generation_task():
    """Generate process data simulation"""
    logger.info("📈 Starting data generation task")
    
    while True:
        try:
            # Generate mock process data
            processes = [
                {
                    "id": "proc-001",
                    "recipeId": "recipe-demo",
                    "recipeName": "Demo Synthesis Process",
                    "status": "running",
                    "startTime": datetime.now().isoformat(),
                    "currentStep": 3,
                    "totalSteps": 8,
                    "progress": 37.5,
                    "data": []
                }
            ]
            
            await sio.emit('processes_update', processes)
            await manager.broadcast({
                "type": "processes_update", 
                "data": processes
            })
            
            await asyncio.sleep(10)  # Update every 10 seconds
            
        except Exception as e:
            logger.error("Data generation task error", exc_info=e)
            await asyncio.sleep(15)

# Socket.IO event handlers
@sio.event
async def connect(sid, environ, auth):
    """Handle Socket.IO client connection"""
    logger.info(f"Socket.IO client {sid} connected")
    await sio.emit('connection_established', {'status': 'connected'}, room=sid)

@sio.event
async def disconnect(sid):
    """Handle Socket.IO client disconnection"""
    logger.info(f"Socket.IO client {sid} disconnected")

# Modern WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Modern WebSocket endpoint with proper error handling"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received from {client_id}: {data}")
            
            # Echo back for now
            await websocket.send_json({
                "type": "echo",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"WebSocket client {client_id} disconnected")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Modern health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections)
    }

# Authentication endpoints (mock for demo)
@app.post("/api/auth/login")
async def login(credentials: dict):
    """Mock login endpoint"""
    username = credentials.get("username")
    password = credentials.get("password")
    
    if username and password:
        return {
            "token": f"mock-token-{username}-{datetime.now().timestamp()}",
            "user": {
                "id": "1",
                "username": username,
                "email": f"{username}@polymorph.com",
                "role": "admin" if username == "admin" else "operator",
                "lastLogin": datetime.now().isoformat(),
                "isActive": True
            }
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )

# Include additional routers if available
if api_router:
    app.include_router(api_router, prefix="/api")
if health_router:
    app.include_router(health_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )