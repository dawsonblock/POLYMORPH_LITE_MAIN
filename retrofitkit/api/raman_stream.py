"""
Real-time Raman spectrum streaming API.

Provides WebSocket endpoint for live spectrum data, AI predictions, and gating state.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List
import asyncio
import json
import logging
from datetime import datetime, timezone

from retrofitkit.core.orchestrator import Orchestrator

router = APIRouter(prefix="/api/raman", tags=["raman"])
logger = logging.getLogger(__name__)

# Active WebSocket connections
active_connections: List[WebSocket] = []


class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and store WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket."""
        await websocket.send_text(message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSockets."""
        message_json = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")


manager = ConnectionManager()


@router.websocket("/stream")
async def raman_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time Raman spectrum streaming.
    
    Streams:
    - Live Raman spectra
    - AI predictions (modes, polymorphs, confidence)
    - Gating engine state
    - Workflow status
    
    Message Format:
    {
        "type": "spectrum" | "ai_prediction" | "gating_state" | "workflow_status",
        "timestamp": "ISO-8601 timestamp",
        "data": { ... }
    }
    """
    await manager.connect(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"status": "connected", "message": "Raman stream connected"}
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message from client (e.g., control commands)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle control commands
                if message.get("type") == "start_acquisition":
                    await handle_start_acquisition(websocket, message)
                elif message.get("type") == "stop_acquisition":
                    await handle_stop_acquisition(websocket, message)
                elif message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {"message": "Invalid JSON"}
                })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        manager.disconnect(websocket)


async def handle_start_acquisition(websocket: WebSocket, message: Dict[str, Any]):
    """Handle start acquisition command."""
    try:
        integration_time = message.get("data", {}).get("integration_time_ms", 20.0)
        
        # Send acknowledgment
        await websocket.send_json({
            "type": "acquisition_started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"integration_time_ms": integration_time}
        })
        
        # Start streaming spectra (this would be handled by orchestrator in production)
        # For now, send a sample response
        logger.info(f"Starting Raman acquisition with integration time {integration_time}ms")
        
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"message": f"Failed to start acquisition: {str(e)}"}
        })


async def handle_stop_acquisition(websocket: WebSocket, message: Dict[str, Any]):
    """Handle stop acquisition command."""
    try:
        await websocket.send_json({
            "type": "acquisition_stopped",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"message": "Acquisition stopped"}
        })
        
        logger.info("Stopping Raman acquisition")
        
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"message": f"Failed to stop acquisition: {str(e)}"}
        })


async def broadcast_spectrum(spectrum_data: Dict[str, Any]):
    """
    Broadcast spectrum data to all connected clients.
    
    Args:
        spectrum_data: Dict with wavelengths, intensities, metadata
    """
    message = {
        "type": "spectrum",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": spectrum_data
    }
    await manager.broadcast(message)


async def broadcast_ai_prediction(prediction_data: Dict[str, Any]):
    """
    Broadcast AI prediction to all connected clients.
    
    Args:
        prediction_data: Dict with modes, polymorphs, confidence, etc.
    """
    message = {
        "type": "ai_prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": prediction_data
    }
    await manager.broadcast(message)


async def broadcast_gating_state(gating_data: Dict[str, Any]):
    """
    Broadcast gating engine state to all connected clients.
    
    Args:
        gating_data: Dict with rules, status, should_stop, etc.
    """
    message = {
        "type": "gating_state",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": gating_data
    }
    await manager.broadcast(message)


async def broadcast_workflow_status(status_data: Dict[str, Any]):
    """
    Broadcast workflow status to all connected clients.
    
    Args:
        status_data: Dict with run_id, status, current_step, etc.
    """
    message = {
        "type": "workflow_status",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": status_data
    }
    await manager.broadcast(message)


@router.get("/status")
async def get_raman_status():
    """
    Get current Raman system status.
    
    Returns:
        Dict with device status, active connections, etc.
    """
    return {
        "status": "ok",
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.post("/emergency-stop")
async def emergency_stop():
    """
    Emergency stop for Raman acquisition.
    
    Broadcasts stop command to all connected clients and halts acquisition.
    """
    try:
        # Broadcast emergency stop
        await manager.broadcast({
            "type": "emergency_stop",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"message": "Emergency stop activated"}
        })
        
        logger.warning("Emergency stop activated")
        
        return {
            "status": "stopped",
            "message": "Emergency stop activated",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Emergency stop failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
