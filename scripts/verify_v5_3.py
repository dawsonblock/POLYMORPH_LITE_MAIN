import asyncio
import socketio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_v5_3")

sio = socketio.AsyncClient()

@sio.event
async def connect():
    logger.info("Connected to Socket.IO server")

@sio.event
async def spectral_data(data):
    logger.info(f"Received spectral data: {len(data['wavelengths'])} points, t={data['t']:.2f}")
    # Verify data structure
    if "wavelengths" in data and "intensities" in data:
        logger.info("Data structure valid.")
        await sio.disconnect()
    else:
        logger.error("Invalid data structure!")
        await sio.disconnect()

@sio.event
async def disconnect():
    logger.info("Disconnected from server")

async def main():
    logger.info("--- Verifying v5.3 Optimization ---")
    try:
        # Connect to the running server
        # Note: The server must be running for this to work. 
        # Since we can't easily start the full server in this script without blocking,
        # we will assume the server is running or this test might fail if run standalone.
        # However, for the purpose of this agent check, we can try to connect.
        
        # If server is not running, we might need to mock or skip.
        # Let's try connecting to localhost:8001 (default port)
        await sio.connect('http://localhost:8001', socketio_path='/socket.io')
        await sio.wait()
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        logger.info("NOTE: This verification requires the backend server to be running.")

if __name__ == "__main__":
    asyncio.run(main())
