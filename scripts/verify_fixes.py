import sys
import os
import asyncio
import httpx
from retrofitkit.api.health import readiness_check
# from retrofitkit.drivers.raman.vendor_horiba import HoribaRaman # Moved inside function
from retrofitkit.core.config_loader import get_loader

# Mock config for Horiba
class MockConfig:
    def __init__(self):
        self.raman = type('obj', (object,), {'provider': 'horiba_raman'})
        self.daq = type('obj', (object,), {'backend': 'simulator'})

async def verify_health():
    print("Verifying Health Check...")
    try:
        # We can't easily run the full FastAPI app here without blocking, 
        # but we can call the function directly if we mock the app instance or ensure environment is set.
        # However, readiness_check calls get_app_instance() which might return None if not running.
        # Let's try to simulate the DB check part which was the bug.
        
        from retrofitkit.db.session import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            print("SUCCESS: Database connection check passed (SELECT 1).")
        except Exception as e:
            print(f"FAILURE: Database connection check failed: {e}")
        finally:
            db.close()

    except Exception as e:
        print(f"FAILURE: Health check verification failed: {e}")

async def verify_horiba():
    print("\nVerifying Horiba Driver Safety...")
    try:
        # Mock the module BEFORE importing the driver
        import sys
        from unittest.mock import MagicMock
        sys.modules['horiba_sdk'] = MagicMock()
        
        # Now import the driver
        from retrofitkit.drivers.raman.vendor_horiba import HoribaRaman
        
        config = MockConfig()
        driver = HoribaRaman(config)
        print("Attempting acquire_spectrum with mocked SDK...")
        spectrum = await driver.acquire_spectrum(100)
        
        if spectrum.meta.get("backend") == "simulator":
             print("SUCCESS: Horiba driver fell back to simulation as expected.")
        else:
             print(f"FAILURE: Horiba driver did not fall back to simulation. Metadata: {spectrum.meta}")

    except Exception as e:
        print(f"FAILURE: Horiba verification failed: {e}")

async def main():
    await verify_health()
    await verify_horiba()

if __name__ == "__main__":
    asyncio.run(main())
