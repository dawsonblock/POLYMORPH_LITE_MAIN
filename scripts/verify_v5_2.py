import asyncio
import logging
import sys
from retrofitkit.core.ai.models.raman_cnn_v2 import RamanCNNv2
from retrofitkit.drivers.production_base import ProductionHardwareDriver
from retrofitkit.api.auth import router as auth_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_v5_2")

async def verify_ai():
    logger.info("--- Verifying AI Upgrade ---")
    model = RamanCNNv2()
    spectrum = {
        "wavelengths": [500 + i for i in range(100)],
        "intensities": [100 + i for i in range(100)],
        "meta": {"temperature": 25.0}
    }
    result = model.predict(spectrum)
    logger.info(f"Prediction Result: {result}")
    if "model_version" in result:
        logger.info(f"Model Version: {result['model_version']}")
    else:
        logger.error("Model version missing!")

async def verify_driver():
    logger.info("--- Verifying Driver Dry-Run ---")
    class TestDriver(ProductionHardwareDriver):
        pass
    
    driver = TestDriver()
    driver.set_dry_run(True)
    logged = driver.log_dry_run("TEST_COMMAND", {"param": 1})
    
    if logged:
        logger.info("Dry-run logging successful.")
    else:
        logger.error("Dry-run logging failed!")

async def verify_auth():
    logger.info("--- Verifying Auth Skeleton ---")
    routes = [route.path for route in auth_router.routes]
    logger.info(f"Auth Routes: {routes}")
    
    if "/login/oidc" in routes and "/callback/oidc" in routes:
        logger.info("OIDC endpoints found.")
    else:
        logger.error("OIDC endpoints missing!")

async def main():
    await verify_ai()
    await verify_driver()
    await verify_auth()

if __name__ == "__main__":
    asyncio.run(main())
