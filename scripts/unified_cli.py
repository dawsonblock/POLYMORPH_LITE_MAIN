#!/usr/bin/env python3
"""
Polymorph-Lite Unified CLI.

Main entry point for operators and developers.
"""
import argparse
import asyncio
import logging
import sys
import os
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrofitkit.core.app import AppContext
from retrofitkit.core.driver_router import get_router
from retrofitkit.core.recipe import Recipe
from retrofitkit.core.orchestrator import Orchestrator
from retrofitkit.core.calibration.spectrometer import SpectrometerCalibrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CLI")

async def cmd_run(args, config):
    """Run a workflow recipe."""
    logger.info(f"Loading recipe from {args.recipe}")
    try:
        recipe = Recipe.from_yaml(args.recipe)
    except Exception as e:
        logger.error(f"Failed to load recipe: {e}")
        return

    ctx = AppContext(config)
    orc = Orchestrator(ctx)
    
    logger.info(f"Starting execution of '{recipe.name}'...")
    try:
        run_id = await orc.run(recipe, operator_email=args.email, simulation=args.sim)
        logger.info(f"Run completed successfully. Run ID: {run_id}")
    except Exception as e:
        logger.error(f"Run failed: {e}")
        sys.exit(1)

async def cmd_test_hardware(args, config):
    """Test connection to configured hardware."""
    logger.info("Testing hardware connections...")
    router = get_router()
    
    # Test DAQ
    try:
        daq = router.get_driver("daq", config)
        await daq.connect()
        health = await daq.health()
        logger.info(f"DAQ Status: {health}")
        await daq.disconnect()
    except Exception as e:
        logger.error(f"DAQ Test Failed: {e}")

    # Test Raman
    try:
        raman = router.get_driver("raman", config)
        await raman.connect()
        health = await raman.health()
        logger.info(f"Raman Status: {health}")
        await raman.disconnect()
    except Exception as e:
        logger.error(f"Raman Test Failed: {e}")

async def cmd_calibrate(args, config):
    """Run calibration routines."""
    if args.device == "spectrometer":
        logger.info("Starting Spectrometer Calibration...")
        # In a real scenario, this would interactively ask user to place a calibration source
        # and then acquire a spectrum.
        # For CLI demo, we'll just instantiate the calibrator and show current coeffs.
        cal = SpectrometerCalibrator(config)
        logger.info(f"Current Coefficients: {cal.coeffs}")
        logger.info("To run full calibration, use the interactive wizard (not implemented in CLI yet).")
    else:
        logger.error(f"Unknown device: {args.device}")

async def cmd_audit(args, config):
    """Audit system commands."""
    if args.subcommand == "verify":
        logger.info("Verifying Audit Chain Integrity...")
        try:
            from retrofitkit.compliance.audit import verify_audit_chain
            from retrofitkit.db.session import SessionLocal
            
            session = SessionLocal()
            try:
                result = verify_audit_chain(session)
                if result["valid"]:
                    logger.info(f"SUCCESS: Audit chain is valid. Checked {result['entries_checked']} entries.")
                else:
                    logger.error(f"FAILURE: Audit chain corrupted! Errors: {result['errors']}")
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Audit verification failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Polymorph-Lite Unified CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Run Command
    run_parser = subparsers.add_parser("run", help="Run a workflow recipe")
    run_parser.add_argument("recipe", help="Path to recipe YAML file")
    run_parser.add_argument("--email", default="operator@example.com", help="Operator email")
    run_parser.add_argument("--sim", action="store_true", help="Run in simulation mode")

    # Test Hardware Command
    test_parser = subparsers.add_parser("test", help="Test hardware")
    test_parser.add_argument("target", choices=["hardware"], help="Target to test")

    # Calibrate Command
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate a device")
    cal_parser.add_argument("device", choices=["spectrometer"], help="Device to calibrate")

    # Audit Command
    audit_parser = subparsers.add_parser("audit", help="Audit system commands")
    audit_parser.add_argument("subcommand", choices=["verify"], help="Audit action")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Load Config
    try:
        from retrofitkit.core.config_loader import get_loader
        config = get_loader().load_base().resolve()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Dispatch
    if args.command == "run":
        asyncio.run(cmd_run(args, config))
    elif args.command == "test":
        asyncio.run(cmd_test_hardware(args, config))
    elif args.command == "calibrate":
        asyncio.run(cmd_calibrate(args, config))
    elif args.command == "audit":
        asyncio.run(cmd_audit(args, config))

if __name__ == "__main__":
    main()