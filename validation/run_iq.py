"""
Installation Qualification (IQ) Validator for POLYMORPH v8.0.

Automated check of file structure, dependencies, and environment configuration.
"""

import os
import sys
import logging
import importlib.util
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IQ_Validator")

REQUIRED_DIRS = [
    "retrofitkit/drivers",
    "retrofitkit/api",
    "retrofitkit/core",
    "ui/pages",
    "ai/models",
    "config"
]

REQUIRED_FILES = [
    "retrofitkit/drivers/ocean_optics_driver.py",
    "retrofitkit/drivers/red_pitaya_driver.py",
    "retrofitkit/api/middleware/audit_log.py",
    "retrofitkit/core/workflow/runner.py",
    "ai/train.py",
    "ai/service.py"
]

REQUIRED_MODULES = [
    "fastapi",
    "uvicorn",
    "numpy",
    "pydantic",
    "joblib",
    "sklearn"
]

def check_structure():
    logger.info("Checking directory structure...")
    all_passed = True
    for d in REQUIRED_DIRS:
        if not os.path.exists(d):
            logger.error(f"Missing directory: {d}")
            all_passed = False
        else:
            logger.info(f"OK: {d}")
            
    for f in REQUIRED_FILES:
        if not os.path.exists(f):
            logger.error(f"Missing file: {f}")
            all_passed = False
        else:
            logger.info(f"OK: {f}")
    return all_passed

def check_dependencies():
    logger.info("Checking Python dependencies...")
    all_passed = True
    for mod in REQUIRED_MODULES:
        if importlib.util.find_spec(mod) is None:
            logger.error(f"Missing module: {mod}")
            all_passed = False
        else:
            logger.info(f"OK: {mod}")
    return all_passed

def run_iq():
    print("="*50)
    print("POLYMORPH v8.0 - Installation Qualification (IQ)")
    print("="*50)
    
    struct_ok = check_structure()
    deps_ok = check_dependencies()
    
    if struct_ok and deps_ok:
        print("\n✅ IQ PASSED: System is correctly installed.")
        return True
    else:
        print("\n❌ IQ FAILED: Installation issues detected.")
        return False

if __name__ == "__main__":
    success = run_iq()
    sys.exit(0 if success else 1)
