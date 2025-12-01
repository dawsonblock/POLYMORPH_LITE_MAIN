"""
Operational Qualification (OQ) Validator for POLYMORPH v8.0.

Automated functional testing of hardware drivers and workflow engine.
"""

import sys
import logging
import asyncio
from retrofitkit.drivers.ocean_optics_driver import OceanOpticsDriver
from retrofitkit.drivers.red_pitaya_driver import RedPitayaDriver
from retrofitkit.core.workflow.runner import WorkflowRunner, WorkflowDefinition, WorkflowStep, StepType, WorkflowStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OQ_Validator")

def test_hardware_drivers():
    logger.info("Testing Hardware Drivers (Simulation Mode)...")
    try:
        # Ocean Optics
        spec = OceanOpticsDriver(simulate=True)
        wl, inten = spec.acquire_spectrum()
        if len(wl) == 2048 and len(inten) == 2048:
            logger.info("OK: Ocean Optics Driver (Acquisition)")
        else:
            logger.error("FAIL: Ocean Optics Driver (Invalid Data Shape)")
            return False
        spec.disconnect()

        # Red Pitaya
        daq = RedPitayaDriver(host="127.0.0.1", simulate=True)
        wave = daq.get_waveform()
        if len(wave) > 0:
             logger.info("OK: Red Pitaya Driver (Waveform)")
        else:
             logger.error("FAIL: Red Pitaya Driver (Empty Waveform)")
             return False
        daq.disconnect()
        
        return True
    except Exception as e:
        logger.error(f"FAIL: Hardware Exception: {e}")
        return False

async def test_workflow_engine():
    logger.info("Testing Workflow Engine...")
    runner = WorkflowRunner()
    
    # Define simple test workflow
    defn = WorkflowDefinition(
        id="oq_test",
        name="OQ Test Workflow",
        version="1.0",
        start_step_id="step1",
        steps=[
            WorkflowStep(id="step1", type=StepType.ACTION, name="Step 1", action="noop")
        ]
    )
    runner.register_action("noop", lambda ctx, p: {})
    runner.load_definition(defn)
    
    try:
        run_id = await runner.start_workflow("oq_test")
        await asyncio.sleep(0.5)
        state = runner.get_state(run_id)
        
        if state.status == WorkflowStatus.COMPLETED:
            logger.info("OK: Workflow Engine (Execution)")
            return True
        else:
            logger.error(f"FAIL: Workflow Engine (Status: {state.status})")
            return False
    except Exception as e:
        logger.error(f"FAIL: Workflow Exception: {e}")
        return False

async def run_oq():
    print("="*50)
    print("POLYMORPH v8.0 - Operational Qualification (OQ)")
    print("="*50)
    
    hw_ok = test_hardware_drivers()
    wf_ok = await test_workflow_engine()
    
    if hw_ok and wf_ok:
        print("\n✅ OQ PASSED: System functions correctly.")
        return True
    else:
        print("\n❌ OQ FAILED: Functional issues detected.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_oq())
    sys.exit(0 if success else 1)
