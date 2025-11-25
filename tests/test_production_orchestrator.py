import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from retrofitkit.core.orchestrator import Orchestrator
from retrofitkit.core.recipe import Recipe, Step
from retrofitkit.core.app import AppContext, Config, SystemCfg, SecurityCfg, DAQCfg, RamanCfg, GatingCfg, SafetyCfg

@pytest.fixture
def mock_ctx():
    cfg = Config(
        system=SystemCfg(name="test", mode="test", timezone="UTC", data_dir="/tmp/data", logs_dir="/tmp/logs"),
        security=SecurityCfg(password_policy={}, two_person_signoff=False, jwt_exp_minutes=60, rsa_private_key="", rsa_public_key=""),
        daq=DAQCfg(backend="simulator", ni={}, redpitaya={}, simulator={}),
        raman=RamanCfg(provider="simulator", simulator={}, vendor={}),
        gating=GatingCfg(rules=[]),
        safety=SafetyCfg(interlocks={}, watchdog_seconds=1.0)
    )
    return AppContext(cfg)

@pytest.mark.asyncio
async def test_orchestrator_checkpointing(mock_ctx):
    # Mock Redis
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    
    with patch("retrofitkit.core.orchestrator.redis.Redis", return_value=mock_redis):
        orch = Orchestrator(mock_ctx)
        # Mock internal components
        orch.redis = mock_redis
        orch.daq = AsyncMock()
        orch.raman = AsyncMock()
        orch.store = MagicMock()
        orch.store.start_run.return_value = "run_123"
        orch.audit = MagicMock()
        
        # Create recipe
        recipe = Recipe(
            name="test_recipe",
            steps=[
                Step(type="bias_set", params={"volts": 1.0}),
                Step(type="hold", params={"seconds": 0.1})
            ]
        )
        
        # Run recipe
        await orch.execute_recipe(recipe, "test@example.com", simulation=True)
        
        # Verify checkpointing calls
        assert mock_redis.setex.call_count >= 2
        # Verify cleanup
        mock_redis.delete.assert_called_once()

@pytest.mark.asyncio
async def test_orchestrator_resume(mock_ctx):
    # Mock Redis with existing checkpoint
    mock_redis = AsyncMock()
    checkpoint_data = json.dumps({"step": 0, "rid": "run_123", "ts": 1234567890})
    mock_redis.get.return_value = checkpoint_data
    
    with patch("retrofitkit.core.orchestrator.redis.Redis", return_value=mock_redis):
        orch = Orchestrator(mock_ctx)
        orch.redis = mock_redis
        orch.daq = AsyncMock()
        orch.raman = AsyncMock()
        orch.store = MagicMock()
        orch.audit = MagicMock()
        
        recipe = Recipe(
            name="test_recipe",
            steps=[
                Step(type="bias_set", params={"volts": 1.0}), # Step 0 (already done)
                Step(type="hold", params={"seconds": 0.1})    # Step 1 (should run)
            ]
        )
        
        await orch.execute_recipe(recipe, "test@example.com", simulation=True, resume=True)
        
        # Verify we skipped step 0 (bias_set) and ran step 1 (hold)
        # daq.set_voltage shouldn't be called for step 0 if we resumed from step 0 (meaning step 0 finished?)
        # Wait, logic is: start_idx = data["step"] + 1
        # So if checkpoint says step 0, we start at step 1.
        
        # Verify bias_set was NOT called (since it's step 0)
        # Actually, wait. If step 0 is bias_set, and we resume from step 0+1=1, then bias_set is skipped.
        # But hold calls read_ai, so daq is used.
        # Let's check call args.
        
        # Verify set_voltage was NOT called with 1.0 (from step 0)
        # But wait, orchestrator calls set_voltage(0.0) in finally block.
        # So we check if it was called with 1.0.
        
        calls = orch.daq.set_voltage.call_args_list
        # Filter out 0.0 calls
        calls_1v = [c for c in calls if c[0][0] == 1.0]
        assert len(calls_1v) == 0
