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
        system=SystemCfg(name="test", environment="testing", timezone="UTC", data_dir="/tmp/data", logs_dir="/tmp/logs"),
        security=SecurityCfg(password_policy={}, two_person_signoff=False, jwt_exp_minutes=60, rsa_private_key="key", rsa_public_key="key"),
        daq=DAQCfg(backend="simulator", ni={}, redpitaya={}, simulator={}),
        raman=RamanCfg(provider="simulator", simulator_peak_nm=532.0),
        gating=GatingCfg(rules=[]),
        safety=SafetyCfg(interlocks={}, watchdog_seconds=1.0)
    )
    return AppContext(cfg)

@pytest.mark.asyncio
@pytest.mark.skip(reason="Checkpointing not implemented in new WorkflowExecutor")
async def test_orchestrator_checkpointing(mock_ctx, db_session):
    # Setup DB
    from retrofitkit.db.models.user import User
    from retrofitkit.db.models.workflow import WorkflowVersion
    import uuid
    
    # Create user
    user = User(email="test@example.com", name="Test User", password_hash=b"hash", role="admin")
    db_session.add(user)
    
    # Create workflow version
    wv = db_session.query(WorkflowVersion).filter_by(workflow_name="test_recipe", version="1.0").first()
    if not wv:
        wv_id = uuid.uuid4()
        wv = WorkflowVersion(
            id=wv_id,
            workflow_name="test_recipe",
            version="1.0",
            definition={},
            definition_hash="hash",
            created_by="test@example.com"
        )
        db_session.add(wv)
        db_session.commit()
    else:
        wv_id = wv.id

    # Mock Redis
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    
    # Mock DriverRouter
    mock_router = MagicMock()
    mock_daq = AsyncMock()
    mock_daq.write_ao.return_value = None
    mock_router.get_driver.return_value = mock_daq

    with patch("retrofitkit.core.orchestrator.redis.Redis", return_value=mock_redis), \
         patch("retrofitkit.core.workflows.executor.get_router", return_value=mock_router):
        orch = Orchestrator(mock_ctx)
        # Mock internal components
        orch.redis = mock_redis
        orch.daq = mock_daq # Also set here for consistency, though executor uses router
        orch.raman = AsyncMock()
        orch.store = MagicMock()
        orch.store.start_run.return_value = "run_123"
        orch.audit = MagicMock()
        
        # Create recipe
        recipe = Recipe(
            id=wv_id,
            name="test_recipe",
            steps=[
                Step(type="daq", params={"action": "write_ao", "channel": 0, "value": 1.0}),
                Step(type="wait", params={"seconds": 0.1})
            ]
        )
        
        # Run recipe
        await orch.execute_recipe(recipe, "test@example.com", simulation=True)
        
        # Verify checkpointing calls
        assert mock_redis.setex.call_count >= 2
        # Verify cleanup
        mock_redis.delete.assert_called_once()

@pytest.mark.asyncio
@pytest.mark.skip(reason="Resume not implemented in new WorkflowExecutor")
async def test_orchestrator_resume(mock_ctx, db_session):
    # Setup DB
    from retrofitkit.db.models.user import User
    from retrofitkit.db.models.workflow import WorkflowVersion
    import uuid
    
    # Create user (check if exists first to avoid dupes if session shared/not rolled back correctly)
    if not db_session.query(User).filter_by(email="test@example.com").first():
        user = User(email="test@example.com", name="Test User", password_hash=b"hash", role="admin")
        db_session.add(user)
    
    # Create workflow version
    wv = db_session.query(WorkflowVersion).filter_by(workflow_name="test_recipe", version="1.0").first()
    if not wv:
        wv_id = uuid.uuid4()
        wv = WorkflowVersion(
            id=wv_id,
            workflow_name="test_recipe",
            version="1.0",
            definition={},
            definition_hash="hash",
            created_by="test@example.com"
        )
        db_session.add(wv)
        db_session.commit()
    else:
        wv_id = wv.id

    # Mock Redis with existing checkpoint
    mock_redis = AsyncMock()
    checkpoint_data = json.dumps({"step": 0, "rid": "run_123", "ts": 1234567890})
    mock_redis.get.return_value = checkpoint_data
    
    # Mock DriverRouter
    mock_router = MagicMock()
    mock_daq = AsyncMock()
    mock_daq.write_ao.return_value = None
    mock_router.get_driver.return_value = mock_daq
    
    with patch("retrofitkit.core.orchestrator.redis.Redis", return_value=mock_redis), \
         patch("retrofitkit.core.workflows.executor.get_router", return_value=mock_router):
        orch = Orchestrator(mock_ctx)
        orch.redis = mock_redis
        orch.daq = mock_daq
        orch.raman = AsyncMock()
        orch.store = MagicMock()
        orch.audit = MagicMock()
        
        recipe = Recipe(
            id=wv_id,
            name="test_recipe",
            steps=[
                Step(type="daq", params={"action": "write_ao", "channel": 0, "value": 1.0}), # Step 0 (already done)
                Step(type="wait", params={"seconds": 0.1})    # Step 1 (should run)
            ]
        )
        
        await orch.execute_recipe(recipe, "test@example.com", simulation=True, resume=True)
        
        # Verify we skipped step 0 (daq) and ran step 1 (wait)
        # Since we mocked the router's driver, we check calls on mock_daq
        calls = mock_daq.write_ao.call_args_list
        # Filter out 0.0 calls (from emergency shutdown or init)
        calls_1v = [c for c in calls if c[0][0] == 0 and c[0][1] == 1.0]
        assert len(calls_1v) == 0
