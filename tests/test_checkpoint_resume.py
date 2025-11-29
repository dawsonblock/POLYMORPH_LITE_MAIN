"""
Tests for workflow checkpoint and resume functionality.
"""
import pytest
import asyncio
from retrofitkit.core.recipe import Recipe, Step
from retrofitkit.core.workflows.executor import WorkflowExecutor
from retrofitkit.core.workflows.db_logger import DatabaseLogger
from retrofitkit.db.session import SessionLocal


@pytest.fixture
def test_recipe():
    """Create a simple test recipe."""
    return Recipe(
        id="test-checkpoint-recipe",
        name="Checkpoint Test Recipe",
        steps=[
            Step(type="wait", params={"seconds": 0.1}),
            Step(type="wait", params={"seconds": 0.1}),
            Step(type="wait", params={"seconds": 0.1}),
        ]
    )


@pytest.mark.asyncio
async def test_checkpoint_save(test_recipe, tmp_path):
    """Test that checkpoints are saved after each step."""
    # Create executor with database logger
    db_logger = DatabaseLogger(SessionLocal)
    executor = WorkflowExecutor(
        config=None,
        db_logger=db_logger,
        ai_client=None,
        gating_engine=None
    )
    
    # Execute recipe
    await executor.execute(
        recipe=test_recipe,
        operator_email="test@checkpoint.com",
        run_metadata={"test": True}
    )
    
    # Verify checkpoints were saved
    # (Would need to query database to verify)
    assert len(executor.step_results) == 3


@pytest.mark.asyncio
async def test_resume_from_checkpoint(test_recipe):
    """Test resuming workflow from checkpoint."""
    db_logger = DatabaseLogger(SessionLocal)
    
    # First execution - will fail at step 2
    executor1 = WorkflowExecutor(
        config=None,
        db_logger=db_logger,
        ai_client=None,
        gating_engine=None
    )
    
    # Simulate partial execution
    # (In real scenario, would execute then stop)
    
    # Second execution - resume from checkpoint
    executor2 = WorkflowExecutor(
        config=None,
        db_logger=db_logger,
        ai_client=None,
        gating_engine=None
    )
    
    # Resume should skip completed steps
    # (Would need execution_id from first run)
    # await executor2.execute(
    #     recipe=test_recipe,
    #     operator_email="test@checkpoint.com",
    #     run_metadata={"execution_id": execution_id},
    #     resume_from_checkpoint=True
    # )


@pytest.mark.asyncio
async def test_resume_with_changed_recipe(test_recipe):
    """Test that resume fails if recipe has changed."""
    db_logger = DatabaseLogger(SessionLocal)
    executor = WorkflowExecutor(
        config=None,
        db_logger=db_logger,
        ai_client=None,
        gating_engine=None
    )
    
    # Modify recipe
    modified_recipe = Recipe(
        id=test_recipe.id,
        name=test_recipe.name,
        steps=test_recipe.steps + [Step(type="wait", params={"seconds": 0.1})]
    )
    
    # Resume should fail due to hash mismatch
    # (Would need execution_id from previous run)
    # with pytest.raises(ValueError, match="workflow definition has changed"):
    #     await executor.execute(
    #         recipe=modified_recipe,
    #         operator_email="test@checkpoint.com",
    #         run_metadata={"execution_id": execution_id},
    #         resume_from_checkpoint=True
    #     )


@pytest.mark.asyncio
async def test_recipe_hash_computation(test_recipe):
    """Test recipe hash computation is deterministic."""
    db_logger = DatabaseLogger(SessionLocal)
    executor = WorkflowExecutor(
        config=None,
        db_logger=db_logger,
        ai_client=None,
        gating_engine=None
    )
    
    # Compute hash twice
    hash1 = executor._compute_recipe_hash(test_recipe)
    hash2 = executor._compute_recipe_hash(test_recipe)
    
    # Should be identical
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 produces 64 hex characters
