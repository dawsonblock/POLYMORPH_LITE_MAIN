"""
Tests for Workflow Engine Runner.
"""
import pytest
import asyncio
import uuid
from retrofitkit.core.workflow.runner import (
    WorkflowRunner, WorkflowDefinition, WorkflowStep, StepType, WorkflowStatus
)
from retrofitkit.db.models.workflow import WorkflowVersion
from retrofitkit.db.models.user import User
from retrofitkit.db.models.audit import AuditEvent as AuditLog
from unittest.mock import patch, MagicMock
from contextlib import asynccontextmanager

@pytest.fixture
def runner(db_session):
    # Patch get_db_session in runner module to use our test session
    @asynccontextmanager
    async def mock_get_db_session():
        yield db_session

    with patch("retrofitkit.core.workflow.runner.get_db_session", side_effect=mock_get_db_session):
        yield WorkflowRunner()

@pytest.fixture(autouse=True)
async def seed_user(db_session):
    """Seed test user."""
    user = User(
        email="operator@example.com",
        name="Operator",
        role="operator",
        password_hash=b"hash",
        is_active="true"
    )
    db_session.add(user)
    await db_session.commit()

@pytest.mark.asyncio
async def test_simple_workflow(runner, db_session):
    # Define a simple linear workflow
    defn = WorkflowDefinition(
        id="test_wf",
        name="Test Workflow",
        version="1.0",
        start_step_id="step1",
        steps=[
            WorkflowStep(
                id="step1",
                type=StepType.ACTION,
                name="Step 1",
                action="add_one",
                next_step_id="step2"
            ),
            WorkflowStep(
                id="step2",
                type=StepType.ACTION,
                name="Step 2",
                action="add_one"
            )
        ]
    )
    
    # Create WorkflowVersion in DB
    wf_version = WorkflowVersion(
        id=uuid.uuid4(),
        workflow_name="Test Workflow",
        version="1.0",
        definition=defn.dict(),
        created_by="operator@example.com",
        definition_hash="hash",
        is_active="true"
    )
    db_session.add(wf_version)
    await db_session.commit()
    
    # Register action
    async def add_one(context, params):
        val = context.get("count", 0)
        return {"count": val + 1}
    
    runner.register_action("add_one", add_one)
    
    # Run
    run_id = await runner.start_workflow(str(wf_version.id), context={"count": 0}, operator="operator@example.com")
    
    # Wait for completion (simple poll for test)
    for _ in range(20):
        state = await runner.get_state(run_id)
        if state.status == WorkflowStatus.COMPLETED:
            break
        await asyncio.sleep(0.1)
        
    state = await runner.get_state(run_id)
    assert state.status == WorkflowStatus.COMPLETED
    assert state.context["count"] == 2
    assert len(state.history) == 2

@pytest.mark.asyncio
async def test_conditional_branching(runner, db_session):
    defn = WorkflowDefinition(
        id="cond_wf",
        name="Conditional Workflow",
        version="1.0",
        start_step_id="check_val",
        steps=[
            WorkflowStep(
                id="check_val",
                type=StepType.CONDITION,
                name="Check Value",
                action="noop",
                params={"variable": "go_left"},
                next_step_map={"true": "left", "false": "right"}
            ),
            WorkflowStep(id="left", type=StepType.ACTION, name="Left", action="mark_left"),
            WorkflowStep(id="right", type=StepType.ACTION, name="Right", action="mark_right")
        ]
    )
    
    wf_version = WorkflowVersion(
        id=uuid.uuid4(),
        workflow_name="Conditional Workflow",
        version="1.0",
        definition=defn.dict(),
        created_by="operator@example.com",
        definition_hash="hash",
        is_active="true"
    )
    db_session.add(wf_version)
    await db_session.commit()
    
    runner.register_action("mark_left", lambda ctx, p: {"path": "left"})
    runner.register_action("mark_right", lambda ctx, p: {"path": "right"})
    
    # Test Left
    run_id = await runner.start_workflow(str(wf_version.id), context={"go_left": True}, operator="operator@example.com")
    
    for _ in range(20):
        state = await runner.get_state(run_id)
        if state.status == WorkflowStatus.COMPLETED:
            break
        await asyncio.sleep(0.1)

    state = await runner.get_state(run_id)
    assert state.context.get("path") == "left"
    
    # Test Right
    run_id = await runner.start_workflow(str(wf_version.id), context={"go_left": False}, operator="operator@example.com")
    
    for _ in range(20):
        state = await runner.get_state(run_id)
        if state.status == WorkflowStatus.COMPLETED:
            break
        await asyncio.sleep(0.1)

    state = await runner.get_state(run_id)
    assert state.context.get("path") == "right"

@pytest.mark.asyncio
async def test_human_input(runner, db_session):
    defn = WorkflowDefinition(
        id="human_wf",
        name="Human Input Workflow",
        version="1.0",
        start_step_id="ask_user",
        steps=[
            WorkflowStep(
                id="ask_user",
                type=StepType.HUMAN_INPUT,
                name="Ask User",
                action="noop",
                next_step_id="finish"
            ),
            WorkflowStep(id="finish", type=StepType.ACTION, name="Finish", action="noop")
        ]
    )
    
    wf_version = WorkflowVersion(
        id=uuid.uuid4(),
        workflow_name="Human Input Workflow",
        version="1.0",
        definition=defn.dict(),
        created_by="operator@example.com",
        definition_hash="hash",
        is_active="true"
    )
    db_session.add(wf_version)
    await db_session.commit()
    
    runner.register_action("noop", lambda ctx, p: {})
    
    run_id = await runner.start_workflow(str(wf_version.id), operator="operator@example.com")
    await asyncio.sleep(0.1)
    
    # Should be waiting
    state = await runner.get_state(run_id)
    assert state.status == WorkflowStatus.WAITING_FOR_INPUT
    assert state.current_step_id == "ask_user"
    
    # Submit input
    await runner.submit_input(run_id, "ask_user", {"user_said": "hello"})
    
    for _ in range(20):
        state = await runner.get_state(run_id)
        if state.status == WorkflowStatus.COMPLETED:
            break
        await asyncio.sleep(0.1)
    
    # Should be done
    state = await runner.get_state(run_id)
    assert state.status == WorkflowStatus.COMPLETED
    assert state.context["user_said"] == "hello"
