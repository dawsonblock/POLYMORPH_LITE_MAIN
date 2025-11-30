"""
Tests for Workflow Engine Runner.
"""
import pytest
import asyncio
from retrofitkit.core.workflow.runner import (
    WorkflowRunner, WorkflowDefinition, WorkflowStep, StepType, WorkflowStatus
)

@pytest.fixture
def runner():
    return WorkflowRunner()

@pytest.mark.asyncio
async def test_simple_workflow(runner):
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
    
    # Register action
    async def add_one(context, params):
        val = context.get("count", 0)
        return {"count": val + 1}
    
    runner.register_action("add_one", add_one)
    runner.load_definition(defn)
    
    # Run
    run_id = await runner.start_workflow("test_wf", context={"count": 0})
    
    # Wait for completion (simple poll for test)
    for _ in range(10):
        state = runner.get_state(run_id)
        if state.status == WorkflowStatus.COMPLETED:
            break
        await asyncio.sleep(0.1)
        
    state = runner.get_state(run_id)
    assert state.status == WorkflowStatus.COMPLETED
    assert state.context["count"] == 2
    assert len(state.history) == 2

@pytest.mark.asyncio
async def test_conditional_branching(runner):
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
    
    runner.register_action("mark_left", lambda ctx, p: {"path": "left"})
    runner.register_action("mark_right", lambda ctx, p: {"path": "right"})
    runner.load_definition(defn)
    
    # Test Left
    run_id = await runner.start_workflow("cond_wf", context={"go_left": True})
    await asyncio.sleep(0.2)
    state = runner.get_state(run_id)
    assert state.context.get("path") == "left"
    
    # Test Right
    run_id = await runner.start_workflow("cond_wf", context={"go_left": False})
    await asyncio.sleep(0.2)
    state = runner.get_state(run_id)
    assert state.context.get("path") == "right"

@pytest.mark.asyncio
async def test_human_input(runner):
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
    
    runner.register_action("noop", lambda ctx, p: {})
    runner.load_definition(defn)
    
    run_id = await runner.start_workflow("human_wf")
    await asyncio.sleep(0.1)
    
    # Should be waiting
    state = runner.get_state(run_id)
    assert state.status == WorkflowStatus.WAITING_FOR_INPUT
    assert state.current_step_id == "ask_user"
    
    # Submit input
    await runner.submit_input(run_id, "ask_user", {"user_said": "hello"})
    await asyncio.sleep(0.1)
    
    # Should be done
    state = runner.get_state(run_id)
    assert state.status == WorkflowStatus.COMPLETED
    assert state.context["user_said"] == "hello"
