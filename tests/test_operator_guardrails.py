import pytest
from fastapi import HTTPException
from retrofitkit.api.workflows import delete_workflow, _workflows
from retrofitkit.core.workflows.models import WorkflowDefinition, WorkflowStep

@pytest.mark.asyncio
async def test_delete_workflow_no_confirm():
    """Verify delete requires confirmation."""
    # Setup dummy workflow
    steps = {"start": WorkflowStep(id="start", kind="action", params={})}
    _workflows["test-wf"] = WorkflowDefinition(id="test-wf", name="Test", steps=steps, entry_step="start")
    
    with pytest.raises(HTTPException) as excinfo:
        await delete_workflow("test-wf", confirm=False)
    
    assert excinfo.value.status_code == 400
    assert "requires confirmation" in excinfo.value.detail
    
    # Verify not deleted
    assert "test-wf" in _workflows

@pytest.mark.asyncio
async def test_delete_workflow_with_confirm():
    """Verify delete works with confirmation."""
    # Setup dummy workflow
    steps = {"start": WorkflowStep(id="start", kind="action", params={})}
    _workflows["test-wf"] = WorkflowDefinition(id="test-wf", name="Test", steps=steps, entry_step="start")
    
    result = await delete_workflow("test-wf", confirm=True)
    
    assert result["message"] == "Workflow 'test-wf' deleted"
    assert "test-wf" not in _workflows
