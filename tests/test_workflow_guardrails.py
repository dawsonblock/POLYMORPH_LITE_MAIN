import pytest
from pydantic import ValidationError
from retrofitkit.core.recipe import Recipe, Step

def test_workflow_max_length():
    """Verify workflow rejects > 50 steps."""
    steps = [Step(type="action", params={}) for _ in range(51)]
    
    with pytest.raises(ValidationError) as excinfo:
        Recipe(name="Too Long", steps=steps)
    
    assert "Workflow exceeds maximum length of 50 steps" in str(excinfo.value)

def test_workflow_unsupported_types():
    """Verify workflow rejects loop/parallel steps."""
    steps = [
        Step(type="action", params={}),
        Step(type="loop", params={})
    ]
    
    with pytest.raises(ValidationError) as excinfo:
        Recipe(name="Bad Type", steps=steps)
        
    assert "Step type 'loop' is not supported" in str(excinfo.value)

def test_workflow_valid():
    """Verify valid workflow passes."""
    steps = [Step(type="action", params={}) for _ in range(10)]
    recipe = Recipe(name="Good", steps=steps)
    assert len(recipe.steps) == 10
