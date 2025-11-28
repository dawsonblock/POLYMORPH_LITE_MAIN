import pytest

from retrofitkit.api import workflow_builder
from retrofitkit.core.recipe import Recipe, RecipeStep


class _FakeWorkflowVersion:
    def __init__(self, workflow_name: str, version: int, definition: dict, id_: str = "wf-1"):
        self.workflow_name = workflow_name
        self.version = version
        self.definition = definition
        self.id = id_


def test_graph_to_recipe_linear_sequence():
    graph = {
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "data": {},
                "position": {"x": 0, "y": 0},
            },
            {
                "id": "n-acquire",
                "type": "acquire",
                "data": {"voltage": 2.5, "device_id": "daq"},
                "position": {"x": 100, "y": 0},
            },
            {
                "id": "n-measure",
                "type": "measure",
                "data": {"timeout": 120, "device_id": "raman"},
                "position": {"x": 200, "y": 0},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "n-acquire"},
            {"id": "e2", "source": "n-acquire", "target": "n-measure"},
        ],
    }

    wf = _FakeWorkflowVersion(
        workflow_name="TestWorkflow",
        version=1,
        definition=graph,
    )

    parameters = {
        "n-acquire": {"voltage": 3.3},
        "n-measure": {"timeout": 150},
    }

    recipe = workflow_builder._graph_to_recipe(wf, parameters)  # type: ignore[attr-defined]

    assert isinstance(recipe, Recipe)
    assert recipe.name == "TestWorkflow"
    assert len(recipe.steps) == 2

    first_step = recipe.steps[0]
    assert isinstance(first_step, RecipeStep)
    assert first_step.type == "bias_set"
    assert first_step.params["volts"] == pytest.approx(3.3)
    assert first_step.params["device"] == "daq"

    second_step = recipe.steps[1]
    assert second_step.type == "wait_for_raman"
    assert second_step.params["timeout_s"] == 150
    assert second_step.params["device"] == "raman"

    assert recipe.metadata["version"] == 1
    assert recipe.metadata["visual_graph"] is True


def test_graph_to_recipe_raises_on_empty_graph():
    graph = {"nodes": [], "edges": []}

    wf = _FakeWorkflowVersion(
        workflow_name="EmptyWorkflow",
        version=1,
        definition=graph,
    )

    with pytest.raises(ValueError) as exc:
        workflow_builder._graph_to_recipe(wf, {})  # type: ignore[attr-defined]

    assert "no nodes" in str(exc.value).lower()


def test_graph_to_recipe_no_start_node_uses_first_node():
    graph = {
        "nodes": [
            {
                "id": "n-acquire",
                "type": "acquire",
                "data": {"voltage": 1.2, "device_id": "daq"},
            },
            {
                "id": "n-measure",
                "type": "measure",
                "data": {"timeout": 10, "device_id": "raman"},
            },
        ],
        "edges": [
            {"id": "e1", "source": "n-acquire", "target": "n-measure"},
        ],
    }

    wf = _FakeWorkflowVersion(
        workflow_name="NoStartWorkflow",
        version=1,
        definition=graph,
    )

    recipe = workflow_builder._graph_to_recipe(wf, {})  # type: ignore[attr-defined]

    assert isinstance(recipe, Recipe)
    assert len(recipe.steps) == 2
    assert recipe.steps[0].type == "bias_set"
    assert recipe.steps[1].type == "wait_for_raman"
