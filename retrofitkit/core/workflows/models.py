"""
Workflow data models for YAML-based laboratory procedures.

Workflows define multi-step procedures that can execute device actions,
introduce delays, loop, and branch based on conditions.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List
from pathlib import Path
import yaml


@dataclass
class WorkflowStep:
    """
    Single step in a workflow.
    
    Attributes:
        id: Unique step identifier within workflow
        kind: Step type ("action", "wait", "loop", "condition")
        params: Step-specific parameters
        children: List of next step IDs to execute
    """
    id: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate step kind."""
        valid_kinds = ["action", "wait", "loop", "condition"]
        if self.kind not in valid_kinds:
            raise ValueError(
                f"Invalid step kind '{self.kind}'. "
                f"Must be one of: {', '.join(valid_kinds)}"
            )


@dataclass
class WorkflowDefinition:
    """
    Complete workflow definition.
    
    Attributes:
        id: Unique workflow identifier
        name: Human-readable workflow name
        steps: Dict mapping step ID to WorkflowStep
        entry_step: ID of first step to execute
        metadata: Optional workflow metadata
    """
    id: str
    name: str
    steps: Dict[str, WorkflowStep]
    entry_step: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate workflow structure."""
        # Check entry step exists
        if self.entry_step not in self.steps:
            raise ValueError(
                f"Entry step '{self.entry_step}' not found in workflow steps"
            )

        # Check all child references are valid
        for step_id, step in self.steps.items():
            for child_id in step.children:
                if child_id not in self.steps:
                    raise ValueError(
                        f"Step '{step_id}' references unknown child '{child_id}'"
                    )

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "WorkflowDefinition":
        """
        Parse workflow from YAML string.
        
        Args:
            yaml_content: YAML workflow definition
            
        Returns:
            WorkflowDefinition instance
            
        Example YAML:
            id: "test_workflow"
            name: "Test Workflow"
            entry_step: "start"
            steps:
              start:
                kind: "action"
                params:
                  device: "ocean_optics"
                  action: "acquire_spectrum"
                children: []
        """
        data = yaml.safe_load(yaml_content)

        # Parse steps
        steps = {}
        for step_id, step_data in data.get("steps", {}).items():
            steps[step_id] = WorkflowStep(
                id=step_id,
                kind=step_data["kind"],
                params=step_data.get("params", {}),
                children=step_data.get("children", []),
            )

        return cls(
            id=data["id"],
            name=data["name"],
            steps=steps,
            entry_step=data["entry_step"],
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_file(cls, path: Path) -> "WorkflowDefinition":
        """
        Load workflow from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            WorkflowDefinition instance
        """
        with open(path, "r") as f:
            return cls.from_yaml(f.read())

    def to_yaml(self) -> str:
        """
        Convert workflow to YAML string.
        
        Returns:
            YAML representation
        """
        data = {
            "id": self.id,
            "name": self.name,
            "entry_step": self.entry_step,
            "metadata": self.metadata,
            "steps": {
                step.id: {
                    "kind": step.kind,
                    "params": step.params,
                    "children": step.children,
                }
                for step in self.steps.values()
            },
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def to_file(self, path: Path) -> None:
        """
        Save workflow to YAML file.
        
        Args:
            path: Output file path
        """
        with open(path, "w") as f:
            f.write(self.to_yaml())


@dataclass
class WorkflowExecutionResult:
    """
    Result of workflow execution.
    
    Attributes:
        workflow_id: ID of executed workflow
        success: Whether workflow completed successfully
        steps_executed: List of step IDs that were executed
        step_results: Dict mapping step ID to result data
        error: Error message if workflow failed
        duration_seconds: Total execution time
    """
    workflow_id: str
    success: bool
    steps_executed: List[str]
    step_results: Dict[str, Any]
    error: str | None = None
    duration_seconds: float = 0.0
