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
        """Validate step kind (logs warning for unknown kinds but doesn't fail)."""
        import logging
        known_kinds = [
            # Core step types
            "action", "wait", "loop", "condition",
            # Device/hardware steps
            "device_discovery", "initialize", "cleanup", "daq_init", "raman_init",
            "daq_set_ao", "daq_read_ai", "daq_set_do", "daq_read_di", "acquire",
            # Control flow
            "parallel", "sequence", "checkpoint",
            # Human interaction
            "notify", "approval", "input",
            # Data steps
            "log", "export", "import_data", "report_generation",
            # Analysis steps
            "analyze", "transform", "validate"
        ]
        if self.kind not in known_kinds:
            logging.getLogger(__name__).warning(
                f"Unknown step kind '{self.kind}' - will be treated as 'action'"
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
        
        Supports two formats for steps:
        
        1. Dict format (preferred):
            steps:
              start:
                kind: "action"
                params: {...}
                children: []
        
        2. List format (legacy):
            steps:
              - name: "Start"
                type: "action"
                params: {...}
        
        Args:
            yaml_content: YAML workflow definition
            
        Returns:
            WorkflowDefinition instance
        """
        data = yaml.safe_load(yaml_content)
        
        # Parse steps - handle both dict and list formats
        steps = {}
        raw_steps = data.get("steps", {})
        
        if isinstance(raw_steps, dict):
            # Dict format: {step_id: {kind, params, children}}
            for step_id, step_data in raw_steps.items():
                steps[step_id] = WorkflowStep(
                    id=step_id,
                    kind=step_data.get("kind", step_data.get("type", "action")),
                    params=step_data.get("params", {}),
                    children=step_data.get("children", []),
                )
        elif isinstance(raw_steps, list):
            # List format: [{name, type, params}, ...]
            for i, step_data in enumerate(raw_steps):
                step_id = step_data.get("name", f"step_{i}")
                # Sanitize step_id for use as key
                step_id_key = step_id.lower().replace(" ", "_").replace("(", "").replace(")", "")
                steps[step_id_key] = WorkflowStep(
                    id=step_id_key,
                    kind=step_data.get("type", step_data.get("kind", "action")),
                    params=step_data.get("params", {}),
                    children=step_data.get("children", []),
                )
        
        # Determine entry step
        entry_step = data.get("entry_step")
        if not entry_step and steps:
            entry_step = list(steps.keys())[0]  # First step is entry
        
        # Get workflow ID and name
        workflow_id = data.get("id", data.get("name", "unnamed").lower().replace(" ", "_"))
        workflow_name = data.get("name", workflow_id)

        return cls(
            id=workflow_id,
            name=workflow_name,
            steps=steps,
            entry_step=entry_step or "start",
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
