# Workflow Engine Capabilities and Limitations

## Overview

The POLYMORPH-LITE workflow engine provides automated experiment execution with safety controls and audit logging. This document clearly outlines what is supported and what is not.

---

## ✅ Supported Features (v3.0)

### Linear Workflows
- **Sequential execution** of actions and waits
- **Device actions** via driver registry (DAQ, Raman, etc.)
- **Time delays** with configurable wait periods
- **Safety checks** at each step (interlocks, watchdog)
- **Audit logging** of all actions and state changes

### Workflow Definition
- **YAML-based workflows** (`workflows/*.yaml`)
- **Visual workflow builder** (DB-stored graph definitions)
- **Version management** with approval workflow
- **Configuration snapshots** for reproducibility

### Execution Control
- **Single-run execution** with unique run IDs
- **Manual abort** capability
- **Error handling** with rollback semantics
- **Result capture** and storage

### Safety & Compliance
- **E-stop integration** (hardware interlock)
- **Door interlock** monitoring
- **Independent watchdog** for runaway protection
- **Gating rules** (measurement quality thresholds)
- **Hash-chain audit trail** for all steps

---

## ❌ Not Supported (Planned for Future)

### Conditional Logic
**Status**: Not implemented

The workflow engine currently does **not** support:
- `if/then/else` branching based on runtime conditions
- Conditional execution based on measurement results
- Decision points within workflows

**Workaround**: Create separate workflows for different paths and execute manually based on intermediate results.

**Reason for limitation**: Conditional logic requires:
1. Expression parser for runtime evaluation
2. Graph traversal with dynamic branching
3. Complex state management for divergent paths

**Planned for**: v3.1 or v4.0

### Loop Constructs
**Status**: Not implemented

The workflow engine does **not** support:
- `for` loops with iteration counts
- `while` loops with conditional continuation
- Nested loops
- Loop counters or iterators

**Workaround**: Duplicate steps in YAML or use external script to generate repeated steps.

**Reason for limitation**: Loop handling requires:
1. Step repetition tracking
2. Loop variable scoping
3. Break/continue semantics
4. Potential infinite loop protection

**Planned for**: v3.1 or v4.0

### Parallel Execution
**Status**: Not implemented

The workflow engine does **not** support:
- Concurrent execution of multiple branches
- Fork/join patterns
- Parallel device operations
- Async step execution

**Workaround**: Launch multiple workflow executions manually or use external orchestration.

**Reason for limitation**: Parallel execution requires:
1. Thread/process safety for shared resources
2. Synchronization primitives
3. Resource allocation and locking
4. Complex failure recovery

**Planned for**: v4.0

### Subworkflows
**Status**: Not implemented

The workflow engine does **not** support:
- Calling workflows from within workflows
- Hierarchical workflow composition
- Workflow templates with parameters

**Workaround**: Copy common steps into each workflow definition.

**Reason for limitation**: Requires:
1. Call stack management
2. Parameter passing and scoping
3. Nested execution context
4. Recursive workflow validation

**Planned for**: v3.1

---

## Current Workflow Pattern

### Supported Structure
```yaml
name: example_workflow
version: "1.0"
steps:
  - type: action                    # ✅ Supported
    action: set_voltage
    params: {volts: 5.0}
  
  - type: wait                      # ✅ Supported
    duration: 10
  
  - type: action                    # ✅ Supported
    action: trigger_raman
    params: {integration_time: 1.0}
```

### Limited/Experimental
```yaml
  - type: condition                 # ❌ NOT SUPPORTED
    condition: "peak_height > 1000"
    then: ...
    else: ...
  
  - type: loop                      # ❌ NOT SUPPORTED
    iterations: 10
    steps: ...
```

**Attempting to use unsupported step types will raise**:
```python
ValueError: "Loop and condition steps are planned for a future release."
```

This is by design and documented.

---

## Error Handling

### What IS Handled
- **Device failures**: Captured and logged, workflow marked failed
- **Safety violations**: Immediate abort with audit log
- **Timeout errors**: Configurable timeouts with failure state
- **Manual abort**: Clean shutdown with state preservation

### What IS NOT Handled
- **Automatic retry** of failed steps (must re-run entire workflow)
- **Partial resume** from checkpoint (must restart from beginning)
- **Dynamic error recovery** (no conditional fallback paths)

---

## Visual Workflow Builder

The visual workflow builder (`api/workflow_builder.py`) provides:

### ✅ Supported
- Drag-and-drop workflow design
- Node types: Start, End, Acquire, Measure, Delay
- Graph-to-recipe conversion
- Versioning and approval workflow
- Recipe generation from visual graphs

### ⚠️ Limitations
- **Linear graphs only** (no branching or loops in visual editor)
- **Limited node types** (Acquire/Measure/Delay)
- **No runtime parameter inputs** (all params defined in graph)
- **No conditional nodes** (only sequential flow)

The visual builder generates valid `Recipe` objects that can be executed by the orchestrator.

---

## Recommendations

### For Simple SOPs
**Use the workflow engine as-is**. Linear workflows are perfect for:
- Sample preparation protocols
- Calibration procedures
- Quality control checks
- Routine measurements

### For Complex Logic
**Use external scripts** with API calls:
```python
from retrofitkit.api.client import PolymorphClient

client = PolymorphClient()

# Run first workflow
result1 = client.execute_workflow("step1.yaml")

# Make decision based on result
if result1.peak_height > threshold:
    result2 = client.execute_workflow("path_a.yaml")
else:
    result2 = client.execute_workflow("path_b.yaml")
```

This gives you full Python control while maintaining audit trails for each workflow execution.

---

## Version History

- **v3.0.0**: Linear workflows with YAML and visual builder
- **v2.x**: YAML workflows only
- **v3.1 (planned)**: Conditional steps and subworkflows
- **v4.0 (planned)**: Loops and parallel execution

---

## Testing Workflow Limitations

To verify the current limitations:

```bash
# This will fail with ValueError
python -c "
from retrofitkit.core.workflows.engine import WorkflowEngine
workflow_def = {
    'steps': [
        {'type': 'loop', 'iterations': 5}
    ]
}
engine = WorkflowEngine()
engine.execute(workflow_def)  # Raises ValueError
"
```

**Expected error**:
```
ValueError: Loop and condition steps are planned for a future release.
```

This is correct and documented behavior.

---

## Summary

**POLYMORPH-LITE workflow engine (v3.0)**:
- ✅ Excellent for linear, repeatable SOPs
- ✅ Full safety and compliance features
- ❌ Not suitable for complex branching logic
- 🔄 Use external scripts for conditional workflows

Be honest about these limitations in deployment planning.
