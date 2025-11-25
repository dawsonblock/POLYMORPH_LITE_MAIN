# Recipe Model

A recipe is a YAML with ordered `steps`. Available steps:

- `bias_set`: set constant voltage (V)
- `bias_ramp`: ramp voltage from A to B over time
- `hold`: wait in place
- `wait_for_raman`: stream Raman and evaluate gating rules
- `gate_stop`: stop sequence (bias to 0 V)

Example: see `recipes/demo_gate.yaml`.
