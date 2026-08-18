# Experiment Template

TEMPLATE ONLY — no experimental values are filled in here.

## Single-Task Manipulation

| Model | Task | Dataset | Nominal SR | Disturbed SR | Mean inference latency | Boundary jump metric | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## Multi-Task Evaluation

| Model | Grasp-only SR | Pick-place SR | Other task SR | Macro Avg | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| TODO | TODO | TODO | TODO | TODO | TODO |

## Smoothness / Action-Chunk Evaluation

| Model | `steps_per_inference` | Mean boundary position jump | Mean boundary rotation jump | p95 tracking error | Success rate | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## MoE / Router Evaluation

| Model | #Experts | Top-k | Router entropy | Task 1 SR | Task 2 SR | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## Required Metadata For Every Run

- Robot and gripper configuration.
- Camera order and camera settings.
- Dataset name and episode count.
- Train/validation split.
- Checkpoint path or release artifact name.
- Number of robot trials.
- Object placement distribution.
- Reset protocol.
- Human intervention rule.
- Failure definition.

