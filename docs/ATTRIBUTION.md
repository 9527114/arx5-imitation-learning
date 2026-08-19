# Attribution

This repository combines upstream robot-learning code with ARX5-specific integration work. Public descriptions should separate algorithmic sources from project-specific engineering.

## Upstream Work

| Component | Role in this repository |
| --- | --- |
| Diffusion Policy | Base policy architecture, training workspace pattern, image policy, scheduler, dataset/workspace utilities. |
| ACT / DETR | Transformer-style action chunking baseline and temporal aggregation reference. |
| SAIL / previous-action conditioning ideas | Inspiration for the DP-CFG experimental branch. |
| ARX5 SDK | Robot interface, controller bindings, CAN communication, models, and hardware-specific utilities. |
| PyTorch / TorchVision | Model training and inference. |
| Robomimic | Observation encoder configuration and image model utilities used by DP. |
| Diffusers | Diffusion scheduler implementation. |
| RealSense / OpenCV / V4L2 tooling | Camera capture and video handling. |

## Repository-Specific Work

| Area | Implemented here |
| --- | --- |
| ARX5 hardware integration | X5 robot setup, CAN wrapper usage, gripper calibration workflow documentation, reset and runtime wrappers. |
| Demonstration collection | SpaceMouse teleoperation, three-camera recording, robot-state/action/timestamp writing. |
| Dataset schema | DP-compatible `replay_buffer.zarr` plus per-episode videos and old ARX5 camera convention. |
| Dataset adapters | DP-EEF, DP-Joint, DP-CFG, ACT-EEF, and ACT-Joint adapters over the same ARX5 demonstrations. |
| Deployment | Checkpoint loading, observation buffering, action-chunk scheduling, timestamping, expired-action filtering, continuous executors. |
| Diagnostics | Dataset alignment checks, checkpoint inspectors, policy logs, camera tests, SpaceMouse tests, gripper tests. |
| Documentation | Public-facing architecture, data schema, deployment notes, results audit, and release checklist. |

## Claim Boundary

Do not claim that this repository invented Diffusion Policy, ACT, DETR, or SAIL. The defensible claim is that it adapts and integrates these ideas into a real ARX5 robot-learning pipeline with dataset, training, deployment, and debugging tools.
