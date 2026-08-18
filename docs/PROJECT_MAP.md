# Project Map

This map describes the public-facing ARX5 imitation learning code paths. It intentionally excludes `project_trash/`.

| Area | Main files | Role | Status |
| --- | --- | --- | --- |
| Environment bootstrap | `activate_arx5_env.sh`, `start_can1.sh`, `conda_environment_arx5_real.yaml` | Activates the conda environment, ARX5 SDK library path, Python path, and CAN helper setup. | Active |
| Data collection entry | `diffusion_policy-main/arx5_collector/scripts/collect_demo.py` | Records robot state, action targets, timestamps, and three camera videos. | Active |
| Robot wrapper | `diffusion_policy-main/arx5_collector/robot/arx5_robot.py` | Wraps ARX5 SDK state reads, EEF commands, gripper commands, gains, and reset-related behavior. | Active |
| SpaceMouse input | `diffusion_policy-main/arx5_collector/input/spacemouse_teleop.py` | Converts SpaceMouse axes/buttons into teleoperation actions. | Active |
| Camera stack | `diffusion_policy-main/arx5_collector/camera/` | USB, RealSense, and three-camera recording utilities. | Active |
| Episode writer | `diffusion_policy-main/arx5_collector/data/dp_episode_writer.py` | Writes `replay_buffer.zarr` and per-episode videos in DP-compatible layout. | Active |
| Dataset inspection | `diffusion_policy-main/arx5_collector/scripts/analyze_recordings.py`, `check_dataset_alignment.py`, `inspect_training_dataset.py`, `inspect_dp_joint_dataset.py` | Checks lengths, timestamps, videos, action ranges, and joint-action conversion. | Active |
| DP dataset | `diffusion_policy-main/diffusion_policy/dataset/arx5_image_dataset.py` | Loads ARX5 RGB + low-dimensional observations and builds EEF or joint actions. | Active |
| DP configs | `diffusion_policy-main/diffusion_policy/config/task/arx5_image.yaml`, `arx5_joint_image.yaml`, `train_diffusion_unet_arx5_hybrid_workspace.yaml`, `train_diffusion_unet_arx5_joint_hybrid_workspace.yaml` | Hydra configs for ARX5 DP training. | Active |
| DP training wrappers | `scripts/train_dp_eef.sh`, `scripts/train_dp_joint.sh`, `scripts/train_dp_three_separate.sh`, `scripts/train_three_models.sh` | Stable command wrappers around `diffusion_policy-main/train.py`. | Active |
| DP deployment | `diffusion_policy-main/arx5_ckpt_loader/run_arx5_policy.py`, `scripts/run_dp_pro.sh` | Loads DP-EEF checkpoints and runs real-robot EEF action chunks. | Active |
| DP-Joint deployment | `diffusion_policy-main/arx5_ckpt_loader/run_arx5_joint_policy.py`, `scripts/run_dp_joint_pro.sh` | Loads DP-Joint checkpoints and executes joint-space chunks. | Experimental |
| Deployment components | `diffusion_policy-main/arx5_ckpt_loader/deployment/` | Action scheduling, continuous execution, reset, runtime, smoothing, and safety helpers. | Active |
| CFG dataset/model | `arx5_dp_cfg/arx5_image_dataset_cfg.py`, `conditional_unet1d_cfg.py`, `diffusion_unet_hybrid_image_policy_cfg.py` | Adds previous-action conditioning to DP. | Experimental |
| CFG training/deployment | `arx5_dp_cfg/train_diffusion_unet_hybrid_workspace_cfg.py`, `arx5_dp_cfg/run_arx5_cfg_policy.py`, `scripts/train_dp_eef_cfg.sh`, `scripts/run_dp_cfg_pro.sh` | Trains and deploys the previous-action-conditioned DP variant. | Experimental |
| CFG diagnostics | `arx5_dp_cfg/scripts/inspect_prev_action_dataset.py`, `inspect_cfg_condition_effect.py`, `inspect_visual_pipeline.py`, `audit_cfg_pipeline.py` | Checks previous-action windows, visual sensitivity, and CFG condition behavior. | Active diagnostics |
| ACT dataset/cache | `arx5_act/dataset.py`, `arx5_act/build_cache.py`, `arx5_act/build_cache_from_dp.py` | Builds ACT-style samples and optional image cache from ARX5 DP-format data. | Experimental |
| ACT training/deployment | `arx5_act/train_act.py`, `arx5_act/run_act_policy.py`, `scripts/train_act.sh`, `scripts/run_act_eef_pro.sh`, `scripts/run_act_joint_pro.sh` | ACT EEF/joint baselines and robot deployment. | Experimental |
| ACT upstream dependency | `act-main/` | Vendored ACT/DETR code used by `arx5_act`. | Reference dependency |
| DP upstream dependency | `diffusion_policy-main/diffusion_policy/` | Vendored Diffusion Policy code plus ARX5 additions. | Reference plus active additions |
| ARX5 SDK | `arx5-sdk-main/` | SDK snapshot, Python bindings, models, and examples. | Hardware dependency |
| DP3 / point cloud | No active implementation found. | Future or external work. | Unknown |
| MoE / MDR | No active implementation found. | Future or external work. | Unknown |

