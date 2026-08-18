# GitHub V1 Repository Audit

Date: 2026-08-10

Scope: static repository scan and minimal public-release cleanup. No robot commands, CAN commands, model training, dataset conversion, or deployment logic changes were performed.

## Current Project Functionality

The repository contains a real-robot ARX5 imitation learning pipeline:

1. SpaceMouse teleoperation drives the ARX5 arm.
2. A USB wrist camera and two RealSense cameras are recorded.
3. Low-dimensional robot state and target action are saved into `replay_buffer.zarr`.
4. Videos are saved per episode under `videos/<episode_id>/<camera_id>.mp4`.
5. DP-compatible dataset adapters load RGB, proprioception, and action chunks.
6. DP-EEF, DP-Joint, DP-CFG, ACT-EEF, and ACT-Joint training wrappers launch experiments.
7. Deployment wrappers load checkpoints, buffer observations, predict action chunks, and stream actions to ARX5 SDK wrappers.

## Core Code

Data collection:

- `diffusion_policy-main/arx5_collector/scripts/collect_demo.py`
- `diffusion_policy-main/arx5_collector/camera/`
- `diffusion_policy-main/arx5_collector/data/dp_episode_writer.py`
- `diffusion_policy-main/arx5_collector/input/spacemouse_teleop.py`
- `diffusion_policy-main/arx5_collector/robot/arx5_robot.py`

Dataset and training:

- `diffusion_policy-main/diffusion_policy/dataset/arx5_image_dataset.py`
- `diffusion_policy-main/diffusion_policy/config/task/arx5_image.yaml`
- `diffusion_policy-main/diffusion_policy/config/task/arx5_joint_image.yaml`
- `diffusion_policy-main/diffusion_policy/config/train_diffusion_unet_arx5_hybrid_workspace.yaml`
- `diffusion_policy-main/diffusion_policy/config/train_diffusion_unet_arx5_joint_hybrid_workspace.yaml`
- `arx5_dp_cfg/arx5_image_dataset_cfg.py`
- `arx5_dp_cfg/train_diffusion_unet_arx5_hybrid_workspace_cfg.yaml`
- `arx5_act/dataset.py`
- `arx5_act/train_act.py`

Deployment:

- `diffusion_policy-main/arx5_ckpt_loader/run_arx5_policy.py`
- `diffusion_policy-main/arx5_ckpt_loader/run_arx5_joint_policy.py`
- `diffusion_policy-main/arx5_ckpt_loader/deployment/`
- `arx5_dp_cfg/run_arx5_cfg_policy.py`
- `arx5_act/run_act_policy.py`
- `scripts/run_dp_pro.sh`
- `scripts/run_dp_cfg_pro.sh`
- `scripts/run_dp_joint_pro.sh`
- `scripts/run_act_eef_pro.sh`
- `scripts/run_act_joint_pro.sh`

Environment and hardware helpers:

- `activate_arx5_env.sh`
- `start_can1.sh`
- `conda_environment_arx5_real.yaml`
- `arx5-sdk-main/`

## Large / Non-GitHub Artifacts Found

These should stay local and are now covered by the root `.gitignore`:

- `diffusion_policy-main/data/outputs/` around 147 GB, including DP checkpoints, Hydra outputs, W&B runs.
- `diffusion_policy-main/data_local/` around 23 GB, including real datasets, replay buffers, videos, zarr caches, policy logs.
- `act_outputs/` around 4.7 GB, including ACT checkpoints.
- `logs/` around 125 MB.
- `project_trash/` around 66 MB, including old project snapshots and build artifacts.
- Many `__pycache__/` directories.
- Individual DP checkpoints around 1.9 GB each.
- ACT checkpoints around 126 MB each.
- Zarr zip caches up to around 1.6 GB each.

No files were deleted.

## Potential Privacy / Portability Issues

Hardcoded local paths were found in:

- `activate_arx5_env.sh`
- `start_can1.sh`
- `scripts/train_act.sh`
- `scripts/train_dp_then_act.sh`
- `scripts/train_three_models.sh`
- `日志.md`
- `正式采集.md`
- `ACT_CACHE_TRAINING.md`
- `ARX5_DIFFUSION_POLICY_ROADMAP.md`
- `diffusion_policy-main/ARX5_WORKFLOW.md`
- `arx5-sdk-main/arx_env.zsh`
- `arx5-sdk-main/diffusion_env.zsh`
- some inherited upstream examples such as `diffusion_policy-main/diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py`

Low-risk script path fixes were applied to:

- `activate_arx5_env.sh`
- `start_can1.sh`
- `scripts/train_act.sh`
- `scripts/train_dp_then_act.sh`
- `scripts/train_three_models.sh`

Historical notes and upstream examples were intentionally not rewritten in v1. They should be reviewed before a polished public release, but changing them now is not necessary for runtime behavior.

No obvious password, API key, SSH key, or private token was found in the static grep scan. The scan did find public URLs, upstream documentation links, local filesystem paths, and example robot/network addresses from upstream Diffusion Policy documentation.

## Open-Source Attribution Issues

The repository includes or adapts several upstream projects:

- `diffusion_policy-main/`: Diffusion Policy code and license are present.
- `act-main/`: ACT code; `act-main/LICENSE` is present.
- `act-main/detr/`: DETR-derived code with Apache 2.0 notice.
- `arx5-sdk-main/`: ARX5 SDK snapshot with its own README/LICENSE.
- `diffusion_policy-main/diffusion_policy/env/kitchen/...`: inherited third-party simulation assets and licenses.

Recommendation: do not add a new top-level license until the combined license status of the vendored upstream code and local modifications is reviewed.

## Files Modified In GitHub V1 Cleanup

- `.gitignore`: added root ignore rules for datasets, checkpoints, videos, caches, logs, W&B, pycache, build artifacts, and scratch folders.
- `activate_arx5_env.sh`: changed fixed project root and conda path into environment-overridable defaults.
- `start_can1.sh`: changed fixed project root into script-relative default.
- `scripts/train_act.sh`: changed fixed project root into script-relative default.
- `scripts/train_dp_then_act.sh`: changed fixed project root into script-relative default.
- `scripts/train_three_models.sh`: changed fixed project root into script-relative default.
- `README.md`: added GitHub-facing project overview, pipeline, commands, data contract, training, deployment, and attribution notes.
- `docs/github_v1_audit.md`: this audit report.

## Files Intentionally Untouched

- Model code and tensor-shape logic.
- Dataset conversion logic.
- Robot control and ARX5 SDK command logic.
- CFG timing/deployment experiments.
- Historical Chinese logs.
- `project_trash/`, because it may still contain recoverable old code.
- Upstream `diffusion_policy-main`, `act-main`, and `arx5-sdk-main` license files.

## Smoke Test Results

Safe checks run during v1 cleanup:

- PASS: `bash -n` on edited shell entry scripts.
- PASS: `python -m py_compile` on the main ARX5 collector, dataset, DP deployment, CFG deployment, and ACT training modules.
- PASS: `python -m arx5_ckpt_loader.load_arx5_ckpt --help`.
- PASS: `python -m arx5_dp_cfg.run_arx5_cfg_policy --help`.
- PASS: `python -m arx5_act.train_act --help`.
- FAIL in current non-hardware shell: `python -m arx5_collector.scripts.collect_demo --help` imports RealSense code, which attempted to initialize `pyrealsense2` and failed with `could not initialize udev monitor`. This was not treated as a code regression because the same entry point depends on local camera/udev access.

Do not run real robot deployment, CAN startup, or camera capture as part of GitHub packaging.

## GitHub V2 TODO

- Replace remaining local paths in documentation with placeholders.
- Add `config.example.yaml` for camera devices, dataset paths, and robot interface.
- Decide whether vendored upstream repositories should remain vendored or become submodules.
- Add a curated small fake dataset or schema-only fixture for CI.
- Add import and config parsing tests.
- Add public demo assets after privacy review.
- Add clearer experiment tables for DP-EEF, DP-Joint, CFG-DP, ACT-EEF, and ACT-Joint.
- Split historical logs from public-facing documentation if needed.
