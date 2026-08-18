# Historical Artifact Audit

Date: 2026-08-10

Scope: static audit only. No files were deleted or moved. Large generated outputs, datasets, checkpoints, logs, and `__pycache__` directories are already covered by `.gitignore` and are not repeated exhaustively here.

Important caveat: many scripts in this project are command-line entry points run with `python -m ...`. Those files are intentionally not imported by other modules, so a simple "unreferenced Python file" scan can produce false positives. In the table below, such files are treated as entry points rather than dead code unless there is a clear replacement.

## KEEP

| file | current purpose | referenced by | replacement | safe to remove? | recommendation |
|---|---|---|---|---|---|
| `diffusion_policy-main/arx5_collector/scripts/collect_demo.py` | Main ARX5 data collection entry point. | README, user workflow, direct `python -m` usage. | None. | No. | KEEP. Core collector. |
| `diffusion_policy-main/arx5_collector/scripts/analyze_recordings.py` | Dataset/video/lowdim quality inspection. | Direct CLI usage in workflow. | None. | No. | KEEP. Useful before training. |
| `diffusion_policy-main/arx5_collector/scripts/check_dataset_alignment.py` | Checks lowdim/video alignment. | Direct CLI usage. | None. | No. | KEEP. Critical for data quality. |
| `diffusion_policy-main/arx5_collector/scripts/inspect_training_dataset.py` | Builds/validates DP training dataset cache. | Direct CLI usage, docs. | None. | No. | KEEP. Training smoke check. |
| `diffusion_policy-main/arx5_collector/scripts/inspect_dp_joint_dataset.py` | Verifies DP-Joint action/state mapping. | Direct CLI usage. | None. | No. | KEEP. Needed for joint-action experiments. |
| `diffusion_policy-main/arx5_collector/scripts/inspect_episode_quality.py` | Episode quality diagnostics. | Direct CLI usage. | `analyze_recordings.py` partially overlaps. | Not yet. | KEEP. Could merge later, but useful now. |
| `diffusion_policy-main/arx5_collector/scripts/filter_episodes.py` | Creates filtered dataset subsets. | Direct CLI usage. | None. | No. | KEEP. Non-destructive data curation helper. |
| `diffusion_policy-main/arx5_collector/scripts/make_joint_dataset.py` | Creates/validates joint-mode dataset copy. | Direct CLI usage from recent workflow. | `action_mode=joint` in `Arx5ImageDataset` handles training-time action construction. | Not yet. | KEEP. Useful as explicit dataset conversion/check tool. |
| `diffusion_policy-main/arx5_collector/scripts/replay_demo.py` | Replays collected trajectories for data sanity checks. | Direct CLI usage. | None. | No. | KEEP. Useful hardware/debug tool; do not run in CI. |
| `diffusion_policy-main/arx5_collector/scripts/prepare_robot_session.py` | Robot startup/gripper zero/session preparation. | Direct CLI usage. | None. | No. | KEEP. Hardware setup helper. |
| `diffusion_policy-main/arx5_collector/scripts/test_robot.py` | Minimal robot state import/start check. | Direct CLI usage. | None. | No. | KEEP. Hardware diagnostic. |
| `diffusion_policy-main/arx5_collector/scripts/test_spacemouse.py` | SpaceMouse input check. | Direct CLI usage. | None. | No. | KEEP. Hardware diagnostic. |
| `diffusion_policy-main/arx5_collector/scripts/test_cameras.py` | Camera chain check. | Direct CLI usage. | None. | No. | KEEP. Hardware diagnostic. |
| `diffusion_policy-main/arx5_collector/scripts/test_gripper.py` | Gripper EEF-control diagnostic. | Direct CLI usage. | None. | No. | KEEP. Hardware diagnostic. |
| `diffusion_policy-main/arx5_collector/scripts/test_gripper_joint.py` | Gripper joint-control diagnostic. | Direct CLI usage. | None. | No. | KEEP. Hardware diagnostic. |
| `diffusion_policy-main/scripts/camera/usb/test_usb_camera.py` | USB camera preview/diagnostic. | Direct CLI usage. | None. | No. | KEEP. Camera diagnostic. |
| `diffusion_policy-main/scripts/camera/usb/tune_usb_camera.py` | USB camera parameter tuning. | `正式采集.md`, direct CLI. | None. | No. | KEEP. Useful before data collection. |
| `diffusion_policy-main/scripts/camera/usb/reset_usb_camera_auto.py` | Reset USB camera to auto exposure/WB. | Direct CLI usage. | None. | No. | KEEP. Useful camera recovery helper. |
| `diffusion_policy-main/scripts/camera/realsense/tune_realsense_exposure.py` | RealSense exposure tuning. | Direct CLI usage. | None. | No. | KEEP. Useful camera setup helper. |
| `diffusion_policy-main/diffusion_policy/dataset/arx5_image_dataset.py` | Main ARX5 DP dataset adapter; supports `action_mode=eef|joint`. | Hydra task configs, dataset inspection scripts. | None. | No. | KEEP. Core training code. |
| `arx5_dp_cfg/arx5_image_dataset_cfg.py` | CFG-specific dataset with previous-action conditioning fields. | CFG Hydra config. | Main `arx5_image_dataset.py` lacks `prev_action`. | No. | KEEP. Active CFG experiment. |
| `arx5_dp_cfg/conditional_unet1d_cfg.py` | CFG-conditioned UNet variant. | `diffusion_unet_hybrid_image_policy_cfg.py`. | Base DP UNet lacks previous-action conditioning. | No. | KEEP. Active CFG experiment. |
| `arx5_dp_cfg/diffusion_unet_hybrid_image_policy_cfg.py` | CFG image policy. | CFG Hydra config, CFG checkpoint loading. | Base DP policy lacks `prev_action` interface. | No. | KEEP. Active CFG experiment. |
| `arx5_dp_cfg/run_arx5_cfg_policy.py` | CFG real-robot deployment. | `scripts/run_dp_cfg_pro.sh`. | `run_arx5_policy.py` is pure DP only. | No. | KEEP. Active deployment. |
| `arx5_dp_cfg/deployment/cfg_prev_action.py` | Builds/validates previous-action conditioning windows. | `run_arx5_cfg_policy.py`. | None. | No. | KEEP. Active CFG timing code. |
| `arx5_dp_cfg/scripts/init_cfg_from_dp.py` | Warm-start CFG checkpoint from DP checkpoint. | `scripts/train_mini_cfg_variants.sh`. | Manual checkpoint surgery. | No. | KEEP. Useful for CFG experiments. |
| `arx5_dp_cfg/scripts/inspect_cfg_condition_effect.py` | Measures CFG condition effect. | `scripts/inspect_cfg_condition.sh`. | None. | No. | KEEP. Active debug/eval tool. |
| `arx5_dp_cfg/scripts/inspect_visual_pipeline.py` | Checks camera/order/visual sensitivity. | Direct CLI usage during CFG audit. | None. | No. | KEEP. Useful for regression checks. |
| `arx5_dp_cfg/scripts/audit_cfg_pipeline.py` | CFG pipeline audit. | Direct CLI usage. | `inspect_visual_pipeline.py` partially overlaps. | Not yet. | KEEP. Could merge later. |
| `arx5_dp_cfg/scripts/inspect_prev_action_dataset.py` | Checks prev-action dataset windows. | Direct CLI usage. | None. | No. | KEEP. Important CFG data validation. |
| `arx5_act/dataset.py` | ACT dataset adapter for ARX5 DP-format data. | `train_act.py`, cache scripts, inspectors. | None. | No. | KEEP. Core ACT code. |
| `arx5_act/train_act.py` | ACT training entry point. | `scripts/train_act.sh`, train-chain scripts. | None. | No. | KEEP. Core ACT code. |
| `arx5_act/run_act_policy.py` | ACT real-robot deployment entry point. | `scripts/run_act_eef_pro.sh`, `scripts/run_act_joint_pro.sh`. | None. | No. | KEEP. Core ACT deployment. |
| `arx5_act/deployment/temporal.py` | ACT temporal action aggregation. | `run_act_policy.py`. | None. | No. | KEEP. Active ACT smoothing. |
| `arx5_act/deployment/joint_robot.py` | Joint-space ARX5 robot adapter used by ACT and DP-Joint. | `run_act_policy.py`, `run_arx5_joint_policy.py`. | None. | No. | KEEP. Active hardware adapter. |
| `diffusion_policy-main/arx5_ckpt_loader/deployment/action_postprocess.py` | Action chunk smoothing/clamping/deadband helpers. | DP, CFG, ACT deployment. | None. | No. | KEEP. Shared deployment code. |
| `diffusion_policy-main/arx5_ckpt_loader/deployment/action_scheduler.py` | Action timestamp scheduling helpers. | DP and DP-Joint deployment. | None. | No. | KEEP. Shared deployment code. |
| `diffusion_policy-main/arx5_ckpt_loader/deployment/continuous_executor.py` | High-rate continuous EEF waypoint executor. | DP and CFG deployment. | None. | No. | KEEP. Core deployment code. |
| `diffusion_policy-main/arx5_ckpt_loader/deployment/joint_continuous_executor.py` | High-rate joint waypoint executor. | DP-Joint deployment. | None. | No. | KEEP. Core deployment code. |
| `diffusion_policy-main/arx5_ckpt_loader/deployment/reset.py` | Checked reset helpers. | DP, CFG, ACT/Joint deployment. | None. | No. | KEEP. Core safety/operation helper. |
| `diffusion_policy-main/arx5_ckpt_loader/trajectory_buffer.py` | Buffered action trajectory and old/new chunk fusion. | DP and CFG deployment. | None. | No. | KEEP. Core timing code. |
| `scripts/run_dp_pro.sh` | Stable DP-EEF deployment wrapper. | Direct CLI usage. | None. | No. | KEEP. Main public entry. |
| `scripts/run_dp_cfg_pro.sh` | Stable CFG-DP deployment wrapper. | Direct CLI usage. | None. | No. | KEEP. Main CFG entry. |
| `scripts/run_dp_joint_pro.sh` | Stable DP-Joint deployment wrapper. | Direct CLI usage. | None. | No. | KEEP. Main joint entry. |
| `scripts/run_act_eef_pro.sh` | ACT-EEF deployment wrapper. | Direct CLI usage. | None. | No. | KEEP. ACT baseline entry. |
| `scripts/run_act_joint_pro.sh` | ACT-Joint deployment wrapper. | Direct CLI usage. | None. | No. | KEEP. ACT baseline entry. |

## MOVE_TO_ARCHIVE

These are already isolated under `project_trash/`. They should remain excluded from GitHub v1 by `.gitignore`. No additional move was performed.

| file | current purpose | referenced by | replacement | safe to remove? | recommendation |
|---|---|---|---|---|---|
| `project_trash/original_dp_configs/**` | Original DP task/train configs parked during ARX5 cleanup. | None in active code. | ARX5 configs under `diffusion_policy-main/diffusion_policy/config/`. | From active runtime: yes. From recovery perspective: not yet. | MOVE_TO_ARCHIVE already done. Keep local, do not publish. |
| `project_trash/original_dp_internal_scripts/**` | Original DP conversion/metrics scripts. | None in active ARX5 code. | ARX5 collector/data scripts. | From active runtime: yes. | MOVE_TO_ARCHIVE already done. Keep local until confident. |
| `project_trash/original_dp_tests/**` | Original DP tests. | None in active ARX5 code. | Future ARX5-specific tests should live separately. | From active runtime: yes. | MOVE_TO_ARCHIVE already done. Do not publish in v1 unless upstream tests are restored intentionally. |
| `project_trash/original_dp_top_level/**` | Original DP top-level demo/eval/ray files. | None in active ARX5 code. | ARX5 train/deploy scripts. | From active runtime: yes. | MOVE_TO_ARCHIVE already done. Keep local. |
| `project_trash/original_dp_media/**` | Original DP media. | None in active ARX5 code. | Future curated ARX5 assets. | Yes for ARX5 runtime. | MOVE_TO_ARCHIVE already done. Do not publish unless attribution/media rights are reviewed. |
| `project_trash/sdk_build_artifacts/**` | CMake/build outputs and compiled test artifacts. | None in active code. | Rebuild from `arx5-sdk-main` if needed. | Yes for runtime after current SDK libs are verified. | MOVE_TO_ARCHIVE already done. Good candidate for deletion after backup. |
| `project_trash/sdk_original_examples/**` | Original SDK C++ examples and editor config. | None in active Python workflow. | `arx5-sdk-main/python/examples/`. | Likely yes. | MOVE_TO_ARCHIVE already done. Keep local until SDK packaging is settled. |
| `project_trash/sdk_packaging/**` | SDK wheel/packaging files. | None in active ARX5 DP workflow. | Direct SDK checkout/install. | Unknown for future packaging. | MOVE_TO_ARCHIVE already done. Keep local if wheel packaging may return. |

## REMOVE_LATER

These look removable only after you make a backup or confirm they are not part of the public story. No deletion was performed.

| file | current purpose | referenced by | replacement | safe to remove? | recommendation |
|---|---|---|---|---|---|
| `project_trash/sdk_build_artifacts/**` | Build cache, object files, CMake state, generated binaries. | None. | Rebuild when needed. | Probably yes. | REMOVE_LATER after external backup; keep ignored for now. |
| `project_trash/original_dp_tests/tests/__pycache__` | Python bytecode inside archived upstream tests. | None. | Source test files. | Yes. | REMOVE_LATER. Low value. |
| `arx5-sdk-main/python/examples/test_bimanual.py` and other SDK `test_*.py` examples | Upstream SDK diagnostic examples. | None in ARX5 DP wrappers. | ARX5 collector test scripts for project-specific checks. | Not from SDK perspective. | REMOVE_LATER only if publishing a slim repo without SDK examples. Otherwise KEEP as upstream SDK content. |
| `arx5-sdk-main/models/X7_left_new.urdf`, `arx5-sdk-main/models/X7_right_new.urdf` | X7 model variants, not ARX5 X5 task runtime. | Not referenced in current X5 workflow. | `arx5-sdk-main/models/X5.urdf` for current robot. | Unknown if future X7/bimanual work matters. | REMOVE_LATER only if repo is scoped strictly to X5. |
| `arx5-sdk-main/arx_env.zsh`, `arx5-sdk-main/diffusion_env.zsh` | Old local environment scripts with hardcoded paths. | None in current wrappers. | Root `activate_arx5_env.sh`. | Probably yes for public v1. | REMOVE_LATER or archive after SDK submodule/licensing decision. |

## UNKNOWN

These are not clearly dead. Keep them until their usage is clarified.

| file | current purpose | referenced by | replacement | safe to remove? | recommendation |
|---|---|---|---|---|---|
| `diffusion_policy-main/arx5_collector/scripts/collect_demo_pro.py` | Wrapper that runs `collect_demo.py` then optionally builds DP/ACT caches. | No direct active wrapper found; internally calls `collect_demo` and `arx5_act.build_cache`. | Manual sequence: collect, inspect DP dataset, build ACT cache. | Unknown. | UNKNOWN. Keep for now; consider documenting or removing in v2 if never used. |
| `diffusion_policy-main/arx5_collector/scripts/prepare_data_pro.py` | Dataset preparation/cache building helper. | No top-level wrapper found; calls ACT cache builder. | Manual inspect/build-cache commands. | Unknown. | UNKNOWN. Keep until data-pro workflow is stable. |
| `diffusion_policy-main/arx5_collector/scripts/merge_data_pro.py` | Dataset merge/helper workflow with cache-building hooks. | No top-level wrapper found. | Manual merge/copy/filter scripts. | Unknown. | UNKNOWN. Keep until all useful datasets are frozen. |
| `diffusion_policy-main/arx5_collector/scripts/migrate_to_old_dp_schema.py` | Migrates newer collector keys to old DP schema. | `PROJECT_OVERVIEW.md`, `ARX5_WORKFLOW.md`, logs. | New collector may already write compatible keys. | Not yet. | UNKNOWN/KEEP. Useful for old datasets. Move to archive only after all datasets are migrated. |
| `diffusion_policy-main/arx5_collector/scripts/reorder_videos_to_old_dp_schema.py` | Reorders videos to old DP camera convention. | `ARX5_WORKFLOW.md`, logs. | Correct camera order during collection/deployment. | Not yet. | UNKNOWN/KEEP. Useful for old datasets/camera-order recovery. |
| `arx5_act/build_cache.py` | Builds ACT image cache from raw DP-format dataset. | Docs, `collect_demo_pro.py`, `prepare_data_pro.py`, `merge_data_pro.py`. | `arx5_act/build_cache_from_dp.py` if DP cache already exists. | No. | KEEP. Main ACT cache path from raw videos. |
| `arx5_act/build_cache_from_dp.py` | Builds ACT cache from an existing DP zarr cache. | Mostly docs/logs; no stable shell wrapper found. | `arx5_act/build_cache.py`. | Unknown. | UNKNOWN. Keep because it is much faster when DP cache exists. Consider documenting better. |
| `arx5_act/inspect_dataset.py` | ACT dataset smoke/shape inspection. | `arx5_act/README.md`, logs. | None. | No. | KEEP. Useful validation tool. |
| `arx5_act/inspect_ckpt.py` | ACT checkpoint inspection. | `arx5_act/README.md`. | None. | No. | KEEP. Useful validation tool. |
| `diffusion_policy-main/arx5_ckpt_loader/deployment/arx_runtime.py` | Runtime scheduler used by ACT deployment; not used by current DP EEF/CFG paths. | `arx5_act/run_act_policy.py`. | None. | No while ACT is kept. | KEEP. |
| `diffusion_policy-main/arx5_ckpt_loader/deployment/action_scheduler.py` | Timestamp helper for DP/Joint deployment; overlaps conceptually with continuous executor. | DP and DP-Joint deployment. | None. | No. | KEEP. Shared logic. |
| `act-main/*` top-level sim scripts | Upstream ACT simulation/training scripts. | Local `arx5_act` imports ACT policy/modules; some top-level scripts not used. | `arx5_act/*` wrappers. | Unknown due upstream attribution/import expectations. | UNKNOWN. Keep vendored ACT snapshot intact for v1. |
| `diffusion_policy-main/diffusion_policy/dataset/*` non-ARX5 datasets | Upstream DP datasets for pusht/robomimic/kitchen/etc. | Upstream configs/workspaces may reference them; ARX5 configs do not. | `arx5_image_dataset.py` for this project. | Unknown due vendored DP integrity. | UNKNOWN. Keep upstream DP snapshot intact for v1. |
| `diffusion_policy-main/diffusion_policy/policy/*` non-DP-ARX5 policies | Upstream policies and baselines. | Upstream configs/workspaces. | DP-UNet ARX5 config uses `diffusion_unet_hybrid_image_policy.py`. | Unknown. | UNKNOWN. Keep upstream DP snapshot intact for v1. |
| `diffusion_policy-main/diffusion_policy/env/**` simulation environments | Upstream DP simulation code. | Upstream runners/tests/configs. | Not used for real ARX5 pipeline. | Unknown due upstream integrity. | UNKNOWN. Move/archive only in a future slim-repo pass. |

## Repeated Implementations That Are Intentional

| file | current purpose | referenced by | replacement | safe to remove? | recommendation |
|---|---|---|---|---|---|
| `diffusion_policy-main/diffusion_policy/dataset/arx5_image_dataset.py` vs `arx5_dp_cfg/arx5_image_dataset_cfg.py` | Base DP dataset vs CFG dataset with `prev_action`. | DP configs vs CFG config. | None. | No. | KEEP both. Difference is functional, not accidental duplication. |
| `diffusion_policy-main/diffusion_policy/model/diffusion/conditional_unet1d.py` vs `arx5_dp_cfg/conditional_unet1d_cfg.py` | Base UNet vs CFG-conditioned UNet. | Base DP policy vs CFG policy. | None. | No. | KEEP both. |
| `diffusion_policy-main/arx5_ckpt_loader/run_arx5_policy.py` vs `arx5_dp_cfg/run_arx5_cfg_policy.py` | Pure DP deployment vs CFG deployment. | `run_dp_pro.sh` vs `run_dp_cfg_pro.sh`. | None. | No. | KEEP both until CFG stabilizes. |
| `diffusion_policy-main/arx5_ckpt_loader/run_arx5_policy.py` vs `diffusion_policy-main/arx5_ckpt_loader/run_arx5_joint_policy.py` | EEF action deployment vs joint action deployment. | `run_dp_pro.sh` vs `run_dp_joint_pro.sh`. | None. | No. | KEEP both. |
| `arx5_act/build_cache.py` vs `arx5_act/build_cache_from_dp.py` | Build ACT cache from raw dataset vs from existing DP cache. | Direct CLI/docs. | One can replace the other only depending on workflow. | Not yet. | KEEP/UNKNOWN. Document difference before pruning. |

## Summary

- No active ARX5 training/deployment file was proven safe to delete.
- Most historical artifacts are already isolated under `project_trash/` and ignored by Git.
- The best near-term cleanup is not deletion, but documentation: mark `collect_demo.py`, `train_dp_eef.sh`, `run_dp_pro.sh`, `train_dp_eef_cfg.sh`, `run_dp_cfg_pro.sh`, and ACT wrappers as canonical entry points.
- The only clear future deletion candidates are build artifacts and pycache under `project_trash/`.
- Upstream DP/ACT/SDK snapshots should be kept intact for GitHub v1 unless you intentionally split them into submodules or a slim repository later.
