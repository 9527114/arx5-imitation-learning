# Results Audit

Evidence before claims. This file records what can be stated publicly from the current repository and what still requires controlled evaluation.

## Public-Ready Claims

| Claim | Evidence | Suitable wording | Caveat |
| --- | --- | --- | --- |
| Real ARX5 pipeline exists | Collector, dataset, training wrappers, checkpoint loaders, deployment wrappers | "Built a real-robot ARX5 imitation-learning pipeline from demonstration collection to deployment." | Requires ARX5 hardware and local setup. |
| Three-camera RGB + robot-state dataset format exists | `dp_episode_writer.py`, `arx5_image.yaml`, `arx5_joint_image.yaml`, `DATA_SCHEMA.md` | "Constructed episode-based datasets with three RGB camera views, robot state, target actions, timestamps, and per-episode videos." | Local datasets are ignored and not included in Git. |
| DP-EEF baseline exists | `train_dp_eef.sh`, `run_dp_pro.sh`, DP config files | "Implemented and deployed an ARX5-adapted DP-EEF baseline." | No public success rate yet. |
| DP-Joint comparison path exists | `train_dp_joint.sh`, `run_dp_joint_pro.sh`, joint dataset config | "Implemented a joint-action Diffusion Policy comparison path." | Experimental; no superiority claim. |
| ACT adapter exists | `arx5_act/dataset.py`, `train_act.py`, `run_act_policy.py` | "Adapted ACT-style action chunking to ARX5 EEF and joint modes." | Experimental; benchmark not curated. |
| DP-CFG prototype exists | `arx5_dp_cfg/*`, `train_dp_eef_cfg.sh`, `run_dp_cfg_pro.sh` | "Built a previous-action-conditioned DP-CFG prototype for studying chunk continuity." | Not claimed as full SAIL reproduction or task-performance improvement. |
| Deployment timing tools exist | `continuous_executor.py`, `joint_continuous_executor.py`, run wrappers, policy logs | "Implemented timestamped action-chunk scheduling with expired-action filtering and future trajectory handling." | Need summarized metrics for public results table. |
| Dataset audit tools exist | `analyze_recordings.py`, `inspect_training_dataset.py`, `check_dataset_alignment.py` | "Added dataset quality and alignment checks for real-robot recordings." | Does not replace a task success benchmark. |

## Quantitative Logs That Can Be Cited With Context

Local logs and checkpoint configs contain values such as horizon, action steps, parameter counts, training loss, and inference timing. These values are useful for CV context, but they should not be turned into task success claims.

Examples already visible in code/config:

- DP horizon/action settings are defined in `diffusion_policy-main/diffusion_policy/config/train_diffusion_unet_arx5_hybrid_workspace.yaml`.
- ARX5 action/observation shapes are defined in `diffusion_policy-main/diffusion_policy/config/task/arx5_image.yaml` and `arx5_joint_image.yaml`.
- DP-CFG previous-action settings are defined in `arx5_dp_cfg/train_diffusion_unet_arx5_hybrid_workspace_cfg.yaml`.
- ACT chunk/training settings are defined in `arx5_act/train_act.py`.
- Deployment timing parameters are exposed by `scripts/run_dp_pro.sh`, `scripts/run_dp_cfg_pro.sh`, and `scripts/run_act_joint_pro.sh`.

## Not Yet Public-Ready

| Result type | Why not ready | Needed before public claim |
| --- | --- | --- |
| Success rate | No curated, consistent evaluation table was found. | Fixed trial set, object poses, model versions, camera order, reset protocol, and failure criteria. |
| DP vs ACT comparison | Code paths exist, but fair matched evaluation is not consolidated. | Same dataset, same test poses, same robot setup, repeated trials. |
| DP-EEF vs DP-Joint | Qualitative notes exist, but no standardized metric file was found. | Run matched trials and record success/failure. |
| DP-CFG task improvement | Smoothness/continuity diagnostics exist, but no proof of better task success. | Report boundary metrics and task outcomes separately. |
| Smoothness improvement | Deployment logs contain boundary/timing fields, but final summary tables are not curated. | Aggregate mean/median/p95 boundary jumps and tracking errors. |
| Training curves | W&B/offline logs exist locally but are not curated figures. | Export selected curves with exact run metadata. |

## Safe README Guidance

Use implementation-status and observed-behavior language:

- "DP-EEF is the current stable baseline."
- "DP-CFG is experimental."
- "ACT EEF/Joint adapters are implemented."
- "No controlled success-rate benchmark is currently reported."

Avoid:

- "achieves X% success"
- "outperforms baseline"
- "fully reproduces SAIL"
- "DP3 / point cloud is implemented"
- "ACT-Joint is production-ready"
