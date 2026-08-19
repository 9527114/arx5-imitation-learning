# Experiments

This document summarizes experiment branches in the current repository. Status means repository maturity and available code path, not benchmark performance.

## Summary Table

| Branch | Goal | Main implementation | Dataset | Deployment status | Current status |
| --- | --- | --- | --- | --- | --- |
| DP-EEF | Primary ARX5 visuomotor imitation-learning baseline. | `train_dp_eef.sh`, `arx5_image.yaml`, `run_dp_pro.sh` | ARX5 RGB + EEF/gripper schema | Real-robot EEF deployment path exists. | STABLE baseline |
| DP-Joint | Compare joint-space action learning against EEF action learning. | `train_dp_joint.sh`, `arx5_joint_image.yaml`, `run_dp_joint_pro.sh` | Same raw demos, joint action derived from `robot_joint` + `robot_gripper` | Real-robot joint deployment path exists. | EXPERIMENTAL |
| ACT-EEF | ACT-style transformer action chunking with EEF qpos/action. | `arx5_act/train_act.py`, `arx5_act/dataset.py`, `run_act_eef_pro.sh` | ARX5 ACT adapter in EEF mode | Real-robot path exists through `run_act_policy.py`. | EXPERIMENTAL |
| ACT-Joint | ACT-style action chunking closer to the original ACT qpos/action setup. | `arx5_act/train_act.py`, `arx5_act/dataset.py`, `run_act_joint_pro.sh` | ARX5 ACT adapter in joint mode | Real-robot joint path exists through `JointActRobot`. | EXPERIMENTAL |
| DP-CFG | Study previous-action conditioning for chunk consistency. | `train_dp_eef_cfg.sh`, `arx5_dp_cfg/*`, `run_dp_cfg_pro.sh` | DP-EEF schema plus `prev_action` and `prev_action_mask` | Real-robot EEF deployment path exists. | EXPERIMENTAL prototype |
| Point-cloud / DP3 | Future 3D observation policy direction. | No active training/deployment wrapper confirmed in this repo. | N/A | N/A | WIP / not active |
| MoE / MDR | Future routing or multi-policy direction. | No active implementation confirmed in this repo. | N/A | N/A | WIP / not active |

## DP-EEF

**Goal:** Establish the most stable ARX5 baseline using image observations, EEF pose, and gripper width.

**Implementation:**

- `scripts/train_dp_eef.sh`
- `diffusion_policy-main/diffusion_policy/config/task/arx5_image.yaml`
- `diffusion_policy-main/diffusion_policy/config/train_diffusion_unet_arx5_hybrid_workspace.yaml`
- `scripts/run_dp_pro.sh`
- `diffusion_policy-main/arx5_ckpt_loader/run_arx5_policy.py`

**Observed behavior:** Project notes identify DP-EEF as the most stable real-robot baseline. This should be presented as an internal observation, not as a measured public success-rate result.

**Known limitations:** Performance is sensitive to dataset distribution, camera order, lighting, object placement, and deployment timing.

**Next experiment:** Build a fixed evaluation protocol with held-out object positions and record success/failure per trial.

## DP-Joint

**Goal:** Test whether learning joint-space targets better matches the ARX5 control stack than EEF pose targets.

**Implementation:**

- `scripts/train_dp_joint.sh`
- `diffusion_policy-main/diffusion_policy/config/task/arx5_joint_image.yaml`
- `diffusion_policy-main/arx5_collector/scripts/make_joint_dataset.py`
- `diffusion_policy-main/arx5_ckpt_loader/run_arx5_joint_policy.py`
- `diffusion_policy-main/arx5_ckpt_loader/deployment/joint_continuous_executor.py`

**Observed behavior:** Project notes report that joint policies can follow object position and may reproduce operator motion habits, but can also amplify undesirable habits such as premature gripper closing or repeated approach motions.

**Known limitations:** Joint-space action is less directly tied to task-space intent and may encode teleoperation style strongly.

**Next experiment:** Compare DP-EEF and DP-Joint on the same dataset and the same held-out object poses.

## ACT-EEF

**Goal:** Adapt ACT-style transformer action chunking to ARX5 EEF observations/actions.

**Implementation:**

- `arx5_act/dataset.py`
- `arx5_act/train_act.py`
- `arx5_act/run_act_policy.py`
- `scripts/run_act_eef_pro.sh`

**Observed behavior:** ACT-style temporal aggregation is available and can smooth action chunks, but no controlled success-rate comparison is published.

**Known limitations:** EEF-mode ACT is experimental and should not be claimed as the best policy.

**Next experiment:** Run matched comparisons against DP-EEF with identical camera order and reset protocol.

## ACT-Joint

**Goal:** Test ACT in a joint qpos/action setting closer to the original ACT formulation.

**Implementation:**

- `arx5_act/dataset.py`
- `arx5_act/train_act.py`
- `arx5_act/deployment/temporal.py`
- `arx5_act/deployment/joint_robot.py`
- `scripts/run_act_joint_pro.sh`

**Observed behavior:** Current notes suggest ACT-Joint is a meaningful comparison branch. The code has an online joint-mode path, but public success-rate evidence is not curated.

**Known limitations:** Real-robot behavior depends on joint command limits, temporal aggregation, and reset behavior.

**Next experiment:** Evaluate ACT-Joint against DP-Joint and DP-EEF under a fixed protocol.

## DP-CFG

**Goal:** Explore whether previous-action conditioning can improve chunk consistency and reduce boundary discontinuity.

**Implementation:**

- `arx5_dp_cfg/arx5_image_dataset_cfg.py`
- `arx5_dp_cfg/diffusion_unet_hybrid_image_policy_cfg.py`
- `arx5_dp_cfg/deployment/cfg_prev_action.py`
- `arx5_dp_cfg/run_arx5_cfg_policy.py`
- `scripts/train_dp_eef_cfg.sh`
- `scripts/run_dp_cfg_pro.sh`

**Observed behavior:** Previous-action conditioning affects predicted trajectories and deployment smoothness, but multiple project notes warn that strong conditioning can reduce observation-driven correction and produce dominant/fixed trajectory behavior.

**Known limitations:** This branch should be described as an experimental prototype inspired by previous-action conditioning ideas, not as a complete SAIL reproduction or a proven improvement over DP.

**Next experiment:** Warm-start from the strongest DP-EEF checkpoint, sweep `CFG_W`, `PREV_COND_STEPS`, and dropout, and record both task success and trajectory boundary metrics.

## Required Evaluation Before Strong Claims

Before publishing model-performance claims, run matched trials with:

- same dataset split
- same checkpoint selection rule
- same camera order
- same reset protocol
- same object placement set
- fixed failure criteria
- recorded success/failure table
- inference latency and boundary jump summaries
