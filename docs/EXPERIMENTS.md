# Experiments

This table describes experiment branches in the current repository. Status means repository maturity, not benchmark performance.

| Branch | Purpose | Main entry | Input | Output | Current status |
| --- | --- | --- | --- | --- | --- |
| Base DP upstream | Reference implementation from Diffusion Policy. | `diffusion_policy-main/train.py`, upstream configs | Upstream tasks and ARX5-added configs | DP checkpoints | LEGACY |
| DP-EEF | Main ARX5 baseline using end-effector pose actions. | `scripts/train_dp_eef.sh`, `scripts/run_dp_pro.sh` | RGB cameras + EEF pose/gripper | 7D EEF action chunks | WORKING |
| DP-Joint | Compare joint-space action learning against EEF action learning. | `scripts/train_dp_joint.sh`, `scripts/run_dp_joint_pro.sh` | RGB cameras + joint/gripper state | 7D joint action chunks | EXPERIMENTAL |
| DP+CFG | Study previous-action-conditioned DP for smoother chunk transitions. | `scripts/train_dp_eef_cfg.sh`, `scripts/run_dp_cfg_pro.sh` | RGB cameras + EEF state + previous action condition | 7D EEF action chunks | EXPERIMENTAL |
| ACT-EEF | ACT-style transformer baseline with EEF state/action. | `scripts/train_act.sh`, `scripts/run_act_eef_pro.sh` | RGB cameras + EEF qpos | ACT action chunks | EXPERIMENTAL |
| ACT-Joint | ACT-style transformer baseline with joint qpos/action. | `scripts/train_act.sh`, `scripts/run_act_joint_pro.sh` | RGB cameras + joint qpos | ACT joint chunks | EXPERIMENTAL |
| RGB variants | Camera order, USB/RealSense exposure, and three-camera RGB studies. | Collector camera configs, run wrappers | Three RGB streams | Dataset/video variants | WORKING |
| Point-cloud / DP3 | Possible future 3D observation policy. | No active entry found | Unknown | Unknown | UNKNOWN |
| MoE / MDR | Possible future routing or mixture-policy experiments. | No active entry found | Unknown | Unknown | UNKNOWN |
| Flow matching | Possible future policy family. | No active entry found | Unknown | Unknown | UNKNOWN |

## Notes

- `WORKING` means there is a maintained code path that has been used in the real-robot workflow.
- `EXPERIMENTAL` means the code is active but performance, timing, or safety behavior is still under investigation.
- `LEGACY` means retained mainly for upstream compatibility or reference.
- `UNKNOWN` means no active implementation was found during this audit.

