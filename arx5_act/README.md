# ARX5 ACT

This package adapts the current ARX5 Diffusion Policy collection format for ACT.

Input data is the same folder used by DP:

```text
data_local/<dataset>/
  replay_buffer.zarr/
  videos/<episode>/0.mp4
  videos/<episode>/1.mp4
  videos/<episode>/2.mp4
```

The ACT convention used here is:

```text
qpos   = [x, y, z, rx, ry, rz, gripper_width]
action = [x, y, z, rx, ry, rz, gripper_width]
camera_0 = USB
camera_1 = RealSense 0
camera_2 = RealSense 1
```

## Inspect

```bash
cd /media/star/Elyos_PSSD/ARX5/CY_arx5_dp
source ./activate_arx5_env.sh

python -m arx5_act.inspect_dataset \
  --dataset-path diffusion_policy-main/data_local/arx5_glue_plus_50hz \
  --chunk-size 50
```

Inspect a trained checkpoint:

```bash
python -m arx5_act.inspect_ckpt \
  --ckpt-dir act_outputs/arx5_glue_plus_50hz_act \
  --ckpt-name policy_best.ckpt \
  --device auto
```

## Train

```bash
python -m arx5_act.train_act \
  --dataset-path diffusion_policy-main/data_local/arx5_glue_plus_50hz \
  --ckpt-dir act_outputs/arx5_glue_plus_50hz_act \
  --num-epochs 50 \
  --batch-size 16 \
  --chunk-size 50 \
  --checkpoint-every 10 \
  --device auto
```

By default this does not download pretrained ResNet weights. To use pretrained
weights, add `--pretrained-backbone` after the weight file is already cached or
network access is available.

Outputs:

```text
act_outputs/<run>/
  policy_best.ckpt
  policy_latest.ckpt
  dataset_stats.pkl
  config.json
```

## Deployment

The ACT deployment runner now follows the same operator flow as the DP runner:

```text
human mode: SpaceMouse controls robot
c: switch to policy mode
h: switch back to human mode
r: reset robot to home in human mode
q: quit
```

First run without `--execute` to inspect predicted action chunks:

```bash
../../scripts/run_act_eef_pro.sh
```

Then enable robot execution:

```bash
../../scripts/run_act_eef_pro.sh --execute
```

For ACT checkpoints trained with `--state-mode joint`, use the joint-space
runner. Its action is interpreted as `[q1, q2, q3, q4, q5, q6, gripper_width]`
and is sent through `Arx5JointController`.

```bash
../../scripts/run_act_joint_pro.sh
../../scripts/run_act_joint_pro.sh --execute
```

Do not run a joint checkpoint through the EEF command path. Joint values are in
radians and are not valid Cartesian `[x, y, z, rx, ry, rz]` targets.

Current ARX5-friendly ACT defaults:

```bash
--command-mode traj
--command-latency 0.01
--arm-gain-mode pro
--arm-kp-scale 1
--arm-kd-scale 1
--preview-time 0.1
--tracking-guard
--temporal-agg
--temporal-agg-k 0.01
```

ACT does temporal action aggregation by default. Each policy query predicts a
chunk; overlapping chunks vote on the current action. ARX5 still receives only
one short-preview target at a time so the SDK can handle interpolation and
velocity internally.

When using explicit Linux video devices:

```bash
python -m arx5_act.run_act_policy \
  --ckpt-dir act_outputs/arx5_glue_plus_50hz_act \
  --ckpt-name policy_best.ckpt \
  --model X5 \
  --interface can1 \
  --video-devices 0,6,12 \
  --device cuda
```

The execution layer reuses the safety style already used by DP deployment:

```text
policy-start hold
prepend current action
max position / rotation / gripper step
ACT temporal action aggregation
ARX5 tracking guard
```
