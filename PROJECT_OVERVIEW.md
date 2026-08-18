# ARX5 Diffusion Policy Project

This repository is now organized around the ARX5 Diffusion Policy reproduction.
Original Diffusion Policy demos, tests, media, unused task configs, and SDK build
artifacts were moved to `project_trash/` instead of being deleted.

## Main Entry Points

- `diffusion_policy-main/arx5_collector/`
  - ARX5 robot wrapper
  - SpaceMouse teleoperation
  - three-camera data collection
  - old ARX5 DP-compatible episode writing
- `diffusion_policy-main/arx5_ckpt_loader/`
  - checkpoint loading
  - observation buffering
  - action conversion
  - online policy dry-run / execution skeleton
- `diffusion_policy-main/diffusion_policy/dataset/arx5_image_dataset.py`
  - dataset adapter for collected ARX5 demonstrations
- `diffusion_policy-main/diffusion_policy/config/task/arx5_image.yaml`
  - ARX5 task shape metadata
- `diffusion_policy-main/diffusion_policy/config/train_diffusion_unet_arx5_hybrid_workspace.yaml`
  - ARX5 training config
- `diffusion_policy-main/ARX5_WORKFLOW.md`
  - current canonical ARX5 collection/training/deployment workflow

## Canonical ARX5 DP Data Contract

Camera order follows the older working ARX5 Diffusion Policy project:

```text
camera_0 = USB wrist camera
camera_1 = first RealSense view
camera_2 = second RealSense view
```

Training observations use:

```text
robot0_eef_pos
robot0_eef_rot_axis_angle
robot0_gripper_width
```

Actions use:

```text
[target_x, target_y, target_z, target_rx, target_ry, target_rz, target_gripper_width]
```

The collector still writes extra compatibility/debug fields such as
`robot_eef_pose`, `robot_joint`, `target_gripper`, and `gripper_torque`.

## Runtime SDK Pieces Kept

- `arx5-sdk-main/lib/x86_64/`
- `arx5-sdk-main/python/arx5_interface*.so`
- `arx5-sdk-main/python/arx5_local_config.py`
- `arx5-sdk-main/python/examples/calibrate.py`
- `arx5-sdk-main/python/examples/prepare_gripper_zero.py`
- `arx5-sdk-main/models/X5.urdf`
- `arx5-sdk-main/models/meshes/`

## Common Commands

Activate the environment:

```bash
source ./activate_arx5_env.sh
```

Prepare CAN:

```bash
./start_can1.sh
```

Prepare robot session:

```bash
python -m arx5_collector.scripts.prepare_robot_session \
  --model X5 \
  --interface can1 \
  --expected-gripper open
```

Collect data:

```bash
python -m arx5_collector.scripts.collect_demo \
  --output data_local/arx5_glue_test \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  --usb-config scripts/camera/usb/current.yaml \
  --camera-pump-fps 30 \
  --preview-fps 5
```

Audit collected data:

```bash
python -m arx5_collector.scripts.analyze_recordings \
  data_local/arx5_glue_test
```

Migrate current-format low-dimensional keys to old ARX5 DP keys:

```bash
python -m arx5_collector.scripts.migrate_to_old_dp_schema \
  data_local/arx5_glue_test
```

Mini training smoke test:

```bash
python train.py \
  --config-name=train_diffusion_unet_arx5_hybrid_workspace \
  training.debug=True \
  logging.mode=offline \
  dataloader.batch_size=4 \
  val_dataloader.batch_size=4 \
  dataloader.num_workers=2 \
  val_dataloader.num_workers=2
```

Load checkpoint:

```bash
python -m arx5_ckpt_loader.load_arx5_ckpt --device cpu
```

Online dry-run:

```bash
python -m arx5_ckpt_loader.run_arx5_policy \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  --device cuda:0
```

Only add `--execute` after the dry-run actions look reasonable.
