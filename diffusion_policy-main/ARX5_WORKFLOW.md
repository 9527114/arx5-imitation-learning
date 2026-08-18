# ARX5 Diffusion Policy Workflow

This project keeps the ARX5 pipeline close to the older working
`try_/arx5_diffusion` layout.

## Canonical Data Schema

Camera order:

```text
camera_0 = USB wrist camera
camera_1 = first RealSense view
camera_2 = second RealSense view
```

Low-dimensional observations:

```text
robot0_eef_pos              shape (3,)
robot0_eef_rot_axis_angle   shape (3,)
robot0_gripper_width        shape (1,)
```

Action:

```text
action = [target_x, target_y, target_z, target_rx, target_ry, target_rz, target_gripper_width]
shape = (7,)
```

The collector also writes compatibility keys such as `robot_eef_pose` and
`robot_gripper`, but training should use the `robot0_*` keys above.

## Prepare Robot Session

From the repository root:

```bash
cd /media/star/Elyos_PSSD/ARX5/CY_arx5_dp
source ./activate_arx5_env.sh
cd diffusion_policy-main

python -m arx5_collector.scripts.prepare_robot_session \
  --model X5 \
  --interface can1 \
  --expected-gripper open
```

If `can1` is missing after a reboot, start CAN first with your `start_can1.sh`.

## Collect Demonstrations

The collector now records videos in the old camera order:

```text
videos/<episode>/0.mp4 = USB
videos/<episode>/1.mp4 = RealSense 0
videos/<episode>/2.mp4 = RealSense 1
```

Example:

```bash
python -m arx5_collector.scripts.collect_demo \
  --output data_local/arx5_glue_test \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  --usb-width 640 \
  --usb-height 480 \
  --camera-pump-fps 30 \
  --preview-fps 5
```

Collection separates control and data rates:

```text
--frequency 100       # SpaceMouse/robot control loop
--data-frequency 20   # replay_buffer.zarr state/action write rate
```

This means new datasets are already close to the older ARX5 DP training rate
while keeping teleoperation smooth.

Keyboard controls:

```text
c = start episode
s = save current episode
d = delete current episode and reset home
r = reset home
q = save current episode, reset home, quit
Backspace = drop last saved episode
Space = stage marker
```

Keep each useful episode compact. A good first target is roughly 200-500 frames
per episode, with little idle time before or after the task.

## Migrate Older Current-Format Data

Datasets collected before the schema cleanup may only contain
`robot_eef_pose` and `robot_gripper`. Add old DP keys with:

```bash
python -m arx5_collector.scripts.migrate_to_old_dp_schema \
  data_local/arx5_glue_test
```

Use `--overwrite` only when you intentionally want to regenerate those arrays.

Older current-format datasets may also have the previous video order:

```text
0.mp4 = RealSense 0
1.mp4 = RealSense 1
2.mp4 = USB
```

Preview the rename first:

```bash
python -m arx5_collector.scripts.reorder_videos_to_old_dp_schema \
  data_local/arx5_glue_test
```

Apply only after you confirm the dataset was recorded with that old current
order:

```bash
python -m arx5_collector.scripts.reorder_videos_to_old_dp_schema \
  data_local/arx5_glue_test \
  --apply
```

## Audit Dataset Quality

Run this after each collection session:

```bash
python -m arx5_collector.scripts.analyze_recordings \
  data_local/arx5_glue_test
```

Check:

```text
episode count
episode length summary
global static action ratio
action min/max/mean
robot0_* min/max/mean
camera frame count and FPS
camera static-frame ratio
```

If the static action ratio is high or episodes are very long, collect shorter
episodes or trim the idle sections before training.

## Training Dataset Processing

The collector controls the robot at 100Hz but writes state/action to
`replay_buffer.zarr` at 20Hz by default. Lowdim/action writing is delayed until
the same scheduled `start_time` used by video recording, so the first lowdim
sample and first video frame stay aligned. The older ARX5 DP pipeline also
trained closer to 20Hz. Training keeps a 20Hz resampling option for old datasets:

```yaml
task.dataset.target_frequency: 20.0
```

This makes each episode much more compact and reduces repeated image frames.
For old 100Hz datasets, the converter now preselects target-frequency lowdim
indices before decoding video, so it only reads the frames that are actually
used for training. It also brings the new dataset distribution closer to the
older working `try_/arx5_diffusion` project.

Inspect the exact training replay buffer:

```bash
python -m arx5_collector.scripts.inspect_training_dataset \
  --dataset-path data_local/arx5_glue_test
```

The first run may take a few minutes because it builds the image cache. After
that, the cache is reused. The command prints `Training readiness: PASS/FAIL`
and warnings for issues such as too few episodes or high static-action ratio.

For a higher-rate experiment, collect with `--data-frequency 50` and train with:

```bash
python train.py \
  --config-name=train_diffusion_unet_arx5_hybrid_workspace \
  task.dataset_path=data_local/arx5_glue_plus_50hz \
  task.dataset.target_frequency=50 \
  training.num_epochs=50 \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=4 \
  val_dataloader.num_workers=4 \
  training.checkpoint_every=10 \
  logging.mode=offline
```

Because video is normally 30fps, 50Hz lowdim/action data means some adjacent
training samples will share camera frames. Use this as an experiment rather
than the default pipeline.

Optional static start/end trimming:

```bash
python -m arx5_collector.scripts.inspect_training_dataset \
  --dataset-path data_local/arx5_glue_test \
  --trim-static
```

For training, enable trimming with:

```bash
task.dataset.trim_static_start_end=True
```

## Train

The default `arx5_image` task now uses the old `robot0_*` observation keys and
a 10% validation split. It also resamples data to 20Hz before constructing
training sequences.

Mini test:

```bash
python train.py \
  --config-name=train_diffusion_unet_arx5_hybrid_workspace \
  task.dataset_path=data_local/arx5_glue_test \
  training.num_epochs=2 \
  dataloader.batch_size=2 \
  val_dataloader.batch_size=2 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  logging.mode=offline
```

Formal run:

```bash
python train.py \
  --config-name=train_diffusion_unet_arx5_hybrid_workspace \
  task.dataset_path=data_local/arx5_glue_test \
  training.num_epochs=200 \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=4 \
  val_dataloader.num_workers=4 \
  training.checkpoint_every=25 \
  logging.mode=offline
```

## Deploy Checkpoint

Dry run first:

```bash
python -m arx5_ckpt_loader.load_arx5_ckpt \
  --ckpt data/outputs/YYYY.MM.DD/RUN/checkpoints/latest.ckpt \
  --dry-run
```

Then live inference without execution:

```bash
python -m arx5_ckpt_loader.run_arx5_policy \
  --ckpt data/outputs/YYYY.MM.DD/RUN/checkpoints/latest.ckpt \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  --device cuda:0
```

Execution:

```bash
python -m arx5_ckpt_loader.run_arx5_policy \
  --ckpt data/outputs/YYYY.MM.DD/RUN/checkpoints/latest.ckpt \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  --device cuda:0 \
  --execute
```

Older checkpoints that were trained with explicit Linux video devices should
keep the original device order, for example:

```bash
python -m arx5_ckpt_loader.run_arx5_policy \
  --ckpt data/outputs/YYYY.MM.DD/RUN/checkpoints/latest.ckpt \
  --model X5 \
  --interface can1 \
  --video-devices 0,6,12 \
  --device cuda:0
```

## Online Action Chunk Scheduling

Online deployment now executes action chunks with timestamps, like the old
`eval_arx5.py` path:

```text
get aligned observation
predict action chunk
schedule future actions at fixed timestamps
run the next policy inference before the chunk is exhausted
```

Default execution schedules `cfg.n_action_steps` future actions per policy
inference. To test without sending commands:

```bash
python -m arx5_ckpt_loader.run_arx5_policy \
  --ckpt data/outputs/YYYY.MM.DD/RUN/checkpoints/latest.ckpt \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  --device cuda:0
```

To execute:

```bash
python -m arx5_ckpt_loader.run_arx5_policy \
  --ckpt data/outputs/YYYY.MM.DD/RUN/checkpoints/latest.ckpt \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  --device cuda:0 \
  --execute
```

Fallback to the previous one-action-at-a-time behavior:

```bash
--single-action
```

If the policy is late, the runner drops stale actions and schedules the newest
available action at the next reachable timestamp.
