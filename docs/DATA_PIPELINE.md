# Data Pipeline

This project uses one raw ARX5 demonstration format and several training adapters.

## Raw Episode Layout

The collector writes one dataset directory:

```text
data_local/<dataset_name>/
├── replay_buffer.zarr/
└── videos/
    ├── 0/
    │   ├── 0.mp4
    │   ├── 1.mp4
    │   └── 2.mp4
    ├── 1/
    └── ...
```

Video mapping:

```text
0.mp4 -> camera_0 -> USB wrist camera
1.mp4 -> camera_1 -> RealSense view 0
2.mp4 -> camera_2 -> RealSense view 1
```

The collector metadata stores the camera order as:

```text
camera_0=usb,camera_1=realsense_0,camera_2=realsense_1
```

## Low-Dimensional Fields

Common `replay_buffer.zarr` keys include:

```text
timestamp
action
robot0_eef_pos
robot0_eef_rot_axis_angle
robot0_gripper_width
robot_eef_pose
robot_eef_pose_vel
robot_gripper
robot_joint
robot_joint_vel
target_gripper
gripper_torque
stage
```

`episode_ends` stores cumulative episode boundaries. For example, if episode lengths are `[100, 120]`, then `episode_ends` is `[100, 220]`.

## Synchronization

The low-dimensional stream is timestamped during collection. Videos are stored per episode. Dataset loaders align image frames to low-dimensional timestamps by using the episode start timestamp and camera FPS:

```text
frame_index ~= round((sample_timestamp - episode_start_timestamp) * camera_fps)
```

If a video is shorter than the low-dimensional episode, the loaders trim valid samples to the part that has matching image frames. This is why an episode can be usable even if the very end of a camera video is missing, as long as the sampled training windows stay inside the valid region.

## DP Dataset

Main file:

```text
diffusion_policy-main/diffusion_policy/dataset/arx5_image_dataset.py
```

Default training shape:

```text
horizon = 16
n_obs_steps = 2
n_action_steps = 8
RGB = [3, 240, 320]
action = [7]
```

DP-EEF observation:

```text
camera_0, camera_1, camera_2
robot0_eef_pos
robot0_eef_rot_axis_angle
robot0_gripper_width
```

DP-EEF action:

```text
[target_eef_pos(3), target_eef_rot_axis_angle(3), target_gripper_width(1)]
```

DP-Joint uses the same raw dataset but changes the action mode:

```text
action = [robot_joint(6), robot_gripper(1)]
obs lowdim = robot_joint, robot_gripper
```

## CFG Dataset

Main file:

```text
arx5_dp_cfg/arx5_image_dataset_cfg.py
```

CFG keeps the DP-EEF observation/action format and adds:

```text
prev_action       [prev_cond_steps, 7]
prev_action_mask  [prev_cond_steps]
```

`prev_action_mode=future` means the condition is built from the previous or continuing future action segment used to smooth chunk transitions. The mask marks which rows are valid, especially near episode boundaries.

## ACT Dataset

Main file:

```text
arx5_act/dataset.py
```

ACT converts the same raw ARX5 dataset into:

```text
image
qpos
action
is_pad
```

`qpos` is the current robot state used by ACT. In this project:

- `state_mode=eef`: `qpos = [eef_pos(3), eef_rot_axis_angle(3), gripper(1)]`
- `state_mode=joint`: `qpos = [joint_q(6), gripper(1)]`

`is_pad` marks padded timesteps in action chunks so the loss can ignore fake trailing values.

## Sample Construction

For a valid index inside one episode:

1. Select two observation timesteps.
2. Load `camera_0`, `camera_1`, and `camera_2` frames for those timesteps.
3. Load matching low-dimensional robot observations.
4. Load a 16-step action sequence from the same episode.
5. Normalize images, low-dimensional observations, and actions.
6. Return a batchable dictionary to the training loop.

Sequences do not cross episode boundaries. Crossing boundaries would mix two unrelated task attempts and create invalid action supervision.

