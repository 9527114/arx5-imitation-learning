# Data Schema

The ARX5 collector writes episode-based datasets designed to be compatible with the local Diffusion Policy, DP-CFG, and ACT adapters.

## Dataset Layout

```text
data_local/<dataset_name>/
├── replay_buffer.zarr/
└── videos/
    ├── 0/
    │   ├── 0.mp4
    │   ├── 1.mp4
    │   └── 2.mp4
    └── ...
```

The `replay_buffer.zarr` stores low-dimensional robot data and actions. The `videos/` tree stores per-episode camera videos.

## Camera Mapping

The canonical ARX5 training order is:

| Key | Video file | Meaning |
| --- | --- | --- |
| `camera_0` | `0.mp4` | USB wrist camera |
| `camera_1` | `1.mp4` | RealSense view 0 |
| `camera_2` | `2.mp4` | RealSense view 1 |

The mapping is recorded by the episode writer metadata and repeated in ARX5 task configs. Deployment wrappers also allow explicit `video-devices` because Linux `/dev/video*` indices can change after reconnecting cameras.

## Low-Dimensional Fields

The collector writes these keys when ending an episode:

| Key | Meaning |
| --- | --- |
| `timestamp` | sample timestamp |
| `action` | expert target action |
| `stage` | task/stage marker |
| `robot0_eef_pos` | current EEF position |
| `robot0_eef_rot_axis_angle` | current EEF rotation in axis-angle form |
| `robot0_gripper_width` | current gripper width |
| `robot_eef_pose` | full EEF pose |
| `robot_eef_pose_vel` | EEF velocity |
| `robot_joint` | 6D joint position |
| `robot_joint_vel` | 6D joint velocity |
| `robot_gripper` | gripper width |
| `target_gripper` | target gripper width |
| `gripper_torque` | gripper torque readout |

## Action Representations

DP-EEF and DP-CFG use:

```text
[target_x, target_y, target_z, target_rx, target_ry, target_rz, target_gripper_width]
```

DP-Joint and ACT-Joint use:

```text
[target_q1, target_q2, target_q3, target_q4, target_q5, target_q6, target_gripper_width]
```

Joint datasets can be derived from the same raw demonstrations because the collector stores both EEF and joint state.

## Frequency

Control frequency and dataset/policy frequency are separate:

- The collector can run robot control at a higher frequency.
- The dataset and training loaders can resample to a lower target frequency, commonly `20 Hz`.
- Camera capture/pump/preview FPS are configured separately from robot control.

This separation is important: increasing control-loop frequency does not automatically mean the training sample frequency is the same.

## Sample Construction

DP loaders construct samples using:

- `horizon`
- `n_obs_steps`
- `n_action_steps`
- episode boundary padding
- video-frame availability checks
- optional target-frequency resampling

DP-CFG additionally returns:

- `prev_action`
- `prev_action_mask`

ACT loaders construct:

- `image`
- `qpos`
- `action`
- `is_pad`

where `qpos` follows either EEF or joint state mode.

## Alignment Checks

The repository includes audit scripts for:

- raw recording summaries
- replay buffer shape checks
- video length vs low-dimensional trajectory length
- camera/state alignment
- DP-Joint action conversion checks

Relevant tools:

```text
diffusion_policy-main/arx5_collector/scripts/analyze_recordings.py
diffusion_policy-main/arx5_collector/scripts/inspect_training_dataset.py
diffusion_policy-main/arx5_collector/scripts/check_dataset_alignment.py
diffusion_policy-main/arx5_collector/scripts/inspect_dp_joint_dataset.py
```
