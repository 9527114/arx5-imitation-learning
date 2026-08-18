# ARX5 ckpt loader

This folder loads and deploys ARX5 Diffusion Policy checkpoints.

Canonical camera order now follows the older ARX5 DP project:

```text
camera_0 = USB wrist camera
camera_1 = first RealSense view
camera_2 = second RealSense view
```

## Load the mini checkpoint

From `diffusion_policy-main`:

```bash
python -m arx5_ckpt_loader.load_arx5_ckpt
```

## Load another checkpoint

```bash
python -m arx5_ckpt_loader.load_arx5_ckpt \
  --ckpt data/outputs/2026.07.08/18.15.27_train_diffusion_unet_arx5_hybrid_arx5_image/checkpoints/latest.ckpt
```

## Also run one dummy inference pass

This only feeds zero images and zero robot states into the model to verify that
`policy.predict_action()` can run.

```bash
python -m arx5_ckpt_loader.load_arx5_ckpt --dry-run
```

The dummy observation path is only for checking that the checkpoint and policy
can run without hardware.

## Run live cameras + robot state, but do not execute actions

```bash
python -m arx5_ckpt_loader.run_arx5_policy \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  --device cuda:0
```

This is the safe default. It prints predicted actions but does not send them to
the robot.

## Execute Predicted Actions On The Robot

Only use this after the dry-run output looks reasonable.

```bash
../../scripts/run_dp_pro.sh
```

Current tuned DP deployment command:

```bash
python -m arx5_ckpt_loader.run_arx5_policy \
  --ckpt data/outputs/manual/glue_motion_edge_v2_three_200/dp/checkpoints/latest.ckpt \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  --device cuda:0 \
  --execute \
  --execution-layer continuous \
  --arm-gain-mode pro \
  --arm-kp-scale 1 \
  --arm-kd-scale 1 \
  --preview-time 0.1 \
  --steps-per-inference 8 \
  --command-latency 0.01 \
  --action-exec-latency 0.01 \
  --boundary-blend-steps 0 \
  --no-prepend-current-action \
  --disable-action-safety \
  --continuous-frequency 200 \
  --continuous-max-pos-speed 0.65 \
  --continuous-max-rot-speed 1.05 \
  --trajectory-log data_local/policy_logs/dp_continuous_pro.jsonl
```

Policy execution uses action chunk scheduling. One inference returns a sequence
of future actions, and the `continuous` executor samples a timestamped trajectory
at 200 Hz before streaming EEF commands to the ARX5 SDK.

Useful timing options:

```bash
--steps-per-inference 8    # override cfg.n_action_steps
--frequency 30            # control/action timestamp rate
--command-latency 0.05    # planned lead time for commands
--single-action           # fallback to old one-action-at-a-time behavior
```

## Deployment Smoothness

The online execution path is split across small modules:

- `run_arx5_policy.py`: cameras, keyboard state machine, policy inference loop
- `deployment/action_postprocess.py`: chunk boundary blend, per-step clamp, deadband
- `trajectory_buffer.py`: old/new chunk fusion and high-rate interpolated execution
- `deployment/reset.py`: session reset trajectory
- `arx5_collector/robot/arx5_robot.py`: ARX5 SDK wrapper and gain application

Current ARX5-friendly defaults:

```bash
--execution-layer continuous
--arm-gain-mode pro
--arm-kp-scale 1
--arm-kd-scale 1
--preview-time 0.1
--steps-per-inference 8
--command-latency 0.01
--action-exec-latency 0.01
--continuous-frequency 200
--continuous-max-pos-speed 0.65
--continuous-max-rot-speed 1.05
--tracking-guard
```

Why this differs from the original UR5 deployment:

- The `continuous` executor keeps a timestamped trajectory and streams EEF
  commands at high rate, matching the older ARX5 DP deployment behavior more
  closely than repeatedly replacing SDK trajectories.
- `pro` gain uses the tuned ARX5 deployment stiffness/damping profile.
- `tracking_guard` stops policy execution from chasing targets when SDK target
  pose and actual pose have already diverged too far.

Tuning order:

1. If the arm has frame-by-frame stepping, keep `--execution-layer buffer` and
   increase `--buffer-blend-time` to `0.16` or `0.20`.
2. If it feels sluggish and cannot reach the object, reduce `--buffer-blend-time`
   to `0.08` before changing PID.
3. If it oscillates around a pose, try `--arm-kd-scale 0.7` with the same
   `--arm-kp-scale 1.5`.
4. If it still trembles under small corrections, try `--arm-kp-scale 1.2`
   and keep `--arm-kd-scale 0.5`.

Use a trajectory log while tuning:

```bash
--trajectory-log data_local/policy_logs/dp_tune.jsonl
```

The action convention is:

- `action[:6]`: absolute target TCP pose
- `action[6]`: target gripper width

The preferred observation keys are:

- `robot0_eef_pos`
- `robot0_eef_rot_axis_angle`
- `robot0_gripper_width`
