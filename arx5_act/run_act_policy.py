import argparse
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import cv2
import numpy as np

from arx5_act.policy_utils import (
    load_bundle,
    make_image_tensor,
    predict_action_chunk,
    qpos_from_robot_state,
    resolve_device,
)
from arx5_act.deployment.temporal import TemporalActionAggregator
from arx5_act.paths import ensure_project_paths
from arx5_act.paths import DP_ROOT

ensure_project_paths()
from arx5_act.deployment.joint_robot import JointActRobot
from arx5_ckpt_loader.action_adapter import clamp_gripper
from arx5_ckpt_loader.deployment.action_postprocess import (
    clamp_action_chunk_delta,
    make_anchor_action,
)
from arx5_ckpt_loader.deployment.arx_runtime import PolicyActionScheduler
from arx5_ckpt_loader.deployment.reset import reset_to_home_checked, reset_to_pose


def resize_frame(frame, width=426, height=240):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def make_preview(camera_frames):
    return np.concatenate(
        [resize_frame(camera_frames[key]) for key in sorted(camera_frames.keys())],
        axis=1,
    )


def resolve_optional_config_path(path):
    if path is None:
        return None
    config_path = Path(path).expanduser()
    if config_path.is_absolute():
        return str(config_path)
    if config_path.is_file():
        return str(config_path)
    dp_config_path = DP_ROOT / config_path
    if dp_config_path.is_file():
        return str(dp_config_path)
    return str(config_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", default="policy_best.ckpt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model", default="X5")
    parser.add_argument("--interface", default="can1")
    parser.add_argument("--realsense-config", default=None)
    parser.add_argument("--usb-device", type=int, default=0)
    parser.add_argument("--usb-config", default=str(DP_ROOT / "scripts/camera/usb/current.yaml"))
    parser.add_argument("--video-devices", default=None)
    parser.add_argument("--camera-order", default="old_dp", choices=["old_dp", "current_collector"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--usb-width", type=int, default=640)
    parser.add_argument("--usb-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--frequency", type=float, default=None)
    parser.add_argument("--steps-per-inference", type=int, default=5)
    parser.add_argument("--temporal-agg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-agg-k", type=float, default=0.01)
    parser.add_argument("--temporal-agg-order", default="oldest", choices=["oldest", "newest"])
    parser.add_argument("--query-frequency", type=int, default=1)
    parser.add_argument("--command-mode", default="traj", choices=["cmd", "traj"])
    parser.add_argument("--command-latency", type=float, default=0.05)
    parser.add_argument("--action-exec-latency", type=float, default=0.01)
    parser.add_argument("--preview-time", type=float, default=0.05)
    parser.add_argument("--arm-gain-mode", default="default", choices=["default", "damping", "pro"])
    parser.add_argument("--arm-kp-scale", type=float, default=1.5)
    parser.add_argument("--arm-kd-scale", type=float, default=0.5)
    parser.add_argument("--tracking-pos-error-limit", type=float, default=0.08)
    parser.add_argument("--tracking-rot-error-limit", type=float, default=0.6)
    parser.add_argument("--tracking-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-log-interval", type=float, default=0.5)
    parser.add_argument("--max-action-pos-step", type=float, default=0.006)
    parser.add_argument("--max-action-rot-step", type=float, default=0.02)
    parser.add_argument("--max-action-joint-step", type=float, default=0.04)
    parser.add_argument("--max-action-gripper-step", type=float, default=0.003)
    parser.add_argument("--policy-start-hold-time", type=float, default=0.5)
    parser.add_argument("--policy-start-max-pos-step", type=float, default=0.002)
    parser.add_argument("--policy-start-max-rot-step", type=float, default=0.008)
    parser.add_argument("--prepend-current-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-anchor", default="target", choices=["target", "actual"])
    parser.add_argument("--disable-action-safety", action="store_true")
    parser.add_argument("--action-y-gain", type=float, default=1.0)
    parser.add_argument("--gripper-margin", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--start-policy", action="store_true")
    parser.add_argument("--reset-to-home-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-target", default="home", choices=["session", "home"])
    parser.add_argument(
        "--reset-gripper-target",
        default="sdk",
        choices=["sdk", "session", "current", "open", "close"],
    )
    parser.add_argument("--reset-duration", type=float, default=2.0)
    parser.add_argument("--reset-gain-mode", default="default", choices=["default", "damping", "pro"])
    parser.add_argument("--reset-arm-kp-scale", type=float, default=1.0)
    parser.add_argument("--reset-arm-kd-scale", type=float, default=1.0)
    parser.add_argument("--reset-attempts", type=int, default=0)
    parser.add_argument("--reset-settle-time", type=float, default=0.35)
    parser.add_argument("--reset-pos-tolerance", type=float, default=0.006)
    parser.add_argument("--reset-rot-tolerance", type=float, default=0.05)
    parser.add_argument("--reset-joint-tolerance", type=float, default=0.08)
    parser.add_argument("--restore-gain-after-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pos-speed", type=float, default=0.4)
    parser.add_argument("--rot-speed", type=float, default=0.75)
    parser.add_argument("--gripper-speed", type=float, default=0.08)
    parser.add_argument("--gripper-safe-torque", type=float, default=0.75)
    parser.add_argument("--gripper-safe-margin", type=float, default=0.002)
    parser.add_argument("--spacemouse-deadzone", type=float, default=0.02)
    parser.add_argument("--spacemouse-smoothing-window", type=int, default=0)
    parser.add_argument("--debug-teleop", action="store_true")
    parser.add_argument("--debug-policy-chunk", action="store_true")
    return parser.parse_args()


def get_requested_key(key_counter, cv_key, key_code):
    requested_key = None
    for key_stroke in key_counter.get_press_events():
        if key_stroke == key_code(char="q"):
            requested_key = "q"
        elif key_stroke == key_code(char="c"):
            requested_key = "c"
        elif key_stroke == key_code(char="h"):
            requested_key = "h"
        elif key_stroke == key_code(char="r"):
            requested_key = "r"

    if cv_key == ord("q"):
        requested_key = "q"
    elif cv_key == ord("c"):
        requested_key = "c"
    elif cv_key == ord("h"):
        requested_key = "h"
    elif cv_key == ord("r"):
        requested_key = "r"
    return requested_key


def make_joint_anchor_action(robot_state, target_joints, target_gripper, action_anchor):
    if action_anchor == "actual":
        joints = np.asarray(robot_state["ActualQ"], dtype=np.float64)
        gripper = float(robot_state["gripper_pos"][0])
    else:
        joints = np.asarray(target_joints, dtype=np.float64)
        gripper = float(target_gripper)
    return np.concatenate([joints[:6], np.array([gripper], dtype=np.float64)])


def clamp_joint_action_delta(
    action_chunk,
    anchor_action,
    max_joint_step: float,
    max_gripper_step: float,
):
    action_chunk = np.asarray(action_chunk, dtype=np.float64).copy()
    anchor_action = np.asarray(anchor_action, dtype=np.float64)
    clipped = False
    if max_joint_step > 0:
        delta = action_chunk[:, :6] - anchor_action[:6]
        clipped_delta = np.clip(delta, -float(max_joint_step), float(max_joint_step))
        clipped = clipped or bool(np.any(np.abs(clipped_delta - delta) > 1e-12))
        action_chunk[:, :6] = anchor_action[:6] + clipped_delta
    if max_gripper_step > 0:
        delta = action_chunk[:, 6] - anchor_action[6]
        clipped_delta = np.clip(delta, -float(max_gripper_step), float(max_gripper_step))
        clipped = clipped or bool(np.any(np.abs(clipped_delta - delta) > 1e-12))
        action_chunk[:, 6] = anchor_action[6] + clipped_delta
    return action_chunk, clipped


def main():
    args = parse_args()
    cv2.setNumThreads(1)

    from arx5_ckpt_loader.obs_buffer import make_camera_frame_dict, make_video_device_frame_dict
    from arx5_ckpt_loader.video_device_reader import VideoDeviceReader
    from arx5_collector.camera import ThreeCameraRecorder
    from arx5_collector.input import SpaceMouseTeleop
    from arx5_collector.robot import Arx5Robot
    from diffusion_policy.real_world.keystroke_counter import KeystrokeCounter
    from pynput.keyboard import KeyCode

    device = resolve_device(args.device)
    config, stats, policy, ckpt_path = load_bundle(args.ckpt_dir, args.ckpt_name, device)
    camera_names = config["camera_names"]
    chunk_size = int(config["chunk_size"])
    state_mode = config.get("state_mode", "eef")
    if args.frequency is None:
        args.frequency = float(config.get("target_frequency") or 20.0)
    steps_per_inference = max(1, min(int(args.steps_per_inference), chunk_size))
    dt = 1.0 / float(args.frequency)
    realsense_config = resolve_optional_config_path(args.realsense_config)
    usb_config = resolve_optional_config_path(args.usb_config)

    print(f"ACT ckpt: {ckpt_path}")
    print(f"device: {device}")
    print(f"camera_names: {camera_names}")
    print(f"chunk_size: {chunk_size}")
    print(f"state_mode: {state_mode}")
    if state_mode not in ("eef", "joint"):
        raise RuntimeError(f"Unsupported ACT state_mode={state_mode!r}.")
    if state_mode == "joint":
        print("Joint ACT mode: actions are sent as [q1..q6, gripper] through Arx5JointController.")
        print("Joint ACT human mode holds the current joint target; SpaceMouse EEF teleop is disabled in this mode.")
    print(f"execute: {args.execute}")
    print("Controls:")
    print("  human mode: SpaceMouse controls robot")
    print("  c: switch to policy mode")
    print("  h: switch back to human mode")
    print("  r: reset robot and switch to human mode")
    print("  q: quit")
    print(
        "Policy timing:",
        f"frequency={args.frequency:.2f}Hz",
        f"steps_per_inference={steps_per_inference}",
        f"temporal_agg={args.temporal_agg}",
        f"temporal_agg_order={args.temporal_agg_order}",
        f"query_frequency={args.query_frequency}",
    )
    print(f"RealSense config: {realsense_config}")
    print(f"USB config: {usb_config}")
    print(
        "Gripper safety:",
        f"safe_torque={args.gripper_safe_torque}",
        f"safe_margin={args.gripper_safe_margin}",
        )
    if args.disable_action_safety:
        print("Action safety: disabled")
    else:
        if state_mode == "joint":
            print(
                "Action safety:",
                f"max_joint_step={args.max_action_joint_step}",
                f"max_gripper_step={args.max_action_gripper_step}",
                f"anchor={args.action_anchor}",
            )
        else:
            print(
                "Action safety:",
                f"max_pos_step={args.max_action_pos_step}",
                f"max_rot_step={args.max_action_rot_step}",
                f"max_gripper_step={args.max_action_gripper_step}",
                f"anchor={args.action_anchor}",
            )

    with SharedMemoryManager() as shm_manager:
        robot_cls = JointActRobot if state_mode == "joint" else Arx5Robot
        robot = robot_cls(
            model=args.model,
            interface=args.interface,
            reset_to_home=False,
            preview_time=args.preview_time,
            command_mode=args.command_mode,
            arm_gain_mode=args.arm_gain_mode,
            arm_kp_scale=args.arm_kp_scale,
            arm_kd_scale=args.arm_kd_scale,
            gripper_safe_torque=args.gripper_safe_torque,
            gripper_safe_margin=args.gripper_safe_margin,
            tracking_pos_error_limit=args.tracking_pos_error_limit,
            tracking_rot_error_limit=args.tracking_rot_error_limit,
        )
        teleop = None
        if state_mode != "joint":
            teleop = SpaceMouseTeleop(
                shm_manager=shm_manager,
                pos_speed=args.pos_speed,
                rot_speed=args.rot_speed,
                gripper_speed=args.gripper_speed,
                gripper_margin=args.gripper_margin,
                deadzone=args.spacemouse_deadzone,
                smoothing_window=args.spacemouse_smoothing_window,
            )
        cameras = None
        video_reader = None
        if args.video_devices:
            video_reader = VideoDeviceReader(
                devices=[int(x.strip()) for x in args.video_devices.split(",") if x.strip()],
                resolution=(args.usb_width, args.usb_height),
                fps=args.camera_fps,
                config_path=usb_config,
            )
        else:
            cameras = ThreeCameraRecorder(
                shm_manager=shm_manager,
                realsense_config=realsense_config,
                usb_device=args.usb_device,
                resolution=(args.width, args.height),
                usb_resolution=(args.usb_width, args.usb_height),
                fps=args.camera_fps,
                usb_config=usb_config,
            )

        key_counter = None
        try:
            robot.start()
            if args.reset_to_home_start:
                print("Resetting robot to home...")
                robot.reset_to_home()
                print("Robot reset to home.")

            robot_state = robot.get_state()
            if state_mode == "joint":
                target_pose = robot_state["ActualQ"].copy()
            else:
                target_pose = robot_state["ActualTCPPose"].copy()
            target_gripper = float(robot_state["gripper_pos"][0])
            session_initial_pose = target_pose.copy()
            session_initial_gripper = target_gripper
            mode = "policy" if args.start_policy else "human"
            if mode == "policy" and not args.execute:
                print("Policy mode requested, but --execute is disabled. Running dry-run predictions.")
            print(f"Initial mode: {mode}")
            gain_summary = robot.get_gain_summary()
            print(
                "Arm gain:",
                f"mode={args.arm_gain_mode}",
                f"kp={np.array2string(gain_summary['kp'], precision=3)}",
                f"kd={np.array2string(gain_summary['kd'], precision=3)}",
                f"gripper_kp={gain_summary['gripper_kp']:.3f}",
                f"gripper_kd={gain_summary['gripper_kd']:.3f}",
            )
            policy_scheduler = PolicyActionScheduler(
                robot=robot,
                command_latency=args.command_latency,
                tracking_guard=args.tracking_guard,
                log_interval=args.policy_log_interval,
            )
            print(
                "Reset target:",
                f"{args.reset_target}",
                f"gripper={args.reset_gripper_target}",
                f"reset_gain={args.reset_gain_mode}",
                f"reset_kp_scale={args.reset_arm_kp_scale}",
                f"reset_kd_scale={args.reset_arm_kd_scale}",
                f"session_{'joints' if state_mode == 'joint' else 'pose'}={np.array2string(session_initial_pose, precision=4)}",
                f"session_gripper={session_initial_gripper:.5f}",
            )

            if teleop is not None:
                teleop.start()
            if video_reader is not None:
                video_reader.start()
                print(f"Video devices: {args.video_devices}")
            else:
                cameras.start()
                print(f"Camera order: {args.camera_order}")
            key_counter = KeystrokeCounter()
            key_counter.start()
            print("Warming up cameras...")
            time.sleep(1.0)

            rs_cache = None
            t_loop = time.monotonic()
            next_policy_time = 0.0
            policy_start_time = time.monotonic() if mode == "policy" else None
            policy_step = 0
            temporal_aggregator = TemporalActionAggregator(
                chunk_size=chunk_size,
                k=args.temporal_agg_k,
                order=args.temporal_agg_order,
            )
            last_debug_print = 0.0
            last_policy_print = 0.0

            def perform_reset(reason):
                nonlocal mode, next_policy_time, policy_start_time, target_pose, target_gripper
                mode = "human"
                next_policy_time = 0.0
                policy_start_time = None
                temporal_aggregator.clear()
                print(f"Resetting robot to {args.reset_target} ({reason})...")
                robot.apply_gain(
                    arm_gain_mode=args.reset_gain_mode,
                    arm_kp_scale=args.reset_arm_kp_scale,
                    arm_kd_scale=args.reset_arm_kd_scale,
                )
                reset_ok = False
                try:
                    if args.reset_target == "home":
                        robot_state, reset_ok = reset_to_home_checked(
                            robot,
                            attempts=args.reset_attempts,
                            settle_time=args.reset_settle_time,
                            pos_tolerance=args.reset_pos_tolerance,
                            rot_tolerance=args.reset_rot_tolerance,
                            joint_tolerance=args.reset_joint_tolerance,
                        )
                    elif state_mode == "joint":
                        current_reset_state = robot.get_state()
                        if args.reset_gripper_target in ("sdk", "open"):
                            reset_gripper = float(robot.robot_config.gripper_width)
                        elif args.reset_gripper_target == "close":
                            reset_gripper = 0.0
                        elif args.reset_gripper_target == "current":
                            reset_gripper = float(current_reset_state["gripper_pos"][0])
                        else:
                            reset_gripper = float(session_initial_gripper)
                        robot_state, reset_ok = robot.reset_to_joints(
                            session_initial_pose,
                            gripper=reset_gripper,
                            duration=args.reset_duration,
                            settle_time=args.reset_settle_time,
                            attempts=args.reset_attempts,
                            joint_tolerance=args.reset_joint_tolerance,
                        )
                    else:
                        current_reset_state = robot.get_state()
                        if args.reset_gripper_target in ("sdk", "open"):
                            reset_gripper = float(robot.robot_config.gripper_width)
                        elif args.reset_gripper_target == "close":
                            reset_gripper = 0.0
                        elif args.reset_gripper_target == "current":
                            reset_gripper = float(current_reset_state["gripper_pos"][0])
                        else:
                            reset_gripper = float(session_initial_gripper)
                        robot_state, reset_ok = reset_to_pose(
                            robot,
                            pose=session_initial_pose,
                            gripper=reset_gripper,
                            duration=args.reset_duration,
                            frequency=max(args.frequency, 20.0),
                            command_latency=args.command_latency,
                            attempts=args.reset_attempts,
                            settle_time=args.reset_settle_time,
                            pos_tolerance=args.reset_pos_tolerance,
                            rot_tolerance=args.reset_rot_tolerance,
                        )
                finally:
                    if args.restore_gain_after_reset:
                        robot.restore_runtime_gain()

                if state_mode == "joint":
                    target_pose = robot_state["ActualQ"].copy()
                else:
                    target_pose = robot_state["ActualTCPPose"].copy()
                target_gripper = float(robot_state["gripper_pos"][0])
                if reset_ok:
                    print(f"Robot reset to {args.reset_target}.")
                else:
                    print(
                        f"WARNING: reset to {args.reset_target} did not reach tolerance "
                        f"after {args.reset_attempts} attempts."
                    )
                return reset_ok

            while True:
                loop_start = time.monotonic()

                if video_reader is not None:
                    video_data = video_reader.get()
                    camera_frames = make_video_device_frame_dict(video_data)
                else:
                    rs_cache = cameras.realsense.get(out=rs_cache)
                    usb_data = cameras.pump_usb_frame()
                    camera_frames = make_camera_frame_dict(
                        rs_cache,
                        usb_data,
                        camera_order=args.camera_order,
                    )
                robot_state = robot.get_state()

                cv_key = -1
                if args.show_preview:
                    preview = make_preview(camera_frames)
                    status = f"ACT {mode.upper()} {'EXECUTE' if args.execute else 'DRY-RUN'}"
                    cv2.putText(
                        preview,
                        status,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("arx5_act_policy_preview", preview)
                    cv_key = cv2.pollKey()
                else:
                    cv_key = cv2.pollKey()

                requested_key = get_requested_key(key_counter, cv_key, KeyCode)
                if requested_key == "q":
                    perform_reset("quit")
                    break
                if requested_key == "c":
                    mode = "policy"
                    robot_state = robot.get_state()
                    if state_mode == "joint":
                        target_pose = robot_state["ActualQ"].copy()
                    else:
                        target_pose = robot_state["ActualTCPPose"].copy()
                    target_gripper = float(robot_state["gripper_pos"][0])
                    if args.execute:
                        robot.schedule_waypoint(
                            target_pose,
                            target_time=time.time() + args.command_latency,
                            gripper_pos=target_gripper,
                        )
                    if hasattr(policy, "reset"):
                        policy.reset()
                    policy_step = 0
                    temporal_aggregator.clear()
                    next_policy_time = 0.0
                    policy_start_time = time.monotonic()
                    print("Switched to policy mode. Holding current pose before first ACT chunk.")
                elif requested_key == "h":
                    mode = "human"
                    robot_state = robot.get_state()
                    if state_mode == "joint":
                        target_pose = robot_state["ActualQ"].copy()
                    else:
                        target_pose = robot_state["ActualTCPPose"].copy()
                    target_gripper = float(robot_state["gripper_pos"][0])
                    temporal_aggregator.clear()
                    next_policy_time = 0.0
                    policy_start_time = None
                    print("Switched to human mode.")
                elif requested_key == "r":
                    perform_reset("manual")

                if mode == "human":
                    if state_mode == "joint":
                        target_pose = robot_state["ActualQ"].copy()
                        target_gripper = float(robot_state["gripper_pos"][0])
                        now_mono = time.monotonic()
                        if args.debug_teleop and now_mono - last_debug_print > 1.0:
                            last_debug_print = now_mono
                            print(
                                "joint hold",
                                "actual_q",
                                np.array2string(target_pose, precision=4),
                                f"gripper={target_gripper:.5f}",
                            )
                    else:
                        command = teleop.update(
                            target_pose=target_pose,
                            target_gripper_pos=target_gripper,
                            dt=dt,
                            gripper_width=robot.robot_config.gripper_width,
                        )
                        target_pose = command.action
                        target_gripper = command.gripper_action
                        now_mono = time.monotonic()
                        if args.debug_teleop and (
                            np.linalg.norm(command.raw_motion) > 1e-3
                            or now_mono - last_debug_print > 1.0
                        ):
                            last_debug_print = now_mono
                            print(
                                "spacemouse",
                                np.array2string(command.raw_motion, precision=3),
                                "actual",
                                np.array2string(robot_state["ActualTCPPose"], precision=4),
                                "target",
                                np.array2string(target_pose, precision=4),
                                f"gripper={target_gripper:.5f}",
                            )
                    robot.schedule_waypoint(
                        target_pose,
                        target_time=time.time() + args.command_latency,
                        gripper_pos=target_gripper,
                    )
                else:
                    now_mono = time.monotonic()
                    in_policy_start_hold = (
                        policy_start_time is not None
                        and now_mono - policy_start_time < args.policy_start_hold_time
                    )
                    if in_policy_start_hold:
                        if args.execute:
                            robot.schedule_waypoint(
                                target_pose,
                                target_time=time.time() + args.command_latency,
                                gripper_pos=target_gripper,
                            )
                        if now_mono - last_policy_print >= args.policy_log_interval:
                            last_policy_print = now_mono
                            remaining = args.policy_start_hold_time - (
                                now_mono - policy_start_time
                            )
                            print(
                                "policy start hold",
                                f"remaining={max(0.0, remaining):.2f}s",
                                f"pose={np.array2string(target_pose, precision=4)}",
                            )
                        t_loop += dt
                        sleep_time = t_loop - time.monotonic()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        else:
                            t_loop = loop_start
                        continue
                    if (not args.temporal_agg) and now_mono < next_policy_time:
                        t_loop += dt
                        sleep_time = t_loop - time.monotonic()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        else:
                            t_loop = loop_start
                        continue

                    if state_mode == "joint":
                        anchor_action = make_joint_anchor_action(
                            robot_state=robot_state,
                            target_joints=target_pose,
                            target_gripper=target_gripper,
                            action_anchor=args.action_anchor,
                        )
                    else:
                        anchor_action = make_anchor_action(
                            robot_state=robot_state,
                            target_pose=target_pose,
                            target_gripper=target_gripper,
                            action_anchor=args.action_anchor,
                        )

                    qpos = qpos_from_robot_state(robot_state, state_mode=state_mode)
                    image = make_image_tensor(camera_frames, camera_names, device)

                    if args.temporal_agg:
                        query_frequency = max(1, int(args.query_frequency))
                        if policy_step % query_frequency == 0:
                            new_chunk = np.asarray(
                                predict_action_chunk(policy, image, qpos, stats),
                                dtype=np.float64,
                            ).copy()
                            new_chunk[:, 6] = [
                                clamp_gripper(
                                    value,
                                    width=robot.robot_config.gripper_width,
                                    margin=args.gripper_margin,
                                )
                                for value in new_chunk[:, 6]
                            ]
                            if args.debug_policy_chunk:
                                print(
                                    "raw act chunk",
                                    f"step={policy_step}",
                                    f"x={new_chunk[:, 0].min():.4f}/{new_chunk[:, 0].max():.4f}",
                                    f"y={new_chunk[:, 1].min():.4f}/{new_chunk[:, 1].max():.4f}",
                                    f"z={new_chunk[:, 2].min():.4f}/{new_chunk[:, 2].max():.4f}",
                                    f"gripper={new_chunk[:, 6].min():.5f}/{new_chunk[:, 6].max():.5f}",
                                )
                            temporal_aggregator.add_chunk(policy_step, new_chunk)

                        action, source_count = temporal_aggregator.current_action(
                            step=policy_step,
                            fallback_action=anchor_action,
                        )
                        if state_mode != "joint" and args.action_y_gain != 1.0:
                            action[1] = anchor_action[1] + float(args.action_y_gain) * (
                                action[1] - anchor_action[1]
                            )

                        if args.disable_action_safety:
                            action_clipped = False
                        elif state_mode == "joint":
                            action_chunk, action_clipped = clamp_joint_action_delta(
                                action[None],
                                anchor_action=anchor_action,
                                max_joint_step=args.max_action_joint_step,
                                max_gripper_step=args.max_action_gripper_step,
                            )
                            action = action_chunk[0]
                        else:
                            action_chunk, action_clipped = clamp_action_chunk_delta(
                                action[None],
                                anchor_action=anchor_action,
                                max_pos_step=args.max_action_pos_step,
                                max_rot_step=args.max_action_rot_step,
                                max_gripper_step=args.max_action_gripper_step,
                            )
                            action = action_chunk[0]

                        curr_time = time.time()
                        action_time = curr_time + args.command_latency
                        now_print = time.monotonic()
                        if now_print - last_policy_print >= args.policy_log_interval:
                            last_policy_print = now_print
                            print(
                                "act temporal",
                                f"step={policy_step}",
                                f"sources={source_count}",
                                f"clipped={action_clipped}",
                                f"pose={np.array2string(action[:6], precision=4)}",
                                f"gripper={action[6]:.5f}",
                            )
                        if args.execute:
                            if state_mode == "joint":
                                robot.schedule_waypoint(
                                    action[:6],
                                    target_time=action_time,
                                    gripper_pos=float(action[6]),
                                )
                            else:
                                policy_scheduler.schedule(
                                    action[:6],
                                    target_time=action_time,
                                    gripper_pos=float(action[6]),
                                )
                            target_pose = action[:6].copy()
                            target_gripper = float(action[6])
                        policy_step += 1
                    else:
                        action_chunk = predict_action_chunk(policy, image, qpos, stats)
                        action_chunk = np.asarray(action_chunk, dtype=np.float64)[
                            :steps_per_inference
                        ].copy()
                        action_chunk[:, 6] = [
                            clamp_gripper(
                                value,
                                width=robot.robot_config.gripper_width,
                                margin=args.gripper_margin,
                            )
                            for value in action_chunk[:, 6]
                        ]
                        if args.prepend_current_action and len(action_chunk) > 0:
                            action_chunk[0] = anchor_action
                        if args.disable_action_safety:
                            action_clipped = False
                        elif state_mode == "joint":
                            action_chunk, action_clipped = clamp_joint_action_delta(
                                action_chunk,
                                anchor_action=anchor_action,
                                max_joint_step=args.max_action_joint_step,
                                max_gripper_step=args.max_action_gripper_step,
                            )
                        else:
                            action_chunk, action_clipped = clamp_action_chunk_delta(
                                action_chunk,
                                anchor_action=anchor_action,
                                max_pos_step=args.max_action_pos_step,
                                max_rot_step=args.max_action_rot_step,
                                max_gripper_step=args.max_action_gripper_step,
                            )

                        curr_time = time.time()
                        action_timestamps = (
                            np.arange(len(action_chunk), dtype=np.float64) * dt
                            + curr_time
                            + args.command_latency
                        )
                        is_new = action_timestamps > (curr_time + args.action_exec_latency)
                        if np.sum(is_new) == 0:
                            action_chunk = action_chunk[[-1]]
                            action_timestamps = np.array([curr_time + max(args.command_latency, dt)])
                            print(
                                "policy over budget;",
                                f"scheduling latest action at +{action_timestamps[0] - curr_time:.3f}s",
                            )
                        else:
                            action_chunk = action_chunk[is_new]
                            action_timestamps = action_timestamps[is_new]

                        now_print = time.monotonic()
                        if now_print - last_policy_print >= args.policy_log_interval:
                            last_policy_print = now_print
                            print(
                                "act chunk",
                                f"n={len(action_chunk)}",
                                f"dt0={action_timestamps[0] - curr_time:.3f}s",
                                f"dtN={action_timestamps[-1] - curr_time:.3f}s",
                                f"clipped={action_clipped}",
                                f"pose0={np.array2string(action_chunk[0, :6], precision=4)}",
                                f"gripper_minmax={action_chunk[:, 6].min():.5f}/{action_chunk[:, 6].max():.5f}",
                            )
                        if args.execute:
                            robot.schedule_waypoints(
                                action_chunk,
                                action_timestamps,
                                gripper_margin=args.gripper_margin,
                            )
                            target_pose = action_chunk[-1, :6].copy()
                            target_gripper = float(action_chunk[-1, 6])
                        next_policy_time = time.monotonic() + len(action_chunk) * dt

                t_loop += dt
                sleep_time = t_loop - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    t_loop = loop_start
        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            if key_counter is not None:
                key_counter.stop()
                key_counter.join()
            if teleop is not None:
                teleop.stop()
            if video_reader is not None:
                video_reader.stop()
            if cameras is not None:
                cameras.stop()
            robot.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
