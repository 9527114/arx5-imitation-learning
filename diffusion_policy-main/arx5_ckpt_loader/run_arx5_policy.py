import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import torch

from arx5_ckpt_loader.action_adapter import clamp_gripper, select_action
from arx5_ckpt_loader.deployment.action_postprocess import (
    apply_action_deadband,
    blend_chunk_start,
    clamp_action_chunk_delta,
    make_anchor_action,
)
from arx5_ckpt_loader.deployment.action_scheduler import (
    filter_future_actions,
    get_policy_action_sequence,
    make_timed_action_candidate,
)
from arx5_ckpt_loader.deployment.alignment import summarize_policy_alignment
from arx5_ckpt_loader.deployment.continuous_executor import ContinuousWaypointExecutor
from arx5_ckpt_loader.deployment.reset import reset_to_home_checked, reset_to_pose
from arx5_ckpt_loader.obs_buffer import (
    Arx5ObsBuffer,
    make_camera_frame_dict,
    make_video_device_frame_dict,
)
from arx5_ckpt_loader.policy_loader import (
    DEFAULT_CKPT,
    load_policy_from_ckpt,
    print_policy_summary,
)
from arx5_ckpt_loader.trajectory_buffer import (
    ActionTrajectoryBuffer,
    BufferedActionExecutor,
)
from arx5_ckpt_loader.trajectory_logger import TrajectoryLogger
from arx5_ckpt_loader.video_device_reader import VideoDeviceReader


DEFAULT_USB_CONFIG = "scripts/camera/usb/current.yaml"


def resize_frame(frame, width=426, height=240):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def make_preview(camera_frames):
    frames = [resize_frame(camera_frames[key]) for key in sorted(camera_frames.keys())]
    return np.concatenate(frames, axis=1)


# DP deployment control entrypoint.
#
# The online control path is:
#   cameras + robot state -> Arx5ObsBuffer -> policy.predict_action
#   -> action_chunk postprocess -> Arx5Robot.schedule_waypoints -> ARX5 SDK.
#
# Most robot smoothness issues show up around action_chunk postprocess and
# schedule_waypoints timing, not in checkpoint loading.
@click.command()
@click.option("--ckpt", default=DEFAULT_CKPT, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--model", default="X5", show_default=True)
@click.option("--interface", default="can1", show_default=True)
@click.option("--realsense-config", default=None)
@click.option("--usb-config", default=DEFAULT_USB_CONFIG, show_default=True)
@click.option("--usb-device", default=0, show_default=True, type=int)
@click.option("--video-devices", default=None, help="Comma-separated /dev/video indices, e.g. 0,6,12.")
@click.option(
    "--camera-order",
    default="old_dp",
    show_default=True,
    type=click.Choice(["old_dp", "current_collector"]),
    help="ThreeCameraRecorder mapping. old_dp: USB,RS0,RS1. current_collector: RS0,RS1,USB.",
)
@click.option("--width", default=1280, show_default=True, type=int)
@click.option("--height", default=720, show_default=True, type=int)
@click.option("--usb-width", default=640, show_default=True, type=int)
@click.option("--usb-height", default=480, show_default=True, type=int)
@click.option("--camera-fps", default=30, show_default=True, type=int)
@click.option("--frequency", "-f", default=None, type=float, help="Control frequency. Defaults to ckpt task.dataset.target_frequency, then 20Hz.")
@click.option("--inference-steps", default=16, show_default=True, type=int)
@click.option("--action-index", default=0, show_default=True, type=int)
@click.option("--steps-per-inference", default=None, type=int, help="Policy control steps scheduled per inference. Defaults to cfg.n_action_steps.")
@click.option("--submit-extra-steps", default=0, show_default=True, type=int, help="Use extra future actions from action_pred before timestamp filtering, so stale dropped points still leave enough scheduled actions.")
@click.option("--chunk-schedule/--single-action", default=True, show_default=True)
@click.option(
    "--replan-mode",
    default="after_chunk",
    show_default=True,
    type=click.Choice(["receding", "after_chunk"]),
    help="receding replans every chunk duration; after_chunk waits until scheduled chunk nearly finishes before replanning.",
)
@click.option("--boundary-blend-steps", default=3, show_default=True, type=int)
@click.option("--replan-lookahead", default=0.12, show_default=True, type=float, help="Seconds before the submitted chunk ends to start the next inference in after_chunk mode.")
@click.option("--command-mode", default="traj", show_default=True, type=click.Choice(["cmd", "traj"]))
@click.option("--command-latency", default=0.05, show_default=True, type=float)
@click.option("--action-exec-latency", default=0.02, show_default=True, type=float)
@click.option("--preview-time", default=0.05, show_default=True, type=float, help="Minimum SDK command lookahead used by Arx5Robot.")
@click.option(
    "--arm-gain-mode",
    default="default",
    show_default=True,
    type=click.Choice(["default", "damping", "pro"]),
    help="default enables SDK cartesian default kp/kd; pro uses the tuned ARX5 DP deployment gain; damping keeps arm kp=0 with kd only.",
)
@click.option("--arm-kp-scale", default=1.5, show_default=True, type=float)
@click.option("--arm-kd-scale", default=0.5, show_default=True, type=float)
@click.option(
    "--timestamp-mode",
    default="obs",
    show_default=True,
    type=click.Choice(["now", "obs", "compensated"]),
    help="Action timestamp base. compensated selects the action_pred slice that matches current execution time.",
)
@click.option("--policy-log-interval", default=0.5, show_default=True, type=float)
@click.option(
    "--execution-layer",
    default="buffer",
    show_default=True,
    type=click.Choice(["direct", "buffer", "continuous"]),
    help="direct submits chunks to SDK. buffer interpolates into one-point traj commands. continuous streams old-DP style interpolated EEF commands.",
)
@click.option("--buffer-frequency", default=100.0, show_default=True, type=float)
@click.option("--buffer-min-lead-time", default=0.01, show_default=True, type=float)
@click.option("--buffer-blend-time", default=0.12, show_default=True, type=float, help="Seconds to blend a newly inserted policy chunk with the old buffered trajectory.")
@click.option("--continuous-frequency", default=200.0, show_default=True, type=float)
@click.option("--continuous-max-pos-speed", default=0.45, show_default=True, type=float)
@click.option("--continuous-max-rot-speed", default=1.05, show_default=True, type=float)
@click.option("--continuous-replace-future/--continuous-append-future", default=False, show_default=True, help="Replace unexecuted future waypoints instead of always appending new chunks.")
@click.option("--continuous-replace-blend-time", default=0.0, show_default=True, type=float, help="Blend duration used when replacing future waypoints.")
@click.option("--continuous-replace-min-lead-time", default=0.0, show_default=True, type=float, help="Keep this much near-future trajectory untouched when replacing future waypoints.")
@click.option("--tracking-guard/--no-tracking-guard", default=True, show_default=True)
@click.option("--tracking-pos-error-limit", default=0.08, show_default=True, type=float)
@click.option("--tracking-rot-error-limit", default=0.6, show_default=True, type=float)
@click.option(
    "--trajectory-log",
    default=None,
    help="Optional JSONL path for deployment diagnostics, e.g. data_local/policy_logs/run.jsonl.",
)
@click.option("--trajectory-log-interval", default=0.1, show_default=True, type=float)
@click.option("--max-action-pos-step", default=0.008, show_default=True, type=float)
@click.option("--max-action-rot-step", default=0.025, show_default=True, type=float)
@click.option("--max-action-gripper-step", default=0.004, show_default=True, type=float)
@click.option("--action-pos-deadband", default=0.0, show_default=True, type=float)
@click.option("--action-rot-deadband", default=0.0, show_default=True, type=float)
@click.option("--action-gripper-deadband", default=0.0, show_default=True, type=float)
@click.option("--policy-start-hold-time", default=0.5, show_default=True, type=float)
@click.option("--policy-start-max-pos-step", default=0.002, show_default=True, type=float)
@click.option("--policy-start-max-rot-step", default=0.008, show_default=True, type=float)
@click.option("--prepend-current-action/--no-prepend-current-action", default=True, show_default=True)
@click.option(
    "--action-anchor",
    default="actual",
    show_default=True,
    type=click.Choice(["target", "actual"]),
    help="Anchor action safety to the last scheduled target or current actual pose.",
)
@click.option("--disable-action-safety", is_flag=True)
@click.option("--gripper-margin", default=0.0, show_default=True, type=float)
@click.option("--execute", is_flag=True, help="Actually send policy actions to the robot.")
@click.option("--start-policy", is_flag=True, help="Start directly in policy mode.")
@click.option("--reset-to-home-start/--no-reset-to-home-start", default=False, show_default=True)
@click.option(
    "--reset-target",
    default="home",
    show_default=True,
    type=click.Choice(["session", "home"]),
    help="Target used by r in human mode. session returns to the pose captured after startup.",
)
@click.option(
    "--reset-gripper-target",
    default="sdk",
    show_default=True,
    type=click.Choice(["sdk", "session", "current", "open", "close"]),
    help="Gripper target used by r when reset-target=session. sdk/open means fully open, matching SDK reset_to_home.",
)
@click.option("--reset-duration", default=2.0, show_default=True, type=float)
@click.option("--reset-attempts", default=0, show_default=True, type=int, help="0 means retry until reset tolerance is reached.")
@click.option("--reset-settle-time", default=0.35, show_default=True, type=float)
@click.option("--reset-pos-tolerance", default=0.006, show_default=True, type=float)
@click.option("--reset-rot-tolerance", default=0.05, show_default=True, type=float)
@click.option("--reset-joint-tolerance", default=0.08, show_default=True, type=float)
@click.option("--show-preview/--no-show-preview", default=True, show_default=True)
@click.option("--pos-speed", default=0.4, show_default=True, type=float)
@click.option("--rot-speed", default=0.75, show_default=True, type=float)
@click.option("--gripper-speed", default=0.08, show_default=True, type=float)
@click.option("--gripper-safe-torque", default=0.75, show_default=True, type=float, help="Hold gripper if abs(torque) exceeds this while closing. Set <=0 to disable.")
@click.option("--gripper-safe-margin", default=0.002, show_default=True, type=float, help="Extra opening margin used by gripper torque safety hold.")
@click.option("--spacemouse-deadzone", default=0.02, show_default=True, type=float)
@click.option("--spacemouse-smoothing-window", default=0, show_default=True, type=int)
@click.option("--debug-teleop", is_flag=True)
def main(
    ckpt,
    device,
    model,
    interface,
    realsense_config,
    usb_config,
    usb_device,
    video_devices,
    camera_order,
    width,
    height,
    usb_width,
    usb_height,
    camera_fps,
    frequency,
    inference_steps,
    action_index,
    steps_per_inference,
    submit_extra_steps,
    chunk_schedule,
    replan_mode,
    boundary_blend_steps,
    replan_lookahead,
    command_mode,
    command_latency,
    action_exec_latency,
    preview_time,
    arm_gain_mode,
    arm_kp_scale,
    arm_kd_scale,
    timestamp_mode,
    policy_log_interval,
    execution_layer,
    buffer_frequency,
    buffer_min_lead_time,
    buffer_blend_time,
    continuous_frequency,
    continuous_max_pos_speed,
    continuous_max_rot_speed,
    continuous_replace_future,
    continuous_replace_blend_time,
    continuous_replace_min_lead_time,
    tracking_guard,
    tracking_pos_error_limit,
    tracking_rot_error_limit,
    trajectory_log,
    trajectory_log_interval,
    max_action_pos_step,
    max_action_rot_step,
    max_action_gripper_step,
    action_pos_deadband,
    action_rot_deadband,
    action_gripper_deadband,
    policy_start_hold_time,
    policy_start_max_pos_step,
    policy_start_max_rot_step,
    prepend_current_action,
    action_anchor,
    disable_action_safety,
    gripper_margin,
    execute,
    start_policy,
    reset_to_home_start,
    reset_target,
    reset_gripper_target,
    reset_duration,
    reset_attempts,
    reset_settle_time,
    reset_pos_tolerance,
    reset_rot_tolerance,
    reset_joint_tolerance,
    show_preview,
    pos_speed,
    rot_speed,
    gripper_speed,
    gripper_safe_torque,
    gripper_safe_margin,
    spacemouse_deadzone,
    spacemouse_smoothing_window,
    debug_teleop,
):
    """Minimal ARX5 online policy loop.

    Safe default: load the policy and print predicted actions only.
    Pass --execute to send target pose and gripper commands to the robot.
    """

    cv2.setNumThreads(1)
    cfg, policy, torch_device, ckpt_path = load_policy_from_ckpt(
        ckpt_path=ckpt,
        device=device,
        inference_steps=inference_steps,
    )
    if frequency is None:
        frequency = 20.0
        try:
            frequency = float(cfg.task.dataset.target_frequency)
        except Exception:
            pass
    print_policy_summary(cfg, policy, torch_device, ckpt_path)
    print(f"execute: {execute}")
    print(
        "Gripper safety:",
        f"safe_torque={gripper_safe_torque:g}",
        f"safe_margin={gripper_safe_margin:g}",
    )
    print("Controls:")
    print("  human mode: SpaceMouse controls robot")
    print("  c: switch to policy mode")
    print("  h: switch back to human mode")
    print("  r: reset robot in human mode")
    print("  q: quit")

    from arx5_collector.input import SpaceMouseTeleop
    from arx5_collector.robot import Arx5Robot
    from diffusion_policy.real_world.keystroke_counter import KeystrokeCounter
    from pynput.keyboard import KeyCode

    obs_buffer = Arx5ObsBuffer(
        shape_meta=cfg.task.shape_meta,
        n_obs_steps=int(cfg.n_obs_steps),
    )
    dt = 1.0 / float(frequency)
    if steps_per_inference is None:
        steps_per_inference = int(cfg.n_action_steps)
    steps_per_inference = max(1, int(steps_per_inference))
    chunk_duration = steps_per_inference * dt
    if command_latency >= chunk_duration:
        print(
            "WARNING:",
            "command_latency is not smaller than scheduled chunk duration.",
            f"command_latency={command_latency:.3f}s",
            f"chunk_duration={chunk_duration:.3f}s.",
            "On ARX traj mode this can repeatedly overwrite future waypoints before they execute.",
        )
    if execution_layer == "buffer" and command_mode != "traj":
        print(
            "WARNING:",
            "ARX5 buffer execution is tuned for --command-mode traj.",
            f"Current command_mode={command_mode}.",
        )

    trajectory_logger = TrajectoryLogger(
        trajectory_log,
        buffer_sample_interval=trajectory_log_interval,
    )

    with SharedMemoryManager() as shm_manager:
        robot = Arx5Robot(
            model=model,
            interface=interface,
            reset_to_home=False,
            preview_time=preview_time,
            command_mode=command_mode,
            arm_gain_mode=arm_gain_mode,
            arm_kp_scale=arm_kp_scale,
            arm_kd_scale=arm_kd_scale,
            gripper_safe_torque=gripper_safe_torque,
            gripper_safe_margin=gripper_safe_margin,
            tracking_pos_error_limit=tracking_pos_error_limit,
            tracking_rot_error_limit=tracking_rot_error_limit,
        )
        teleop = SpaceMouseTeleop(
            shm_manager=shm_manager,
            pos_speed=pos_speed,
            rot_speed=rot_speed,
            gripper_speed=gripper_speed,
            gripper_margin=gripper_margin,
            deadzone=spacemouse_deadzone,
            smoothing_window=spacemouse_smoothing_window,
        )
        cameras = None
        if video_devices is None:
            from arx5_collector.camera import ThreeCameraRecorder

            cameras = ThreeCameraRecorder(
                shm_manager=shm_manager,
                realsense_config=realsense_config,
                usb_config=usb_config,
                usb_device=usb_device,
                resolution=(width, height),
                usb_resolution=(usb_width, usb_height),
                fps=camera_fps,
            )
        video_reader = None
        if video_devices is not None:
            video_reader = VideoDeviceReader(
                devices=[int(x.strip()) for x in video_devices.split(",") if x.strip()],
                resolution=(usb_width, usb_height),
                fps=camera_fps,
                config_path=usb_config,
            )

        key_counter = None
        action_buffer = ActionTrajectoryBuffer()
        buffered_executor = None
        continuous_executor = None
        try:
            trajectory_logger.open()
            robot.start()
            if execution_layer == "buffer":
                buffered_executor = BufferedActionExecutor(
                    robot=robot,
                    buffer=action_buffer,
                    frequency=buffer_frequency,
                    gripper_margin=gripper_margin,
                    command_latency=action_exec_latency,
                    log_interval=trajectory_log_interval,
                    logger=trajectory_logger,
                    tracking_guard=tracking_guard,
                )
                buffered_executor.start()
            elif execution_layer == "continuous":
                continuous_executor = ContinuousWaypointExecutor(
                    robot=robot,
                    frequency=continuous_frequency,
                    gripper_margin=gripper_margin,
                    command_latency=action_exec_latency,
                    log_interval=trajectory_log_interval,
                    logger=trajectory_logger,
                    tracking_guard=tracking_guard,
                    max_pos_speed=continuous_max_pos_speed,
                    max_rot_speed=continuous_max_rot_speed,
                    replace_future=continuous_replace_future,
                    replace_blend_time=continuous_replace_blend_time,
                    replace_min_lead_time=continuous_replace_min_lead_time,
                )
                continuous_executor.start()
            if reset_to_home_start:
                print("Resetting robot to home...")
                robot.reset_to_home()
            robot_state = robot.get_state()
            gain_summary = robot.get_gain_summary()
            target_pose = robot_state["ActualTCPPose"].copy()
            target_gripper = float(robot_state["gripper_pos"][0])
            session_initial_pose = target_pose.copy()
            session_initial_gripper = target_gripper
            mode = "policy" if start_policy else "human"
            if mode == "policy" and not execute:
                print("Policy mode requested, but --execute is disabled. Running dry-run policy predictions.")
            print(f"Initial mode: {mode}")
            print(
                "Policy timing:",
                f"frequency={frequency:.2f}Hz",
                f"steps_per_inference={steps_per_inference}",
                f"submit_extra_steps={submit_extra_steps}",
                f"chunk_schedule={chunk_schedule}",
                f"replan_mode={replan_mode}",
                f"boundary_blend_steps={boundary_blend_steps}",
                f"execution_layer={execution_layer}",
                f"preview_time={preview_time:.3f}s",
            )
            print(
                "Arm gain:",
                f"mode={arm_gain_mode}",
                f"kp={np.array2string(gain_summary['kp'], precision=3)}",
                f"kd={np.array2string(gain_summary['kd'], precision=3)}",
                f"gripper_kp={gain_summary['gripper_kp']:.3f}",
                f"gripper_kd={gain_summary['gripper_kd']:.3f}",
            )
            if execution_layer == "buffer":
                print(
                    "Local trajectory buffer:",
                    f"frequency={buffer_frequency:.1f}Hz",
                    f"min_lead_time={buffer_min_lead_time:.3f}s",
                    f"blend_time={buffer_blend_time:.3f}s",
                    f"tracking_guard={tracking_guard}",
                )
            elif execution_layer == "continuous":
                print(
                    "Continuous executor:",
                    f"frequency={continuous_frequency:.1f}Hz",
                    f"max_pos_speed={continuous_max_pos_speed:.3f}",
                    f"max_rot_speed={continuous_max_rot_speed:.3f}",
                    f"replace_future={continuous_replace_future}",
                    f"replace_blend_time={continuous_replace_blend_time:.3f}s",
                    f"replace_min_lead_time={continuous_replace_min_lead_time:.3f}s",
                    f"tracking_guard={tracking_guard}",
                )
            if trajectory_logger.enabled:
                print(f"Trajectory log: {trajectory_logger.path}")
            if disable_action_safety:
                print("Action safety: disabled")
            else:
                print(
                    "Action safety:",
                    f"max_pos_step={max_action_pos_step}",
                    f"max_rot_step={max_action_rot_step}",
                    f"max_gripper_step={max_action_gripper_step}",
                    f"pos_deadband={action_pos_deadband}",
                    f"rot_deadband={action_rot_deadband}",
                    f"gripper_deadband={action_gripper_deadband}",
                    f"anchor={action_anchor}",
                )
            print(
                "Reset target:",
                f"{reset_target}",
                f"gripper={reset_gripper_target}",
                f"session_pose={np.array2string(session_initial_pose, precision=4)}",
                f"session_gripper={session_initial_gripper:.5f}",
            )
            if video_reader is None:
                print(f"Camera order: {camera_order}")
            else:
                print("Camera order: explicit --video-devices order maps to camera_0,camera_1,camera_2")
            teleop.start()
            if video_reader is not None:
                video_reader.start()
                print(f"Video device mode: /dev/video{video_devices.replace(',', ', /dev/video')}")
            else:
                cameras.start()
            key_counter = KeystrokeCounter()
            key_counter.start()
            print("Warming up cameras and observation buffer...")
            time.sleep(1.0)

            rs_cache = None
            if hasattr(policy, "reset"):
                policy.reset()

            t_loop = time.monotonic()
            next_policy_time = 0.0
            policy_start_time = time.monotonic() if mode == "policy" else None
            last_debug_print = 0.0
            last_policy_print = 0.0

            def perform_reset(reason):
                nonlocal mode, target_pose, target_gripper, next_policy_time, policy_start_time
                if buffered_executor is not None:
                    buffered_executor.disable()
                if continuous_executor is not None:
                    continuous_executor.disable()
                    continuous_executor.clear()
                action_buffer.clear()
                mode = "human"
                next_policy_time = 0.0
                policy_start_time = None
                trajectory_logger.log("reset_start", reset_target=reset_target, reason=reason)
                print(f"Resetting robot to {reset_target} ({reason})...")
                if reset_target == "home":
                    robot_state, reset_ok = reset_to_home_checked(
                        robot,
                        attempts=reset_attempts,
                        settle_time=reset_settle_time,
                        pos_tolerance=reset_pos_tolerance,
                        rot_tolerance=reset_rot_tolerance,
                        joint_tolerance=reset_joint_tolerance,
                    )
                else:
                    current_reset_state = robot.get_state()
                    if reset_gripper_target in ("sdk", "open"):
                        reset_gripper = float(robot.robot_config.gripper_width)
                    elif reset_gripper_target == "close":
                        reset_gripper = 0.0
                    elif reset_gripper_target == "current":
                        reset_gripper = float(current_reset_state["gripper_pos"][0])
                    else:
                        reset_gripper = float(session_initial_gripper)
                    robot_state, reset_ok = reset_to_pose(
                        robot,
                        pose=session_initial_pose,
                        gripper=reset_gripper,
                        duration=reset_duration,
                        frequency=max(frequency, 20.0),
                        command_latency=command_latency,
                        attempts=reset_attempts,
                        settle_time=reset_settle_time,
                        pos_tolerance=reset_pos_tolerance,
                        rot_tolerance=reset_rot_tolerance,
                    )
                target_pose = robot_state["ActualTCPPose"].copy()
                target_gripper = float(robot_state["gripper_pos"][0])
                trajectory_logger.log(
                    "reset_finish",
                    target_pose=target_pose,
                    target_gripper=target_gripper,
                    reset_ok=reset_ok,
                    reason=reason,
                )
                if reset_ok:
                    print(f"Robot reset to {reset_target}.")
                else:
                    print(
                        f"WARNING: reset to {reset_target} did not reach tolerance "
                        f"after {reset_attempts} attempts."
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
                        camera_order=camera_order,
                    )
                robot_state = robot.get_state()
                obs_buffer.append(camera_frames=camera_frames, robot_state=robot_state)

                requested_key = None
                for key_stroke in key_counter.get_press_events():
                    if key_stroke == KeyCode(char="q"):
                        requested_key = "q"
                    elif key_stroke == KeyCode(char="c"):
                        requested_key = "c"
                    elif key_stroke == KeyCode(char="h"):
                        requested_key = "h"
                    elif key_stroke == KeyCode(char="r"):
                        requested_key = "r"

                key = -1
                if show_preview:
                    preview = make_preview(camera_frames)
                    status = f"{mode.upper()} {'EXECUTE' if execute else 'DRY-RUN'}"
                    cv2.putText(
                        preview,
                        status,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("arx5_policy_preview", preview)
                    key = cv2.pollKey()
                else:
                    key = cv2.pollKey()

                if key == ord("q"):
                    requested_key = "q"
                elif key == ord("c"):
                    requested_key = "c"
                elif key == ord("h"):
                    requested_key = "h"
                elif key == ord("r"):
                    requested_key = "r"

                if requested_key == "q":
                    trajectory_logger.log("key", key="q", mode=mode)
                    perform_reset("quit")
                    break
                if requested_key == "c":
                    mode = "policy"
                    robot_state = robot.get_state()
                    target_pose = robot_state["ActualTCPPose"].copy()
                    target_gripper = float(robot_state["gripper_pos"][0])
                    if execute:
                        if execution_layer == "buffer":
                            hold_action = np.concatenate([
                                target_pose,
                                np.asarray([target_gripper], dtype=np.float64),
                            ])
                            action_buffer.set_hold(hold_action)
                            buffered_executor.enable()
                        elif execution_layer == "continuous":
                            hold_action = np.concatenate([
                                target_pose,
                                np.asarray([target_gripper], dtype=np.float64),
                            ])
                            continuous_executor.set_hold(hold_action)
                            continuous_executor.enable()
                        else:
                            robot.schedule_waypoint(
                                target_pose,
                                target_time=time.time() + command_latency,
                                gripper_pos=target_gripper,
                            )
                    if hasattr(policy, "reset"):
                        policy.reset()
                    obs_buffer.clear()
                    next_policy_time = 0.0
                    policy_start_time = time.monotonic()
                    trajectory_logger.log(
                        "mode_switch",
                        mode="policy",
                        target_pose=target_pose,
                        target_gripper=target_gripper,
                    )
                    print(
                        "Switched to policy mode.",
                        "Holding current pose while refilling observations.",
                    )
                elif requested_key == "h":
                    if buffered_executor is not None:
                        buffered_executor.disable()
                    if continuous_executor is not None:
                        continuous_executor.disable()
                        continuous_executor.clear()
                    action_buffer.clear()
                    mode = "human"
                    robot_state = robot.get_state()
                    target_pose = robot_state["ActualTCPPose"].copy()
                    target_gripper = float(robot_state["gripper_pos"][0])
                    next_policy_time = 0.0
                    policy_start_time = None
                    trajectory_logger.log(
                        "mode_switch",
                        mode="human",
                        target_pose=target_pose,
                        target_gripper=target_gripper,
                    )
                    print("Switched to human mode.")
                elif requested_key == "r":
                    perform_reset("manual")

                if mode == "human":
                    command = teleop.update(
                        target_pose=target_pose,
                        target_gripper_pos=target_gripper,
                        dt=dt,
                        gripper_width=robot.robot_config.gripper_width,
                    )
                    target_pose = command.action
                    target_gripper = command.gripper_action
                    now_mono = time.monotonic()
                    if debug_teleop and (
                        np.linalg.norm(command.raw_motion) > 1e-3
                        or now_mono - last_debug_print > 1.0
                    ):
                        last_debug_print = now_mono
                        actual_pose = robot_state["ActualTCPPose"]
                        print(
                            "spacemouse",
                            np.array2string(command.raw_motion, precision=3),
                            "actual",
                            np.array2string(actual_pose, precision=4),
                            "target",
                            np.array2string(target_pose, precision=4),
                            f"gripper={target_gripper:.5f}",
                        )
                    robot.schedule_waypoint(
                        target_pose,
                        target_time=time.time() + command_latency,
                        gripper_pos=target_gripper,
                    )
                    if debug_teleop and now_mono - last_debug_print < 0.05:
                        post_cmd_state = robot.get_state()
                        print(
                            "sdk_target",
                            np.array2string(post_cmd_state["TargetTCPPose"], precision=4),
                            f"sdk_gripper={post_cmd_state['target_gripper_pos'][0]:.5f}",
                        )
                elif not obs_buffer.is_ready:
                    print(f"buffering obs {len(obs_buffer.frames)}/{cfg.n_obs_steps}")
                else:
                    now_mono = time.monotonic()
                    in_policy_start_hold = (
                        policy_start_time is not None
                        and now_mono - policy_start_time < policy_start_hold_time
                    )
                    if in_policy_start_hold:
                        if execute:
                            if execution_layer == "buffer":
                                hold_action = np.concatenate([
                                    np.asarray(target_pose, dtype=np.float64),
                                    np.asarray([target_gripper], dtype=np.float64),
                                ])
                                action_buffer.set_hold(hold_action)
                                if buffered_executor is not None:
                                    buffered_executor.enable()
                            elif execution_layer == "continuous":
                                hold_action = np.concatenate([
                                    np.asarray(target_pose, dtype=np.float64),
                                    np.asarray([target_gripper], dtype=np.float64),
                                ])
                                continuous_executor.set_hold(hold_action)
                                if continuous_executor is not None:
                                    continuous_executor.enable()
                            else:
                                robot.schedule_waypoint(
                                    target_pose,
                                    target_time=time.time() + command_latency,
                                    gripper_pos=target_gripper,
                                )
                        if now_mono - last_policy_print >= policy_log_interval:
                            last_policy_print = now_mono
                            remaining = policy_start_hold_time - (now_mono - policy_start_time)
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
                    if now_mono < next_policy_time:
                        t_loop += dt
                        sleep_time = t_loop - time.monotonic()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        else:
                            t_loop = loop_start
                        continue
                    obs_dict = obs_buffer.as_policy_input(torch_device)
                    infer_start = time.time()
                    with torch.no_grad():
                        result = policy.predict_action(obs_dict)
                    infer_latency = time.time() - infer_start
                    action_sequence = get_policy_action_sequence(result, cfg)
                    if chunk_schedule:
                        curr_time = time.time()
                        obs_timestamps = obs_buffer.get_timestamps()
                        candidate = make_timed_action_candidate(
                            action_sequence=action_sequence,
                            obs_timestamps=obs_timestamps,
                            curr_time=curr_time,
                            dt=dt,
                            steps_per_inference=steps_per_inference,
                            submit_extra_steps=submit_extra_steps,
                            command_latency=command_latency,
                            action_exec_latency=action_exec_latency,
                            timestamp_mode=timestamp_mode,
                        )
                        action_chunk = candidate.action_chunk.copy()
                        action_timestamps = candidate.action_timestamps.copy()
                        raw_action_chunk = action_chunk.copy()
                        anchor_action = make_anchor_action(
                            robot_state=robot_state,
                            target_pose=target_pose,
                            target_gripper=target_gripper,
                            action_anchor=action_anchor,
                        )
                        action_chunk[:, 6] = [
                            clamp_gripper(
                                value,
                                width=robot.robot_config.gripper_width,
                                margin=gripper_margin,
                            )
                            for value in action_chunk[:, 6]
                        ]
                        if prepend_current_action and len(action_chunk) > 0:
                            action_chunk[0] = anchor_action
                            if boundary_blend_steps > 0 and len(action_chunk) > 1:
                                blended_tail = blend_chunk_start(
                                    action_chunk[1:],
                                    anchor_action=anchor_action,
                                    blend_steps=boundary_blend_steps,
                                )
                                action_chunk[1:] = blended_tail
                        elif boundary_blend_steps > 0:
                            action_chunk = blend_chunk_start(
                                action_chunk,
                                anchor_action=anchor_action,
                                blend_steps=boundary_blend_steps,
                            )
                        if not disable_action_safety:
                            in_policy_start = (
                                policy_start_time is not None
                                and time.monotonic() - policy_start_time < policy_start_hold_time
                            )
                            this_max_pos_step = max_action_pos_step
                            this_max_rot_step = max_action_rot_step
                            if in_policy_start:
                                if this_max_pos_step is None:
                                    this_max_pos_step = policy_start_max_pos_step
                                else:
                                    this_max_pos_step = min(
                                        this_max_pos_step,
                                        policy_start_max_pos_step,
                                    )
                                if this_max_rot_step is None:
                                    this_max_rot_step = policy_start_max_rot_step
                                else:
                                    this_max_rot_step = min(
                                        this_max_rot_step,
                                        policy_start_max_rot_step,
                                    )
                            action_chunk, action_clipped = clamp_action_chunk_delta(
                                action_chunk,
                                anchor_action=anchor_action,
                                max_pos_step=this_max_pos_step,
                                max_rot_step=this_max_rot_step,
                                max_gripper_step=max_action_gripper_step,
                            )
                        else:
                            action_clipped = False
                        action_chunk, deadband_applied = apply_action_deadband(
                            action_chunk,
                            anchor_action=anchor_action,
                            pos_deadband=action_pos_deadband,
                            rot_deadband=action_rot_deadband,
                            gripper_deadband=action_gripper_deadband,
                        )
                        action_chunk, action_timestamps, over_budget = filter_future_actions(
                            action_chunk=action_chunk,
                            action_timestamps=action_timestamps,
                            curr_time=curr_time,
                            action_exec_latency=action_exec_latency,
                            command_latency=command_latency,
                            dt=dt,
                        )
                        if over_budget:
                            print(
                                "policy over budget;",
                                f"scheduling latest action at +{action_timestamps[0] - curr_time:.3f}s",
                            )
                        alignment_summary = summarize_policy_alignment(
                            obs_timestamps=obs_timestamps,
                            raw_action_chunk=raw_action_chunk,
                            action_chunk=action_chunk,
                            action_timestamps=action_timestamps,
                            curr_time=curr_time,
                        )
                        alignment_summary.update(
                            {
                                "timestamp_mode": timestamp_mode,
                                "action_start_idx": int(candidate.action_start_idx),
                                "submit_steps": int(candidate.submit_steps),
                                "timestamp_base": float(candidate.timestamp_base),
                                "replace_future": bool(continuous_replace_future),
                            }
                        )
                        trajectory_logger.log(
                            "policy_chunk",
                            timestamp=curr_time,
                            execution_layer=execution_layer,
                            raw_chunk=raw_action_chunk,
                            action_chunk=action_chunk,
                            action_timestamps=action_timestamps,
                            actual_pose=robot_state["ActualTCPPose"],
                            actual_gripper=robot_state["gripper_pos"],
                            target_pose=target_pose,
                            target_gripper=target_gripper,
                            clipped=action_clipped,
                            deadband=deadband_applied,
                            alignment=alignment_summary,
                            infer_latency=infer_latency,
                        )
                        now_print = time.monotonic()
                        if now_print - last_policy_print >= policy_log_interval:
                            last_policy_print = now_print
                            print(
                                "policy chunk",
                                f"n={len(action_chunk)}",
                                f"dt0={action_timestamps[0] - curr_time:.3f}s",
                                f"dtN={action_timestamps[-1] - curr_time:.3f}s",
                                f"obs_age={alignment_summary['obs_latest_age']:.3f}s",
                                f"dropped={alignment_summary['dropped_action_len']}",
                                f"submit_extra={submit_extra_steps}",
                                f"start_idx={candidate.action_start_idx}",
                                f"clipped={action_clipped}",
                                f"deadband={deadband_applied}",
                                f"infer={infer_latency:.3f}s",
                                f"pose0={np.array2string(action_chunk[0, :6], precision=4)}",
                                f"gripper_minmax={action_chunk[:, 6].min():.5f}/{action_chunk[:, 6].max():.5f}",
                            )
                        if execute:
                            if execution_layer == "buffer":
                                current_action = np.concatenate(
                                    [
                                        np.asarray(robot_state["ActualTCPPose"], dtype=np.float64),
                                        np.asarray([robot_state["gripper_pos"][0]], dtype=np.float64),
                                    ]
                                )
                                inserted = action_buffer.add_chunk(
                                    action_chunk,
                                    action_timestamps,
                                    current_action=current_action,
                                    now=curr_time,
                                    min_lead_time=buffer_min_lead_time,
                                    blend_time=buffer_blend_time,
                                )
                                if buffered_executor is not None:
                                    buffered_executor.enable()
                                if inserted == 0:
                                    print("buffer skipped stale chunk; waiting for next policy output")
                                trajectory_logger.log(
                                    "buffer_insert",
                                    timestamp=curr_time,
                                    inserted=inserted,
                                    chunk_len=len(action_chunk),
                                    horizon=action_buffer.horizon(curr_time),
                                )
                            elif execution_layer == "continuous":
                                inserted = continuous_executor.add_chunk(
                                    action_chunk,
                                    action_timestamps,
                                    now=curr_time,
                                )
                                if continuous_executor is not None:
                                    continuous_executor.enable()
                                if inserted == 0:
                                    print("continuous skipped stale chunk; waiting for next policy output")
                                trajectory_logger.log(
                                    "continuous_insert",
                                    timestamp=curr_time,
                                    inserted=inserted,
                                    chunk_len=len(action_chunk),
                                )
                            else:
                                robot.schedule_waypoints(
                                    action_chunk,
                                    action_timestamps,
                                    gripper_margin=gripper_margin,
                                )
                            target_pose = action_chunk[-1, :6].copy()
                            target_gripper = float(action_chunk[-1, 6])
                        if replan_mode == "after_chunk":
                            next_policy_time = (
                                time.monotonic()
                                + max(0.0, action_timestamps[-1] - curr_time)
                                - max(action_exec_latency, replan_lookahead)
                            )
                        else:
                            next_policy_time = time.monotonic() + steps_per_inference * dt
                    else:
                        selected = select_action(action_sequence, index=action_index)
                        gripper = clamp_gripper(
                            selected.gripper,
                            width=robot.robot_config.gripper_width,
                            margin=gripper_margin,
                        )
                        print(
                            "policy action",
                            f"pose={np.array2string(selected.pose, precision=4)}",
                            f"gripper={gripper:.5f}",
                        )

                        if execute:
                            robot.schedule_waypoint(
                                selected.pose,
                                target_time=time.time() + command_latency,
                                gripper_pos=gripper,
                            )
                            target_pose = selected.pose.copy()
                            target_gripper = gripper
                        trajectory_logger.log(
                            "policy_action",
                            pose=selected.pose,
                            gripper=gripper,
                            execute=execute,
                        )
                        next_policy_time = time.monotonic() + dt

                t_loop += dt
                sleep_time = t_loop - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    t_loop = loop_start
        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            if buffered_executor is not None:
                buffered_executor.stop()
            if continuous_executor is not None:
                continuous_executor.stop()
            if key_counter is not None:
                key_counter.stop()
                key_counter.join()
            teleop.stop()
            if video_reader is not None:
                video_reader.stop()
            elif cameras is not None:
                cameras.stop()
            robot.stop()
            trajectory_logger.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
