import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import click
import cv2
import numpy as np

from arx5_act.deployment.joint_robot import JointActRobot
from arx5_ckpt_loader.action_adapter import clamp_gripper
from arx5_ckpt_loader.deployment.action_scheduler import (
    filter_future_actions,
    get_policy_action_sequence,
    make_timed_action_candidate,
)
from arx5_ckpt_loader.deployment.joint_continuous_executor import JointContinuousWaypointExecutor
from arx5_ckpt_loader.deployment.reset import reset_to_home_checked
from arx5_ckpt_loader.obs_buffer import Arx5ObsBuffer, make_camera_frame_dict
from arx5_ckpt_loader.policy_loader import load_policy_from_ckpt, print_policy_summary
from arx5_ckpt_loader.video_device_reader import VideoDeviceReader
from arx5_collector.camera import ThreeCameraRecorder
from diffusion_policy.real_world.keystroke_counter import KeystrokeCounter
from pynput.keyboard import KeyCode


DEFAULT_JOINT_CKPT = (
    "data/outputs/manual/glue_clean_three_200/dp_joint/checkpoints/latest.ckpt"
)


def _resolve_config_path(path):
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return str(p)
    dp_root = Path(__file__).resolve().parents[1]
    return str(dp_root / p)


def _preview(camera_frames):
    frames = []
    for key in sorted(camera_frames.keys()):
        frames.append(cv2.resize(camera_frames[key], (426, 240), interpolation=cv2.INTER_AREA))
    return np.concatenate(frames, axis=1)


def _requested_key(key_counter, cv_key):
    requested = None
    for key_stroke in key_counter.get_press_events():
        if key_stroke == KeyCode(char="q"):
            requested = "q"
        elif key_stroke == KeyCode(char="c"):
            requested = "c"
        elif key_stroke == KeyCode(char="h"):
            requested = "h"
        elif key_stroke == KeyCode(char="r"):
            requested = "r"
    if cv_key in (ord("q"), ord("c"), ord("h"), ord("r")):
        requested = chr(cv_key)
    return requested


def _clamp_joint_chunk_delta(action_chunk, anchor_action, max_joint_step, max_gripper_step):
    action_chunk = np.asarray(action_chunk, dtype=np.float64).copy()
    anchor_action = np.asarray(anchor_action, dtype=np.float64)
    clipped = False
    if max_joint_step is not None and float(max_joint_step) > 0:
        delta = action_chunk[:, :6] - anchor_action[:6]
        clipped_delta = np.clip(delta, -float(max_joint_step), float(max_joint_step))
        clipped = clipped or bool(np.any(np.abs(clipped_delta - delta) > 1e-12))
        action_chunk[:, :6] = anchor_action[:6] + clipped_delta
    if max_gripper_step is not None and float(max_gripper_step) > 0:
        delta = action_chunk[:, 6] - anchor_action[6]
        clipped_delta = np.clip(delta, -float(max_gripper_step), float(max_gripper_step))
        clipped = clipped or bool(np.any(np.abs(clipped_delta - delta) > 1e-12))
        action_chunk[:, 6] = anchor_action[6] + clipped_delta
    return action_chunk, clipped


@click.command()
@click.option("--ckpt", default=DEFAULT_JOINT_CKPT, show_default=True)
@click.option("--model", default="X5", show_default=True)
@click.option("--interface", default="can1", show_default=True)
@click.option("--usb-device", default=0, show_default=True, type=int)
@click.option("--video-devices", default=None, help="Comma-separated /dev/video indices, e.g. 0,6,12.")
@click.option("--realsense-config", default=None)
@click.option("--usb-config", default="scripts/camera/usb/current.yaml", show_default=True)
@click.option("--width", default=1280, show_default=True, type=int)
@click.option("--height", default=720, show_default=True, type=int)
@click.option("--usb-width", default=640, show_default=True, type=int)
@click.option("--usb-height", default=480, show_default=True, type=int)
@click.option("--camera-fps", default=30, show_default=True, type=int)
@click.option("--camera-order", default="old_dp", show_default=True, type=click.Choice(["old_dp", "current_collector"]))
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--inference-steps", default=16, show_default=True, type=int)
@click.option("--frequency", default=None, type=float)
@click.option("--steps-per-inference", default=8, show_default=True, type=int)
@click.option("--submit-extra-steps", default=0, show_default=True, type=int)
@click.option("--replan-lookahead", default=0.2, show_default=True, type=float)
@click.option("--timestamp-mode", default="compensated", show_default=True, type=click.Choice(["now", "obs", "compensated"]))
@click.option("--command-mode", default="traj", show_default=True, type=click.Choice(["cmd", "traj"]))
@click.option("--command-latency", default=0.01, show_default=True, type=float)
@click.option("--action-exec-latency", default=0.01, show_default=True, type=float)
@click.option("--preview-time", default=0.05, show_default=True, type=float)
@click.option("--continuous-frequency", default=200.0, show_default=True, type=float)
@click.option("--continuous-replace-blend-time", default=0.08, show_default=True, type=float)
@click.option("--continuous-replace-min-lead-time", default=0.06, show_default=True, type=float)
@click.option("--continuous-replace-future/--continuous-append-future", default=False, show_default=True)
@click.option("--arm-gain-mode", default="pro", show_default=True, type=click.Choice(["default", "damping", "pro"]))
@click.option("--arm-kp-scale", default=1.0, show_default=True, type=float)
@click.option("--arm-kd-scale", default=1.0, show_default=True, type=float)
@click.option("--max-action-joint-step", default=None, type=float)
@click.option("--max-action-gripper-step", default=None, type=float)
@click.option("--prepend-current-action/--no-prepend-current-action", default=True, show_default=True)
@click.option("--gripper-margin", default=0.0, show_default=True, type=float)
@click.option("--close-gate-delay", default=0.0, show_default=True, type=float)
@click.option("--close-gate-threshold", default=0.003, show_default=True, type=float)
@click.option("--close-gate-release-width", default=0.04, show_default=True, type=float)
@click.option("--gripper-safe-torque", default=0.75, show_default=True, type=float)
@click.option("--gripper-safe-margin", default=0.002, show_default=True, type=float)
@click.option("--reset-attempts", default=0, show_default=True, type=int)
@click.option("--reset-duration", default=2.0, show_default=True, type=float)
@click.option("--reset-hold-time", default=0.3, show_default=True, type=float)
@click.option("--reset-settle-time", default=0.35, show_default=True, type=float)
@click.option("--reset-joint-tolerance", default=0.08, show_default=True, type=float)
@click.option(
    "--reset-mode",
    default="hold_sdk_home",
    show_default=True,
    type=click.Choice(["session", "sdk_home", "hold_sdk_home"]),
)
@click.option("--policy-start-hold-time", default=0.4, show_default=True, type=float)
@click.option("--policy-log-interval", default=0.4, show_default=True, type=float)
@click.option("--show-preview/--no-show-preview", default=True, show_default=True)
@click.option("--execute", is_flag=True)
@click.option("--start-policy", is_flag=True)
@click.option("--debug-policy-chunk", is_flag=True)
def main(
    ckpt,
    model,
    interface,
    usb_device,
    video_devices,
    realsense_config,
    usb_config,
    width,
    height,
    usb_width,
    usb_height,
    camera_fps,
    camera_order,
    device,
    inference_steps,
    frequency,
    steps_per_inference,
    submit_extra_steps,
    replan_lookahead,
    timestamp_mode,
    command_mode,
    command_latency,
    action_exec_latency,
    preview_time,
    continuous_frequency,
    continuous_replace_blend_time,
    continuous_replace_min_lead_time,
    continuous_replace_future,
    arm_gain_mode,
    arm_kp_scale,
    arm_kd_scale,
    max_action_joint_step,
    max_action_gripper_step,
    prepend_current_action,
    gripper_margin,
    close_gate_delay,
    close_gate_threshold,
    close_gate_release_width,
    gripper_safe_torque,
    gripper_safe_margin,
    reset_attempts,
    reset_duration,
    reset_hold_time,
    reset_settle_time,
    reset_joint_tolerance,
    reset_mode,
    policy_start_hold_time,
    policy_log_interval,
    show_preview,
    execute,
    start_policy,
    debug_policy_chunk,
):
    cv2.setNumThreads(1)
    cfg, policy, torch_device, ckpt_path = load_policy_from_ckpt(
        ckpt_path=ckpt,
        device=device,
        inference_steps=inference_steps,
    )
    print_policy_summary(cfg, policy, torch_device, ckpt_path)
    if cfg.task.name != "arx5_joint_image":
        raise RuntimeError(f"This runner only supports arx5_joint_image ckpts, got {cfg.task.name!r}.")

    if frequency is None:
        frequency = float(cfg.task.dataset.get("target_frequency", 20.0))
    dt = 1.0 / float(frequency)
    shape_meta = cfg.task.shape_meta
    n_obs_steps = int(cfg.n_obs_steps)

    print("DP-Joint mode: action=[q1..q6, gripper], controller=Arx5JointController")
    print("Controls: c=policy, h=hold human, r=reset home, q=reset home and quit")
    print(
        "Policy timing:",
        f"frequency={frequency:.2f}Hz",
        f"steps_per_inference={steps_per_inference}",
        f"submit_extra_steps={submit_extra_steps}",
        f"replan_lookahead={replan_lookahead:.3f}s",
        f"timestamp_mode={timestamp_mode}",
        f"replace_future={continuous_replace_future}",
        f"execute={execute}",
    )

    with SharedMemoryManager() as shm_manager:
        robot = JointActRobot(
            model=model,
            interface=interface,
            preview_time=preview_time,
            command_mode=command_mode,
            arm_gain_mode=arm_gain_mode,
            arm_kp_scale=arm_kp_scale,
            arm_kd_scale=arm_kd_scale,
            gripper_safe_torque=gripper_safe_torque,
            gripper_safe_margin=gripper_safe_margin,
        )
        cameras = None
        video_reader = None
        if video_devices is None:
            cameras = ThreeCameraRecorder(
                shm_manager=shm_manager,
                realsense_config=_resolve_config_path(realsense_config),
                usb_device=usb_device,
                resolution=(width, height),
                usb_resolution=(usb_width, usb_height),
                fps=camera_fps,
                usb_config=_resolve_config_path(usb_config),
            )
        else:
            video_reader = VideoDeviceReader(
                devices=[int(x.strip()) for x in video_devices.split(",") if x.strip()],
                resolution=(usb_width, usb_height),
                fps=camera_fps,
                config_path=_resolve_config_path(usb_config),
            )
        key_counter = None
        try:
            robot.start()
            continuous_executor = JointContinuousWaypointExecutor(
                robot=robot,
                frequency=continuous_frequency,
                gripper_margin=gripper_margin,
                command_latency=command_latency,
                replace_blend_time=continuous_replace_blend_time,
                replace_min_lead_time=continuous_replace_min_lead_time,
                replace_future=continuous_replace_future,
            )
            continuous_executor.start()
            if cameras is not None:
                cameras.start()
            if video_reader is not None:
                video_reader.start()
            key_counter = KeystrokeCounter()
            key_counter.start()
            obs_buffer = Arx5ObsBuffer(shape_meta=shape_meta, n_obs_steps=n_obs_steps)
            mode = "policy" if start_policy else "human"
            target_state = robot.get_state()
            target_joints = target_state["ActualQ"].copy()
            target_gripper = float(target_state["gripper_pos"][0])
            session_initial_joints = target_joints.copy()
            session_initial_gripper = target_gripper
            policy_start_time = time.monotonic() if mode == "policy" else None
            next_policy_time = 0.0
            last_policy_print = 0.0
            close_gate_until = 0.0
            close_gate_fired = False
            rs_cache = None
            t_loop = time.monotonic()
            print(f"Initial mode: {mode}")
            time.sleep(1.0)

            def reset_home(reason):
                nonlocal mode, target_joints, target_gripper, policy_start_time, next_policy_time, close_gate_until, close_gate_fired
                continuous_executor.disable()
                continuous_executor.clear()
                mode = "human"
                policy_start_time = None
                next_policy_time = 0.0
                close_gate_until = 0.0
                close_gate_fired = False
                obs_buffer.clear()
                print(f"Resetting robot to {reset_mode} ({reason})...")
                current_state = robot.get_state()
                current_joints = current_state["ActualQ"].copy()
                current_gripper = float(current_state["gripper_pos"][0])
                if reset_mode == "hold_sdk_home":
                    hold_until = time.time() + max(0.0, float(reset_hold_time))
                    while time.time() < hold_until:
                        robot.schedule_waypoint(
                            current_joints,
                            target_time=time.time() + max(command_latency, 0.05),
                            gripper_pos=current_gripper,
                        )
                        time.sleep(0.05)
                    obs_buffer.clear()
                    state, ok = reset_to_home_checked(
                        robot,
                        attempts=reset_attempts,
                        settle_time=reset_settle_time,
                        joint_tolerance=reset_joint_tolerance,
                    )
                elif reset_mode == "session":
                    attempts = int(reset_attempts)
                    attempt_idx = 0
                    state = current_state
                    ok = False
                    while True:
                        start_state = robot.get_state()
                        start_joints = start_state["ActualQ"].copy()
                        start_gripper = float(start_state["gripper_pos"][0])
                        steps = max(2, int(max(20.0, frequency) * float(reset_duration)))
                        reset_dt = float(reset_duration) / float(steps)
                        for step_idx in range(steps):
                            alpha = float(step_idx + 1) / float(steps)
                            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                            interp_joints = (1.0 - alpha) * start_joints + alpha * session_initial_joints
                            interp_gripper = (1.0 - alpha) * start_gripper + alpha * session_initial_gripper
                            robot.send_joint_cmd(
                                interp_joints,
                                target_time=time.time() + max(command_latency, reset_dt),
                                gripper_pos=float(interp_gripper),
                            )
                            time.sleep(reset_dt)
                        hold_until = time.time() + max(0.0, float(reset_settle_time))
                        while time.time() < hold_until:
                            robot.send_joint_cmd(
                                session_initial_joints,
                                target_time=time.time() + max(command_latency, 0.05),
                                gripper_pos=session_initial_gripper,
                            )
                            time.sleep(0.05)
                        state = robot.get_state()
                        joint_error = float(np.linalg.norm(state["ActualQ"] - session_initial_joints))
                        print(
                            "reset streamed joint check",
                            f"attempt={attempt_idx + 1}/{'until_success' if attempts <= 0 else attempts}",
                            f"joint_error={joint_error:.5f}",
                        )
                        ok = reset_joint_tolerance is None or joint_error <= float(reset_joint_tolerance)
                        if ok:
                            break
                        attempt_idx += 1
                        if attempts > 0 and attempt_idx >= attempts:
                            break
                else:
                    state, ok = reset_to_home_checked(
                        robot,
                        attempts=reset_attempts,
                        settle_time=reset_settle_time,
                        joint_tolerance=reset_joint_tolerance,
                    )
                target_joints = state["ActualQ"].copy()
                target_gripper = float(state["gripper_pos"][0])
                continuous_executor.set_hold(
                    np.concatenate([target_joints[:6], np.asarray([target_gripper])])
                )
                hold_until = time.time() + max(0.0, float(reset_hold_time))
                while time.time() < hold_until:
                    robot.schedule_waypoint(
                        target_joints,
                        target_time=time.time() + max(command_latency, 0.05),
                        gripper_pos=target_gripper,
                    )
                    time.sleep(0.05)
                obs_buffer.clear()
                print("Robot reset finished." if ok else "WARNING: reset did not reach tolerance.")

            while True:
                loop_start = time.monotonic()
                if cameras is not None:
                    rs_cache = cameras.realsense.get(out=rs_cache)
                    usb_data = cameras.pump_usb_frame()
                    camera_frames = make_camera_frame_dict(
                        rs_cache,
                        usb_data,
                        camera_order=camera_order,
                    )
                else:
                    camera_frames = {
                        key: value["color"]
                        for key, value in video_reader.get().items()
                    }
                robot_state = robot.get_state()
                obs_buffer.append(camera_frames, robot_state)

                cv_key = -1
                if show_preview:
                    preview = _preview(camera_frames)
                    cv2.putText(
                        preview,
                        f"DP-JOINT {mode.upper()} {'EXECUTE' if execute else 'DRY-RUN'}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("arx5_dp_joint_policy_preview", preview)
                    cv_key = cv2.pollKey()
                else:
                    cv_key = cv2.pollKey()

                requested = _requested_key(key_counter, cv_key)
                if requested == "q":
                    reset_home("quit")
                    break
                if requested == "r":
                    reset_home("manual")
                    t_loop = time.monotonic()
                    continue
                elif requested == "h":
                    continuous_executor.disable()
                    continuous_executor.clear()
                    mode = "human"
                    policy_start_time = None
                    next_policy_time = 0.0
                    close_gate_until = 0.0
                    close_gate_fired = False
                    target_joints = robot_state["ActualQ"].copy()
                    target_gripper = float(robot_state["gripper_pos"][0])
                    obs_buffer.clear()
                    print("Switched to human hold mode.")
                elif requested == "c":
                    mode = "policy"
                    target_joints = robot_state["ActualQ"].copy()
                    target_gripper = float(robot_state["gripper_pos"][0])
                    continuous_executor.clear()
                    continuous_executor.set_hold(
                        np.concatenate([target_joints[:6], np.asarray([target_gripper])])
                    )
                    if execute:
                        continuous_executor.enable()
                    obs_buffer.clear()
                    policy_start_time = time.monotonic()
                    next_policy_time = 0.0
                    close_gate_until = 0.0
                    close_gate_fired = False
                    print("Switched to policy mode. Refilling observations.")

                if mode == "human":
                    target_joints = robot_state["ActualQ"].copy()
                    target_gripper = float(robot_state["gripper_pos"][0])
                    if execute:
                        robot.schedule_waypoint(
                            target_joints,
                            target_time=time.time() + command_latency,
                            gripper_pos=target_gripper,
                        )
                else:
                    now_mono = time.monotonic()
                    if not obs_buffer.is_ready:
                        if now_mono - last_policy_print >= policy_log_interval:
                            last_policy_print = now_mono
                            print(f"buffering obs {len(obs_buffer.frames)}/{n_obs_steps}")
                    elif policy_start_time is not None and now_mono - policy_start_time < policy_start_hold_time:
                        if execute:
                            robot.schedule_waypoint(
                                target_joints,
                                target_time=time.time() + command_latency,
                                gripper_pos=target_gripper,
                            )
                    elif now_mono >= next_policy_time:
                        obs_dict = obs_buffer.as_policy_input(torch_device)
                        obs_timestamps = obs_buffer.get_timestamps()
                        curr_time = time.time()
                        infer_start = time.perf_counter()
                        result = policy.predict_action(obs_dict)
                        infer_latency = time.perf_counter() - infer_start
                        action_sequence = get_policy_action_sequence(result, cfg)
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
                        action_chunk[:, 6] = [
                            clamp_gripper(v, width=robot.robot_config.gripper_width, margin=gripper_margin)
                            for v in action_chunk[:, 6]
                        ]
                        anchor_action = np.concatenate(
                            [
                                np.asarray(robot_state["ActualQ"], dtype=np.float64)[:6],
                                np.asarray([float(robot_state["gripper_pos"][0])], dtype=np.float64),
                            ]
                        )
                        if prepend_current_action and len(action_chunk) > 0:
                            action_chunk[0] = anchor_action
                        now_for_gate = time.monotonic()
                        close_gate_active = False
                        if close_gate_delay > 0 and len(action_chunk) > 0:
                            actual_gripper = float(robot_state["gripper_pos"][0])
                            min_pred_gripper = float(np.min(action_chunk[:, 6]))
                            is_closing = min_pred_gripper < actual_gripper - float(close_gate_threshold)
                            already_closed = actual_gripper <= float(close_gate_release_width)
                            if already_closed:
                                close_gate_fired = True
                            if (
                                is_closing
                                and not already_closed
                                and not close_gate_fired
                                and now_for_gate >= close_gate_until
                            ):
                                close_gate_until = now_for_gate + float(close_gate_delay)
                                close_gate_fired = True
                                print(
                                    "joint close gate start",
                                    f"delay={close_gate_delay:.3f}s",
                                    f"actual={actual_gripper:.5f}",
                                    f"pred_min={min_pred_gripper:.5f}",
                                )
                            if now_for_gate < close_gate_until:
                                close_gate_active = True
                                hold_width = actual_gripper
                                action_chunk[:, 6] = np.maximum(action_chunk[:, 6], hold_width)
                        action_chunk, clipped = _clamp_joint_chunk_delta(
                            action_chunk,
                            anchor_action,
                            max_joint_step=max_action_joint_step,
                            max_gripper_step=max_action_gripper_step,
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
                            print("policy over budget; scheduling latest joint action")
                        print(
                            "dp-joint chunk",
                            f"n={len(action_chunk)}",
                            f"dt0={action_timestamps[0] - curr_time:.3f}s",
                            f"dtN={action_timestamps[-1] - curr_time:.3f}s",
                            f"start_idx={candidate.action_start_idx}",
                            f"clipped={clipped}",
                            f"close_gate={close_gate_active}",
                            f"infer={infer_latency:.3f}s",
                            f"q0={np.array2string(action_chunk[0, :6], precision=4)}",
                            f"gripper={action_chunk[:, 6].min():.5f}/{action_chunk[:, 6].max():.5f}",
                        )
                        if debug_policy_chunk:
                            print(
                                "raw joint range",
                                f"q_min={np.array2string(action_chunk[:, :6].min(axis=0), precision=4)}",
                                f"q_max={np.array2string(action_chunk[:, :6].max(axis=0), precision=4)}",
                            )
                        if execute:
                            inserted = continuous_executor.add_chunk(
                                action_chunk,
                                action_timestamps,
                                now=curr_time,
                            )
                            continuous_executor.enable()
                            if inserted == 0:
                                print("joint continuous skipped stale chunk")
                            target_joints = action_chunk[-1, :6].copy()
                            target_gripper = float(action_chunk[-1, 6])
                        next_delay = max(
                            dt,
                            max(0.0, float(action_timestamps[-1]) - float(curr_time))
                            - max(float(action_exec_latency), max(0.0, float(replan_lookahead))),
                        )
                        next_policy_time = time.monotonic() + next_delay

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
            try:
                continuous_executor.stop()
            except UnboundLocalError:
                pass
            if cameras is not None:
                cameras.stop()
            if video_reader is not None:
                video_reader.stop()
            robot.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
