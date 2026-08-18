import os
import threading
import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
from pynput.keyboard import Key, KeyCode

from arx5_collector.camera import ThreeCameraRecorder
from arx5_collector.data import DPEpisodeWriter
from arx5_collector.input import SpaceMouseTeleop
from arx5_collector.robot import Arx5Robot
from diffusion_policy.real_world.keystroke_counter import KeystrokeCounter


def resize_frame(frame, width=426, height=240):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def make_preview(realsense_data, usb_data):
    frames = []
    frames.append(resize_frame(usb_data["color"]))
    for idx in sorted(realsense_data.keys()):
        frames.append(resize_frame(realsense_data[idx]["color"]))
    return np.concatenate(frames, axis=1)


def clamp_pose_delta(
    pose,
    center_pose,
    max_pos_delta,
    max_x_delta,
    max_y_delta,
    max_z_delta,
    max_rot_delta,
    min_z,
):
    pose = np.asarray(pose, dtype=np.float64).copy()
    center_pose = np.asarray(center_pose, dtype=np.float64)
    if max_pos_delta is None and max_x_delta is None and max_y_delta is None and max_z_delta is None:
        max_xyz_delta = None
    else:
        fallback_delta = np.inf if max_pos_delta is None else max_pos_delta
        max_xyz_delta = np.array(
            [
                fallback_delta if max_x_delta is None else max_x_delta,
                fallback_delta if max_y_delta is None else max_y_delta,
                fallback_delta if max_z_delta is None else max_z_delta,
            ],
            dtype=np.float64,
        )
    if max_xyz_delta is not None:
        pose[:3] = np.clip(
            pose[:3],
            center_pose[:3] - max_xyz_delta,
            center_pose[:3] + max_xyz_delta,
        )
    if min_z is not None:
        pose[2] = max(pose[2], min_z)
    if max_rot_delta is not None:
        pose[3:] = np.clip(
            pose[3:],
            center_pose[3:] - max_rot_delta,
            center_pose[3:] + max_rot_delta,
        )
    return pose


@click.command()
@click.option("--output", "-o", required=True)
@click.option("--model", default="X5", show_default=True)
@click.option("--interface", default="can1", show_default=True)
@click.option("--realsense-config", default=None)
@click.option("--usb-config", default=None)
@click.option("--usb-device", default=0, show_default=True, type=int)
@click.option("--disable-usb-recording", is_flag=True)
@click.option("--frequency", "-f", default=100.0, show_default=True, type=float)
@click.option("--data-frequency", default=20.0, show_default=True, type=float)
@click.option("--width", default=1280, show_default=True, type=int)
@click.option("--height", default=720, show_default=True, type=int)
@click.option("--usb-width", default=640, show_default=True, type=int)
@click.option("--usb-height", default=480, show_default=True, type=int)
@click.option("--camera-fps", default=30, show_default=True, type=int)
@click.option("--camera-pump-fps", default=30.0, show_default=True, type=float)
@click.option("--preview-fps", default=15.0, show_default=True, type=float)
@click.option("--realsense-crf", default=23, show_default=True, type=int)
@click.option("--command-latency", default=0.05, show_default=True, type=float)
@click.option("--debug-teleop", is_flag=True)
@click.option("--record-on-motion-only", is_flag=True, help="Only write lowdim/action samples while SpaceMouse or gripper is active.")
@click.option("--motion-threshold", default=0.015, show_default=True, type=float)
@click.option("--motion-prepad", default=0.20, show_default=True, type=float, help="Seconds of samples kept before motion starts.")
@click.option("--motion-postpad", default=0.35, show_default=True, type=float, help="Seconds of samples kept after motion stops.")
@click.option("--pos-speed", default=0.4, show_default=True, type=float)
@click.option("--rot-speed", default=0.75, show_default=True, type=float)
@click.option("--gripper-speed", default=0.08, show_default=True, type=float)
@click.option("--gripper-margin", default=0.0, show_default=True, type=float)
@click.option("--gripper-safe-torque", default=0.75, show_default=True, type=float, help="Hold gripper if abs(torque) exceeds this while closing. Set <=0 to disable.")
@click.option("--gripper-safe-margin", default=0.002, show_default=True, type=float, help="Extra opening margin used by gripper torque safety hold.")
@click.option("--max-pos-delta", default=None, type=float)
@click.option("--max-x-delta", default=None, type=float)
@click.option("--max-y-delta", default=None, type=float)
@click.option("--max-z-delta", default=None, type=float)
@click.option("--max-rot-delta", default=None, type=float)
@click.option("--min-z", default=None, type=float)
@click.option("--reset-to-home-start/--no-reset-to-home-start", default=True, show_default=True)
@click.option("--enable-spacemouse-reset", is_flag=True)
@click.option("--command-mode", default="traj", show_default=True, type=click.Choice(["cmd", "traj"]))
def main(
    output,
    model,
    interface,
    realsense_config,
    usb_config,
    usb_device,
    disable_usb_recording,
    frequency,
    data_frequency,
    width,
    height,
    usb_width,
    usb_height,
    camera_fps,
    camera_pump_fps,
    preview_fps,
    realsense_crf,
    command_latency,
    debug_teleop,
    record_on_motion_only,
    motion_threshold,
    motion_prepad,
    motion_postpad,
    pos_speed,
    rot_speed,
    gripper_speed,
    gripper_margin,
    gripper_safe_torque,
    gripper_safe_margin,
    max_pos_delta,
    max_x_delta,
    max_y_delta,
    max_z_delta,
    max_rot_delta,
    min_z,
    reset_to_home_start,
    enable_spacemouse_reset,
    command_mode,
):
    if data_frequency <= 0:
        raise click.ClickException("--data-frequency must be positive.")
    if data_frequency > camera_fps:
        click.echo(
            "WARNING: --data-frequency is higher than --camera-fps. "
            "Lowdim/action will be recorded faster than video, so multiple "
            "training samples may share the same camera frame.",
            err=True,
        )
    if camera_pump_fps <= 0 or preview_fps <= 0 or frequency <= 0:
        raise click.ClickException("--frequency, --camera-pump-fps, and --preview-fps must be positive.")
    dt = 1.0 / frequency
    data_dt = 1.0 / data_frequency
    camera_dt = 1.0 / camera_pump_fps
    preview_dt = 1.0 / preview_fps
    cv2.setNumThreads(1)
    with SharedMemoryManager() as shm_manager:
        robot = Arx5Robot(
            model=model,
            interface=interface,
            reset_to_home=False,
            command_mode=command_mode,
            gripper_safe_torque=gripper_safe_torque,
            gripper_safe_margin=gripper_safe_margin,
        )
        teleop = SpaceMouseTeleop(
            shm_manager=shm_manager,
            pos_speed=pos_speed,
            rot_speed=rot_speed,
            gripper_speed=gripper_speed,
            gripper_margin=gripper_margin,
        )
        cameras = ThreeCameraRecorder(
            shm_manager=shm_manager,
            realsense_config=realsense_config,
            usb_config=usb_config,
            usb_device=usb_device,
            resolution=(width, height),
            usb_resolution=(usb_width, usb_height),
            fps=camera_fps,
            realsense_video_crf=realsense_crf,
        )
        writer = DPEpisodeWriter(
            output_dir=output,
            frequency=data_frequency,
            metadata={
                "control_frequency": float(frequency),
                "camera_fps": float(camera_fps),
                "camera_pump_fps": float(camera_pump_fps),
                "preview_fps": float(preview_fps),
            },
        )

        print("Controls:")
        print("  c: start recording")
        print("  s: save current episode")
        print("  d: discard current episode and reset home")
        print("  r: reset robot to home")
        print("  q: save current episode, reset home, and quit")
        print("  Backspace: drop last saved episode")
        print("  Space: stage marker")
        print("Preview/video order: camera_0=usb | camera_1=realsense_0 | camera_2=realsense_1")
        print(
            f"Camera settings: realsense={width}x{height}@{camera_fps} "
            f"crf={realsense_crf}, usb={usb_width}x{usb_height}@{camera_fps}"
        )
        print(
            f"Control frequency={frequency:g}Hz, "
            f"data frequency={data_frequency:g}Hz"
        )
        print(
            "Gripper safety:",
            f"safe_torque={gripper_safe_torque:g}",
            f"safe_margin={gripper_safe_margin:g}",
        )
        if record_on_motion_only:
            print(
                "Motion-gated recording enabled:",
                f"threshold={motion_threshold:g}",
                f"prepad={motion_prepad:g}s",
                f"postpad={motion_postpad:g}s",
            )

        robot.start()
        if reset_to_home_start:
            print("Resetting robot to home before teleop...")
            robot.reset_to_home()
        teleop.start()
        cameras.start()
        camera_lock = threading.Lock()
        camera_state_lock = threading.Lock()
        camera_stop_event = threading.Event()
        camera_state = {
            "preview": None,
            "error": None,
        }

        def camera_worker():
            rs_cache = None
            next_camera_time = time.monotonic()
            next_preview_time = next_camera_time
            while not camera_stop_event.is_set():
                now_mono = time.monotonic()
                if now_mono < next_camera_time:
                    time.sleep(min(0.002, next_camera_time - now_mono))
                    continue
                try:
                    with camera_lock:
                        rs_cache = cameras.realsense.get(out=rs_cache)
                        usb_data = cameras.pump_usb_frame()
                    if now_mono >= next_preview_time:
                        preview = make_preview(rs_cache, usb_data)
                        with camera_state_lock:
                            camera_state["preview"] = preview
                        next_preview_time += preview_dt
                        if next_preview_time < now_mono - preview_dt:
                            next_preview_time = now_mono + preview_dt
                except Exception as exc:
                    with camera_state_lock:
                        camera_state["error"] = exc
                    time.sleep(0.1)
                next_camera_time += camera_dt
                if next_camera_time < now_mono - camera_dt:
                    next_camera_time = now_mono + camera_dt

        camera_thread = threading.Thread(
            target=camera_worker,
            name="arx5_camera_worker",
            daemon=True,
        )
        camera_thread.start()
        key_counter = KeystrokeCounter()
        key_counter.start()
        is_recording = False
        try:
            robot_state = robot.get_state()
            target_pose = robot_state["ActualTCPPose"].copy()
            pose_center = target_pose.copy()
            target_gripper = float(robot_state["gripper_pos"][0])
            next_data_time = None
            motion_sample_buffer = []
            last_motion_time = -np.inf
            last_written_motion_buffer_idx = 0
            last_stage_value = 0
            def fmt_limit(value):
                return "disabled" if value is None else f"{value:.4f}"

            x_limit = max_pos_delta if max_x_delta is None else max_x_delta
            y_limit = max_pos_delta if max_y_delta is None else max_y_delta
            z_limit = max_pos_delta if max_z_delta is None else max_z_delta
            print("Initial target_pose", np.array2string(target_pose, precision=4))
            print("Pose center", np.array2string(pose_center, precision=4))
            print(
                "Position limits",
                f"x={fmt_limit(x_limit)}",
                f"y={fmt_limit(y_limit)}",
                f"z={fmt_limit(z_limit)}",
            )
            print(f"min_z {fmt_limit(min_z)}")
            print(f"Initial gripper_pos {target_gripper:.4f}")
            last_action_time = {}

            def allow_action(name, cooldown=0.35):
                now = time.monotonic()
                last_time = last_action_time.get(name, -np.inf)
                if now - last_time < cooldown:
                    return False
                last_action_time[name] = now
                return True

            def reset_robot_to_home(reason):
                nonlocal robot_state, target_pose, pose_center, target_gripper
                if is_recording:
                    print(f"{reason} reset ignored while recording.")
                    return False
                print(f"{reason} Resetting robot to home...")
                robot_state = robot.reset_to_home()
                target_pose = robot_state["TargetTCPPose"].copy()
                pose_center = target_pose.copy()
                target_gripper = float(robot_state["target_gripper_pos"][0])
                print("Robot reset to home.")
                return True

            def abort_current_episode_and_reset(reason):
                nonlocal is_recording, next_data_time
                if not is_recording:
                    print(f"{reason} no active episode to delete.")
                    reset_robot_to_home(reason)
                    return
                with camera_lock:
                    cameras.stop_recording()
                writer.abort_episode()
                is_recording = False
                next_data_time = None
                print("Current episode deleted.")
                reset_robot_to_home(reason)

            def start_episode():
                nonlocal is_recording, next_data_time, motion_sample_buffer
                nonlocal last_motion_time, last_written_motion_buffer_idx, last_stage_value
                if not allow_action("start_recording"):
                    return
                if is_recording:
                    print("Start requested but an episode is already recording.")
                    return
                start_time = time.time() + 0.5
                writer.start_episode(start_time=start_time)
                next_data_time = time.monotonic() + max(0.0, start_time - time.time())
                with camera_lock:
                    cameras.start_recording(
                        writer.episode_video_dir(),
                        start_time=start_time,
                        record_usb=not disable_usb_recording,
                    )
                is_recording = True
                motion_sample_buffer = []
                last_motion_time = -np.inf
                last_written_motion_buffer_idx = 0
                last_stage_value = 0
                print(f"Episode {writer.current_episode_id} started.")

            def save_current_episode(reason, debounce=True):
                nonlocal is_recording, next_data_time, motion_sample_buffer
                if debounce and not allow_action("save_recording"):
                    return False
                if not is_recording:
                    print(f"{reason} no active episode to save.")
                    return False
                episode_id = writer.current_episode_id
                n_samples = writer.sample_count
                with camera_lock:
                    cameras.stop_recording()
                video_dir = writer.episode_video_dir(episode_id)
                if disable_usb_recording:
                    expected_indices = range(cameras.realsense.n_cameras)
                else:
                    expected_indices = range(cameras.n_cameras)
                expected_videos = [
                    os.path.join(video_dir, f"{idx}.mp4")
                    for idx in expected_indices
                ]
                missing_videos = [
                    path for path in expected_videos if not os.path.exists(path)
                ]
                if n_samples == 0:
                    writer.abort_episode()
                    is_recording = False
                    next_data_time = None
                    print(f"Episode {episode_id} had 0 samples and was discarded.")
                    return False
                saved = writer.end_episode()
                is_recording = False
                next_data_time = None
                motion_sample_buffer = []
                if saved:
                    print(f"Episode {episode_id} saved with {n_samples} samples.")
                    if missing_videos:
                        print("WARNING: missing video files:")
                        for path in missing_videos:
                            print(f"  {path}")
                    return True
                print(f"Episode {episode_id} was not saved.")
                return False

            def save_reset_and_quit(reason):
                if is_recording:
                    save_current_episode(reason, debounce=False)
                reset_robot_to_home(reason)

            def write_motion_gated_sample(sample, now_mono):
                nonlocal last_written_motion_buffer_idx
                if not record_on_motion_only:
                    writer.put_step(**sample)
                    return

                motion_sample_buffer.append((now_mono, sample))
                keep_after = max(float(motion_prepad) + float(motion_postpad) + data_dt, 1.0)
                if last_written_motion_buffer_idx > 0:
                    while (
                        len(motion_sample_buffer) > 1
                        and motion_sample_buffer[0][0] < now_mono - keep_after
                        and last_written_motion_buffer_idx > 0
                    ):
                        motion_sample_buffer.pop(0)
                        last_written_motion_buffer_idx -= 1
                else:
                    while (
                        len(motion_sample_buffer) > 1
                        and motion_sample_buffer[0][0] < now_mono - motion_prepad
                    ):
                        motion_sample_buffer.pop(0)

                should_write = now_mono <= last_motion_time + motion_postpad
                if not should_write:
                    return

                start_time_for_write = max(-np.inf, last_motion_time - motion_prepad)
                while last_written_motion_buffer_idx < len(motion_sample_buffer):
                    sample_time, buffered_sample = motion_sample_buffer[last_written_motion_buffer_idx]
                    if sample_time >= start_time_for_write:
                        writer.put_step(**buffered_sample)
                    last_written_motion_buffer_idx += 1

            t_loop = time.monotonic()
            while True:
                loop_start = time.monotonic()
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char="q"):
                        save_reset_and_quit("Quit requested.")
                        return
                    elif key_stroke == KeyCode(char="c"):
                        start_episode()
                    elif key_stroke == KeyCode(char="s"):
                        save_current_episode("Save requested.")
                    elif key_stroke == KeyCode(char="d"):
                        abort_current_episode_and_reset("Delete requested.")
                    elif key_stroke == Key.backspace and not is_recording:
                        writer.drop_last_episode()
                        print("Dropped last saved episode.")
                    elif key_stroke == KeyCode(char="r"):
                        reset_robot_to_home("Keyboard r requested.")

                robot_state = robot.get_state()
                command = teleop.update(
                    target_pose=target_pose,
                    target_gripper_pos=target_gripper,
                    dt=dt,
                    gripper_width=robot.robot_config.gripper_width,
                    stage=key_counter[Key.space],
                )
                target_pose = clamp_pose_delta(
                    command.action,
                    pose_center,
                    max_pos_delta=max_pos_delta,
                    max_x_delta=max_x_delta,
                    max_y_delta=max_y_delta,
                    max_z_delta=max_z_delta,
                    max_rot_delta=max_rot_delta,
                    min_z=min_z,
                )
                target_gripper = command.gripper_action
                raw_motion_norm = float(np.linalg.norm(command.raw_motion))
                gripper_active = command.left_pressed != command.right_pressed
                stage_changed = int(command.stage) != int(last_stage_value)
                motion_active = (
                    raw_motion_norm >= motion_threshold
                    or gripper_active
                    or stage_changed
                )
                if motion_active:
                    last_motion_time = time.monotonic()
                last_stage_value = int(command.stage)
                if command.reset_requested:
                    if not enable_spacemouse_reset:
                        print("Reset request ignored. Pass --enable-spacemouse-reset to enable it.")
                    elif is_recording:
                        print("Reset request ignored while recording.")
                    else:
                        reset_robot_to_home("SpaceMouse reset requested.")
                        continue
                if debug_teleop and np.linalg.norm(command.raw_motion) > 1e-3:
                    print(
                        "spacemouse",
                        np.array2string(command.raw_motion, precision=3),
                        "target",
                        np.array2string(target_pose, precision=3),
                    )
                robot.schedule_waypoint(
                    target_pose,
                    target_time=time.time() + command_latency,
                    gripper_pos=target_gripper,
                )
                if debug_teleop and (command.left_pressed or command.right_pressed):
                    post_cmd_state = robot.get_state()
                    gripper_intent = "hold"
                    if command.left_pressed and not command.right_pressed:
                        gripper_intent = "close"
                    elif command.right_pressed and not command.left_pressed:
                        gripper_intent = "open"
                    elif command.left_pressed and command.right_pressed:
                        gripper_intent = "reset"
                    print(
                        f"buttons left={command.left_pressed} right={command.right_pressed} "
                        f"intent={gripper_intent} "
                        f"cmd_gripper={target_gripper:.5f} "
                        f"sdk_target={post_cmd_state['target_gripper_pos'][0]:.5f} "
                        f"actual={post_cmd_state['gripper_pos'][0]:.5f} "
                        f"torque={post_cmd_state['gripper_torque'][0]:.3f}"
                    )

                now_for_data = time.monotonic()
                now_wall = time.time()
                if (
                    is_recording
                    and next_data_time is not None
                    and now_for_data >= next_data_time
                    and writer.start_time is not None
                    and now_wall >= writer.start_time
                ):
                    sample = dict(
                        timestamp=now_wall,
                        robot_state=robot_state,
                        action=target_pose.copy(),
                        gripper_action=target_gripper,
                        stage=command.stage,
                    )
                    write_motion_gated_sample(sample, now_for_data)
                    next_data_time += data_dt
                    if next_data_time < now_for_data - data_dt:
                        next_data_time = now_for_data + data_dt

                with camera_state_lock:
                    latest_preview = camera_state["preview"]
                    camera_error = camera_state["error"]
                    camera_state["error"] = None
                if camera_error is not None:
                    print(f"Camera worker error: {camera_error}")
                if latest_preview is not None:
                    preview = latest_preview.copy()
                    status = f"recording={is_recording} episodes={writer.n_episodes}"
                    cv2.putText(
                        preview,
                        status,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("arx5_collect_demo", preview)
                key = cv2.pollKey()
                if key == ord("q"):
                    save_reset_and_quit("Quit requested.")
                    return
                elif key == ord("c"):
                    start_episode()
                elif key == ord("r"):
                    reset_robot_to_home("Window r requested.")
                elif key == ord("d"):
                    abort_current_episode_and_reset("Window delete requested.")
                elif key == ord("s"):
                    save_current_episode("Window save requested.")

                t_loop += dt
                sleep_time = t_loop - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    t_loop = loop_start
        except KeyboardInterrupt:
            print("Interrupted. Stopping and saving current episode if needed...")
            if is_recording:
                with camera_lock:
                    cameras.stop_recording()
                writer.end_episode()
        finally:
            camera_stop_event.set()
            camera_thread.join(timeout=2.0)
            key_counter.stop()
            key_counter.join()
            cameras.stop()
            teleop.stop()
            robot.stop()
            cv2.destroyAllWindows()
            print(f"Dataset output: {os.path.abspath(output)}")


if __name__ == "__main__":
    main()
