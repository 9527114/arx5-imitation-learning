import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer


def load_episode(replay, episode_idx):
    starts = replay.episode_ends[:] - replay.episode_lengths[:]
    start = int(starts[episode_idx])
    end = int(replay.episode_ends[episode_idx])
    timestamps = np.asarray(replay["timestamp"][start:end], dtype=np.float64)
    actions = np.asarray(replay["action"][start:end], dtype=np.float64)
    return timestamps, actions


def print_episode_summary(dataset_path, episode_idx, timestamps, actions):
    dt = np.diff(timestamps)
    dpos = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)
    drot = np.linalg.norm(np.diff(actions[:, 3:6], axis=0), axis=1)
    dgrip = np.abs(np.diff(actions[:, 6], axis=0))

    print(f"Dataset: {dataset_path}")
    print(f"Episode: {episode_idx}")
    print(f"Steps: {len(actions)}")
    if len(dt):
        print(
            "Timing:",
            f"duration={timestamps[-1] - timestamps[0]:.3f}s",
            f"dt_mean={dt.mean():.4f}s",
            f"dt_min={dt.min():.4f}s",
            f"dt_max={dt.max():.4f}s",
            f"freq≈{1.0 / max(dt.mean(), 1e-9):.1f}Hz",
        )
        print(
            "Action delta:",
            f"pos_max={dpos.max():.5f}m",
            f"rot_max={drot.max():.5f}rad",
            f"gripper_max={dgrip.max():.5f}m",
        )
    print(
        "Action range:",
        f"pos_min={np.array2string(actions[:, :3].min(axis=0), precision=4)}",
        f"pos_max={np.array2string(actions[:, :3].max(axis=0), precision=4)}",
        f"rot_min={np.array2string(actions[:, 3:6].min(axis=0), precision=4)}",
        f"rot_max={np.array2string(actions[:, 3:6].max(axis=0), precision=4)}",
        f"gripper={actions[:, 6].min():.5f}/{actions[:, 6].max():.5f}",
    )


def chunk_stats(actions):
    if len(actions) < 2:
        return 0.0, 0.0, 0.0
    dpos = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)
    drot = np.linalg.norm(np.diff(actions[:, 3:6], axis=0), axis=1)
    dgrip = np.abs(np.diff(actions[:, 6], axis=0))
    return float(dpos.max()), float(drot.max()), float(dgrip.max())


def print_chunk_summaries(timestamps, actions, chunk_size, speed, max_steps=None):
    rel_t = timestamps - timestamps[0]
    rel_t = rel_t / max(float(speed), 1e-6)
    if max_steps is not None:
        rel_t = rel_t[:max_steps]
        actions = actions[:max_steps]

    print("Chunk check:")
    for chunk_id, start in enumerate(range(0, len(actions), chunk_size)):
        end = min(start + chunk_size, len(actions))
        chunk = actions[start:end]
        pos_delta, rot_delta, gripper_delta = chunk_stats(chunk)
        jump = 0.0
        if start > 0:
            jump = float(np.linalg.norm(actions[start, :3] - actions[start - 1, :3]))
        print(
            f"  chunk {chunk_id:03d}:",
            f"steps={start}:{end}",
            f"t={rel_t[start]:.3f}-{rel_t[end - 1]:.3f}s",
            f"jump_from_prev={jump:.5f}m",
            f"pos_max={pos_delta:.5f}m",
            f"rot_max={rot_delta:.5f}rad",
            f"gripper_max={gripper_delta:.5f}m",
            f"first={np.array2string(chunk[0, :3], precision=4)}",
            f"last={np.array2string(chunk[-1, :3], precision=4)}",
        )


def show_episode_videos(dataset_path, episode_idx, fps):
    ep_dir = Path(dataset_path) / "videos" / str(episode_idx)
    video_paths = sorted(ep_dir.glob("*.mp4"), key=lambda p: int(p.stem))
    if not video_paths:
        print(f"No videos found in {ep_dir}")
        return

    caps = [cv2.VideoCapture(str(path)) for path in video_paths]
    try:
        delay_ms = max(1, int(round(1000.0 / fps)))
        while True:
            frames = []
            for cap in caps:
                ok, frame = cap.read()
                if not ok:
                    return
                frame = cv2.resize(frame, (426, 240), interpolation=cv2.INTER_AREA)
                frames.append(frame)
            preview = np.concatenate(frames, axis=1)
            cv2.imshow("arx5_replay_video", preview)
            if cv2.waitKey(delay_ms) == ord("q"):
                return
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyWindow("arx5_replay_video")


def clamp_action_steps(actions, max_pos_step, max_rot_step, max_gripper_step):
    actions = np.asarray(actions, dtype=np.float64).copy()
    clipped = 0
    for idx in range(1, len(actions)):
        prev = actions[idx - 1].copy()
        action = actions[idx].copy()
        if max_pos_step is not None:
            dpos = action[:3] - prev[:3]
            norm = float(np.linalg.norm(dpos))
            if norm > max_pos_step:
                action[:3] = prev[:3] + dpos / max(norm, 1e-9) * max_pos_step
                clipped += 1
        if max_rot_step is not None:
            drot = action[3:6] - prev[3:6]
            norm = float(np.linalg.norm(drot))
            if norm > max_rot_step:
                action[3:6] = prev[3:6] + drot / max(norm, 1e-9) * max_rot_step
                clipped += 1
        if max_gripper_step is not None:
            dgrip = float(action[6] - prev[6])
            if abs(dgrip) > max_gripper_step:
                action[6] = prev[6] + np.sign(dgrip) * max_gripper_step
                clipped += 1
        actions[idx] = action
    return actions, clipped


def get_replay_key(key_counter, key_code):
    for key_stroke in key_counter.get_press_events():
        if key_stroke == key_code(char="q"):
            return "q"
        if key_stroke == key_code(char="r"):
            return "r"
        if key_stroke == key_code(char="h"):
            return "h"
        if key_stroke == key_code(char="c"):
            return "c"
    return None


def wait_until_with_keys(target_time, key_counter, key_code, poll_dt=0.02):
    while True:
        requested_key = get_replay_key(key_counter, key_code)
        if requested_key is not None:
            return requested_key
        remaining = target_time - time.time()
        if remaining <= 0:
            return None
        time.sleep(min(float(poll_dt), remaining))


def current_to_action_delta(robot, action):
    state = robot.get_state()
    actual_pose = np.asarray(state["ActualTCPPose"], dtype=np.float64)
    actual_gripper = float(state["gripper_pos"][0])
    return (
        float(np.linalg.norm(action[:3] - actual_pose[:3])),
        float(np.linalg.norm(action[3:6] - actual_pose[3:6])),
        abs(float(action[6]) - actual_gripper),
    )


def run_human_mode(robot, teleop, args, key_counter, key_code):
    dt = 1.0 / float(args.human_frequency)
    state = robot.get_state()
    target_pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).copy()
    target_gripper = float(state["gripper_pos"][0])
    print("Human mode: SpaceMouse active. c=start/resume replay, r=reset home, q=stop.")
    while True:
        requested_key = get_replay_key(key_counter, key_code)
        if requested_key == "q":
            print("Emergency stop requested in human mode.")
            return "quit"
        if requested_key == "c":
            print("Resume replay requested.")
            return "resume"
        if requested_key == "r":
            print("Resetting robot to home...")
            state = robot.reset_to_home()
            target_pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).copy()
            target_gripper = float(state["gripper_pos"][0])
            print("Robot reset to home.")

        command = teleop.update(
            target_pose=target_pose,
            target_gripper_pos=target_gripper,
            dt=dt,
            gripper_width=robot.robot_config.gripper_width,
        )
        target_pose = command.action
        target_gripper = command.gripper_action
        robot.schedule_waypoint(
            target_pose,
            target_time=time.time() + args.command_latency,
            gripper_pos=target_gripper,
        )
        time.sleep(dt)


def replay_on_robot(args, timestamps, actions):
    from arx5_collector.robot import Arx5Robot
    from arx5_collector.input import SpaceMouseTeleop
    from diffusion_policy.real_world.keystroke_counter import KeystrokeCounter
    from pynput.keyboard import KeyCode

    rel_t = timestamps - timestamps[0]
    rel_t = rel_t / max(float(args.speed), 1e-6)
    if args.max_steps is not None:
        rel_t = rel_t[: args.max_steps]
        actions = actions[: args.max_steps]
    if not args.disable_step_clamp:
        actions, clipped = clamp_action_steps(
            actions,
            max_pos_step=args.max_pos_step,
            max_rot_step=args.max_rot_step,
            max_gripper_step=args.max_gripper_step,
        )
        print(f"Replay step clamp: clipped_steps={clipped}")

    robot = Arx5Robot(
        model=args.model,
        interface=args.interface,
        reset_to_home=False,
        command_mode=args.command_mode,
    )
    key_counter = None
    teleop = None
    try:
        robot.start()
        teleop = SpaceMouseTeleop(
            pos_speed=args.pos_speed,
            rot_speed=args.rot_speed,
            gripper_speed=args.gripper_speed,
            gripper_margin=args.gripper_margin,
            deadzone=args.spacemouse_deadzone,
            smoothing_window=args.spacemouse_smoothing_window,
        )
        teleop.start()
        key_counter = KeystrokeCounter()
        key_counter.start()
        print("Replay controls: q=emergency stop, h=human mode, r=reset home, c=resume from human")
        if args.start_human:
            result = run_human_mode(
                robot=robot,
                teleop=teleop,
                args=args,
                key_counter=key_counter,
                key_code=KeyCode,
            )
            if result == "quit":
                return
        state = robot.get_state()
        actual_pose = np.asarray(state["ActualTCPPose"], dtype=np.float64)
        actual_gripper = float(state["gripper_pos"][0])
        first = actions[0]
        start_pos_delta = float(np.linalg.norm(first[:3] - actual_pose[:3]))
        start_rot_delta = float(np.linalg.norm(first[3:6] - actual_pose[3:6]))
        start_gripper_delta = abs(float(first[6]) - actual_gripper)
        print(
            "Current-to-first delta:",
            f"pos={start_pos_delta:.5f}m",
            f"rot={start_rot_delta:.5f}rad",
            f"gripper={start_gripper_delta:.5f}m",
        )
        if (
            start_pos_delta > args.max_start_pos_delta
            or start_rot_delta > args.max_start_rot_delta
        ) and not args.allow_large_start_delta:
            raise RuntimeError(
                "Current robot pose is too far from the first recorded action. "
                "Manually move the robot near the episode start, then run replay again. "
                "If you intentionally want to override this check, pass "
                "--allow-large-start-delta."
            )
        if args.move_to_start:
            print("Moving to first recorded action...")
            robot.schedule_waypoint(
                first[:6],
                target_time=time.time() + args.command_latency,
                gripper_pos=float(first[6]),
            )
            time.sleep(args.start_settle_time)

        print("Replaying episode on robot...")
        replay_start = time.time() + args.command_latency
        idx = 0
        while idx < len(actions):
            now = time.time()
            target_times = replay_start + rel_t[idx : idx + args.chunk_size]
            chunk = actions[idx : idx + args.chunk_size].copy()
            if len(chunk) == 0:
                continue
            is_future = target_times > now + args.action_exec_latency
            if not np.any(is_future):
                idx += args.chunk_size
                continue
            chunk = chunk[is_future]
            target_times = target_times[is_future]
            pos_delta, rot_delta, gripper_delta = chunk_stats(chunk)
            chunk_id = idx // args.chunk_size
            print(
                f"send chunk {chunk_id:03d}:",
                f"n={len(chunk)}",
                f"dt0={target_times[0] - now:.3f}s",
                f"dtN={target_times[-1] - now:.3f}s",
                f"pos_max={pos_delta:.5f}m",
                f"rot_max={rot_delta:.5f}rad",
                f"gripper_max={gripper_delta:.5f}m",
            )
            robot.schedule_waypoints(
                chunk,
                target_times,
                gripper_margin=args.gripper_margin,
            )
            next_idx = idx + args.chunk_size
            if next_idx < len(rel_t):
                wait_target = replay_start + float(rel_t[next_idx]) - args.command_latency
            else:
                wait_target = target_times[-1]
            requested_key = wait_until_with_keys(
                wait_target,
                key_counter=key_counter,
                key_code=KeyCode,
                poll_dt=args.key_poll_dt,
            )
            if requested_key == "q":
                print("Emergency stop requested. Setting robot to damping.")
                return
            if requested_key == "h":
                while True:
                    result = run_human_mode(
                        robot=robot,
                        teleop=teleop,
                        args=args,
                        key_counter=key_counter,
                        key_code=KeyCode,
                    )
                    if result == "quit":
                        return
                    pos_delta, rot_delta, gripper_delta = current_to_action_delta(
                        robot,
                        actions[idx],
                    )
                    print(
                        "Current-to-resume delta:",
                        f"pos={pos_delta:.5f}m",
                        f"rot={rot_delta:.5f}rad",
                        f"gripper={gripper_delta:.5f}m",
                    )
                    if (
                        pos_delta <= args.max_start_pos_delta
                        and rot_delta <= args.max_start_rot_delta
                    ) or args.allow_large_start_delta:
                        replay_start = time.time() + args.command_latency - float(rel_t[idx])
                        print(f"Resuming replay from step {idx}.")
                        break
                    print(
                        "Current pose is too far from the next replay action. "
                        "Stay in human mode and move closer, then press c again."
                    )
            if requested_key == "r":
                print("Reset requested. Resetting robot to home and entering human mode...")
                robot.reset_to_home()
                print("Robot reset to home.")
                result = run_human_mode(
                    robot=robot,
                    teleop=teleop,
                    args=args,
                    key_counter=key_counter,
                    key_code=KeyCode,
                )
                if result == "quit":
                    return
                replay_start = time.time() + args.command_latency - float(rel_t[idx])
            idx += args.chunk_size
        wait_until_with_keys(
            time.time() + args.command_latency + 0.2,
            key_counter=key_counter,
            key_code=KeyCode,
            poll_dt=args.key_poll_dt,
        )
        print("Replay finished.")
    finally:
        if key_counter is not None:
            key_counter.stop()
            key_counter.join()
        if teleop is not None:
            teleop.stop()
        robot.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--model", default="X5")
    parser.add_argument("--interface", default="can1")
    parser.add_argument("--command-mode", default="traj", choices=["traj", "cmd"])
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 is real time; 0.5 is half speed.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--command-latency", type=float, default=0.15)
    parser.add_argument("--action-exec-latency", type=float, default=0.01)
    parser.add_argument("--gripper-margin", type=float, default=0.0)
    parser.add_argument("--move-to-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-start-pos-delta", type=float, default=0.03)
    parser.add_argument("--max-start-rot-delta", type=float, default=0.15)
    parser.add_argument("--allow-large-start-delta", action="store_true")
    parser.add_argument("--disable-step-clamp", action="store_true")
    parser.add_argument("--max-pos-step", type=float, default=0.006)
    parser.add_argument("--max-rot-step", type=float, default=0.02)
    parser.add_argument("--max-gripper-step", type=float, default=0.003)
    parser.add_argument("--start-settle-time", type=float, default=2.0)
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--chunk-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--key-poll-dt", type=float, default=0.02)
    parser.add_argument("--start-human", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--human-frequency", type=float, default=50.0)
    parser.add_argument("--pos-speed", type=float, default=0.4)
    parser.add_argument("--rot-speed", type=float, default=0.75)
    parser.add_argument("--gripper-speed", type=float, default=0.08)
    parser.add_argument("--spacemouse-deadzone", type=float, default=0.02)
    parser.add_argument("--spacemouse-smoothing-window", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    replay_path = dataset_path / "replay_buffer.zarr"
    if not replay_path.is_dir():
        raise FileNotFoundError(f"Missing replay buffer: {replay_path}")

    replay = ReplayBuffer.create_from_path(str(replay_path.absolute()), mode="r")
    if args.episode < 0 or args.episode >= replay.n_episodes:
        raise ValueError(f"Episode {args.episode} out of range: 0..{replay.n_episodes - 1}")

    timestamps, actions = load_episode(replay, args.episode)
    print_episode_summary(dataset_path, args.episode, timestamps, actions)
    if args.chunk_check:
        print_chunk_summaries(
            timestamps,
            actions,
            chunk_size=args.chunk_size,
            speed=args.speed,
            max_steps=args.max_steps,
        )

    if args.show_video:
        show_episode_videos(dataset_path, args.episode, fps=args.video_fps)

    if not args.execute:
        print("Dry-run only. Add --execute to replay this episode on the robot.")
        return

    replay_on_robot(args, timestamps, actions)


if __name__ == "__main__":
    main()
