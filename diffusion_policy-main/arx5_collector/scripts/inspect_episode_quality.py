from pathlib import Path

import click
import cv2
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer


def video_frame_count(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames


@click.command()
@click.argument("dataset_path")
@click.option("--min-steps", default=60, show_default=True, type=int)
@click.option("--static-ratio-threshold", default=0.65, show_default=True, type=float)
@click.option("--min-pos-range", default=0.04, show_default=True, type=float)
@click.option("--close-threshold", default=0.06, show_default=True, type=float)
@click.option("--low-z-threshold", default=0.09, show_default=True, type=float)
@click.option("--require-close/--no-require-close", default=True, show_default=True)
@click.option("--require-low-z/--no-require-low-z", default=True, show_default=True)
def main(
    dataset_path,
    min_steps,
    static_ratio_threshold,
    min_pos_range,
    close_threshold,
    low_z_threshold,
    require_close,
    require_low_z,
):
    dataset_path = Path(dataset_path)
    replay_path = dataset_path / "replay_buffer.zarr"
    video_dir = dataset_path / "videos"
    if not replay_path.is_dir():
        raise click.ClickException(f"Missing replay buffer: {replay_path}")
    if not video_dir.is_dir():
        raise click.ClickException(f"Missing video dir: {video_dir}")

    replay = ReplayBuffer.create_from_path(str(replay_path.absolute()), mode="r")
    starts = replay.episode_ends[:] - replay.episode_lengths[:]
    action = np.asarray(replay["action"][:], dtype=np.float64)
    timestamps = np.asarray(replay["timestamp"][:], dtype=np.float64)

    bad = []
    print(f"Dataset: {dataset_path}")
    print(f"Episodes: {replay.n_episodes}, steps: {replay.n_steps}")
    print(
        "Columns:",
        "episode steps duration static pos_range z_min grip_min close_ratio low_z_ratio reasons",
    )
    for episode, (start, end) in enumerate(zip(starts, replay.episode_ends[:])):
        ep_action = action[start:end]
        ep_t = timestamps[start:end]
        reasons = []
        steps = int(end - start)
        duration = float(ep_t[-1] - ep_t[0]) if len(ep_t) > 1 else 0.0

        if steps < min_steps:
            reasons.append("short")
        if len(ep_action) >= 2:
            pose_delta = np.linalg.norm(np.diff(ep_action[:, :6], axis=0), axis=1)
            grip_delta = np.abs(np.diff(ep_action[:, 6]))
            static_ratio = float(np.mean((pose_delta < 1e-4) & (grip_delta < 1e-4)))
        else:
            static_ratio = 1.0
        if static_ratio > static_ratio_threshold:
            reasons.append("static")

        pos_range = float(np.linalg.norm(ep_action[:, :3].max(axis=0) - ep_action[:, :3].min(axis=0)))
        if pos_range < min_pos_range:
            reasons.append("low_motion")

        z_min = float(ep_action[:, 2].min())
        grip_min = float(ep_action[:, 6].min())
        close_ratio = float(np.mean(ep_action[:, 6] < close_threshold))
        low_z_ratio = float(np.mean(ep_action[:, 2] < low_z_threshold))
        if require_close and close_ratio <= 0:
            reasons.append("no_close")
        if require_low_z and low_z_ratio <= 0:
            reasons.append("no_low_z")

        ep_video_dir = video_dir / str(episode)
        frame_counts = []
        for camera_idx in range(3):
            video_path = ep_video_dir / f"{camera_idx}.mp4"
            if not video_path.is_file():
                reasons.append(f"missing_video_{camera_idx}")
                frame_counts.append(None)
            else:
                frame_counts.append(video_frame_count(video_path))
        readable_counts = [x for x in frame_counts if x is not None]
        if readable_counts and min(readable_counts) < max(1, steps - 5):
            reasons.append("short_video")

        if reasons:
            bad.append(episode)
        print(
            f"{episode:03d}",
            f"{steps:4d}",
            f"{duration:7.3f}",
            f"{static_ratio:6.2%}",
            f"{pos_range:8.4f}",
            f"{z_min:7.4f}",
            f"{grip_min:8.5f}",
            f"{close_ratio:6.2%}",
            f"{low_z_ratio:6.2%}",
            ",".join(reasons) if reasons else "ok",
        )

    print("")
    print(f"Suggested bad episodes ({len(bad)}):")
    print(",".join(str(x) for x in bad) if bad else "<none>")


if __name__ == "__main__":
    main()
