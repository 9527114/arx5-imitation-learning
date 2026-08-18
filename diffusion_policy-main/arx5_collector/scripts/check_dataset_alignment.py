import pathlib

import click
import cv2
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer


def read_video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        fps = 30.0
    return fps, frames


@click.command()
@click.argument("dataset_path")
@click.option("--expected-frequency", default=20.0, show_default=True, type=float)
@click.option("--camera-count", default=3, show_default=True, type=int)
@click.option("--safety-frames", default=2, show_default=True, type=int)
@click.option("--dt-warning-ratio", default=1.8, show_default=True, type=float)
def main(dataset_path, expected_frequency, camera_count, safety_frames, dt_warning_ratio):
    dataset_path = pathlib.Path(dataset_path)
    replay = ReplayBuffer.create_from_path(
        str((dataset_path / "replay_buffer.zarr").absolute()),
        mode="r",
    )
    timestamps = np.asarray(replay["timestamp"][:], dtype=np.float64)
    episode_lengths = np.asarray(replay.episode_lengths[:], dtype=np.int64)
    episode_ends = np.asarray(replay.episode_ends[:], dtype=np.int64)
    episode_starts = episode_ends - episode_lengths

    expected_dt = 1.0 / float(expected_frequency)
    dt_warn = expected_dt * float(dt_warning_ratio)

    total_steps = 0
    total_uncovered = 0
    total_dt_warn = 0
    bad_episodes = []
    dt_bad_episodes = []

    print(f"Dataset: {dataset_path}")
    print(f"Episodes: {replay.n_episodes} Steps: {replay.n_steps}")
    print(
        "Expected lowdim dt:",
        f"{expected_dt:.4f}s",
        "warning if dt >",
        f"{dt_warn:.4f}s",
    )

    for episode in range(replay.n_episodes):
        start = int(episode_starts[episode])
        end = int(episode_ends[episode])
        t = timestamps[start:end]
        total_steps += len(t)

        dt = np.diff(t)
        dt_warning_count = int(np.sum(dt > dt_warn)) if len(dt) else 0
        total_dt_warn += dt_warning_count
        if dt_warning_count:
            dt_bad_episodes.append(
                (
                    episode,
                    dt_warning_count,
                    float(dt.max()),
                    float(np.percentile(dt, 95)),
                )
            )

        t0 = float(t[0]) if len(t) else 0.0
        episode_uncovered = 0
        camera_reports = []
        for camera_idx in range(camera_count):
            video_path = dataset_path / "videos" / str(episode) / f"{camera_idx}.mp4"
            fps, frames = read_video_info(video_path)
            usable_last = max(0, frames - 1 - int(safety_frames))
            frame_indices = np.rint((t - t0) * fps).astype(np.int64)
            uncovered = int(np.sum((frame_indices < 0) | (frame_indices > usable_last)))
            episode_uncovered += uncovered
            camera_reports.append((camera_idx, fps, frames, usable_last, uncovered))

        total_uncovered += episode_uncovered
        if episode_uncovered:
            bad_episodes.append((episode, episode_uncovered, camera_reports))

    print("\nVideo timestamp coverage:")
    if bad_episodes:
        print("  FAIL")
        print(f"  uncovered lowdim-camera pairs: {total_uncovered}")
        for episode, uncovered, reports in bad_episodes[:20]:
            detail = ", ".join(
                f"cam{cam}:frames={frames},usable_last={usable_last},miss={miss}"
                for cam, _, frames, usable_last, miss in reports
                if miss
            )
            print(f"  episode {episode}: miss={uncovered} {detail}")
        if len(bad_episodes) > 20:
            print(f"  ... {len(bad_episodes) - 20} more bad episode(s)")
    else:
        print("  PASS")
        print("  every lowdim timestamp maps to valid frames for all cameras")

    print("\nLowdim timestamp regularity:")
    if dt_bad_episodes:
        print("  WARN")
        print(f"  dt warnings: {total_dt_warn}")
        for episode, count, dt_max, dt_p95 in dt_bad_episodes[:30]:
            print(
                f"  episode {episode}: count={count} "
                f"dt_max={dt_max:.4f}s dt_p95={dt_p95:.4f}s"
            )
        if len(dt_bad_episodes) > 30:
            print(f"  ... {len(dt_bad_episodes) - 30} more episode(s)")
    else:
        print("  PASS")
        print("  lowdim timestamps are regular under the selected threshold")

    print("\nSummary:")
    print(f"  checked lowdim steps: {total_steps}")
    print(f"  checked camera streams: {replay.n_episodes * camera_count}")
    print(f"  uncovered pairs: {total_uncovered}")
    print(f"  dt warning count: {total_dt_warn}")


if __name__ == "__main__":
    main()
