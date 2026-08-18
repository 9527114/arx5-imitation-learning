import json
import os
import pathlib
import subprocess

import click
import cv2
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer


def ffprobe_stream(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    streams = json.loads(out).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {path}")
    return streams[0]


def frame_diff_stats(path, max_samples):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, n_frames // max_samples)
    prev = None
    diffs = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(
                    float(
                        np.mean(
                            np.abs(gray.astype(np.float32) - prev.astype(np.float32))
                        )
                    )
                )
            prev = gray
        idx += 1
    cap.release()
    if not diffs:
        return None
    arr = np.asarray(diffs)
    return {
        "diff_mean": float(arr.mean()),
        "diff_p50": float(np.percentile(arr, 50)),
        "low_diff_ratio": float(np.mean(arr < 1.0)),
        "static_ratio": float(np.mean(arr < 0.5)),
    }


def print_array_stats(replay, key):
    if key not in replay:
        return
    arr = np.asarray(replay[key][:])
    flat = arr.reshape(arr.shape[0], -1)
    print(
        f"{key}: shape={arr.shape} "
        f"min={np.array2string(flat.min(axis=0), precision=4)} "
        f"max={np.array2string(flat.max(axis=0), precision=4)} "
        f"mean={np.array2string(flat.mean(axis=0), precision=4)}"
    )


def static_action_ratio(replay):
    if "action" not in replay:
        return None
    action = np.asarray(replay["action"][:])
    if len(action) < 2:
        return None
    delta = np.linalg.norm(np.diff(action[:, :6], axis=0), axis=1)
    grip_delta = np.abs(np.diff(action[:, 6], axis=0)) if action.shape[1] > 6 else 0
    static = (delta < 1e-4) & (grip_delta < 1e-4)
    return float(np.mean(static))


@click.command()
@click.argument("dataset_path")
@click.option("--max-samples", default=300, show_default=True, type=int)
def main(dataset_path, max_samples):
    dataset_path = pathlib.Path(dataset_path)
    zarr_path = dataset_path / "replay_buffer.zarr"
    video_dir = dataset_path / "videos"
    if not zarr_path.is_dir():
        raise click.ClickException(f"Missing replay buffer: {zarr_path}")
    if not video_dir.is_dir():
        raise click.ClickException(f"Missing videos dir: {video_dir}")

    replay = ReplayBuffer.create_from_path(str(zarr_path), mode="r")
    print(f"Dataset: {dataset_path}")
    print(f"ReplayBuffer: episodes={replay.n_episodes} steps={replay.n_steps}")
    print(f"Keys: {list(replay.keys())}")
    print(f"Episode lengths: {replay.episode_lengths[:]}")
    if replay.n_episodes:
        print(
            "Episode length summary: "
            f"min={int(replay.episode_lengths[:].min())} "
            f"mean={float(replay.episode_lengths[:].mean()):.1f} "
            f"max={int(replay.episode_lengths[:].max())}"
        )
    ratio = static_action_ratio(replay)
    if ratio is not None:
        print(f"Global static action ratio: {ratio:.1%}")
    for key in [
        "action",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
        "robot_eef_pose",
        "robot_gripper",
    ]:
        print_array_stats(replay, key)

    timestamps = replay["timestamp"][:]
    starts = replay.episode_ends[:] - replay.episode_lengths[:]
    for episode_idx, (start, end) in enumerate(zip(starts, replay.episode_ends[:])):
        t = timestamps[start:end]
        d = np.diff(t)
        duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        print(
            f"\nEpisode {episode_idx}: lowdim_steps={end-start} "
            f"lowdim_duration={duration:.3f}s "
            f"dt_mean={d.mean():.4f}s dt_min={d.min():.4f}s dt_max={d.max():.4f}s"
            if len(d)
            else f"\nEpisode {episode_idx}: lowdim_steps={end-start}"
        )

        ep_dir = video_dir / str(episode_idx)
        for video_path in sorted(ep_dir.glob("*.mp4"), key=lambda p: int(p.stem)):
            info = ffprobe_stream(video_path)
            diff = frame_diff_stats(video_path, max_samples=max_samples)
            nb_frames = int(info.get("nb_frames", 0))
            duration = float(info.get("duration", 0.0))
            bitrate = int(info.get("bit_rate", 0)) / 1e6
            print(
                f"  camera_{video_path.stem}: "
                f"{info.get('width')}x{info.get('height')} "
                f"frames={nb_frames} duration={duration:.3f}s "
                f"fps={nb_frames / max(duration, 1e-6):.2f} "
                f"bitrate={bitrate:.2f}Mbps"
            )
            if diff is not None:
                print(
                    f"    diff_mean={diff['diff_mean']:.2f} "
                    f"diff_p50={diff['diff_p50']:.2f} "
                    f"low<1={diff['low_diff_ratio']:.1%} "
                    f"static<0.5={diff['static_ratio']:.1%}"
                )


if __name__ == "__main__":
    main()
