import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numcodecs
import numpy as np
from tqdm import tqdm
import zarr

from arx5_act.dataset import VIDEO_ALIGNMENT_SAFETY_FRAMES
from arx5_act.paths import ensure_project_paths

ensure_project_paths()
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs


def dataset_signature(dataset_path: Path, episode_ends, n_steps: int, camera_names):
    videos = []
    video_dir = dataset_path / "videos"
    for episode in range(len(episode_ends)):
        for camera_name in camera_names:
            camera_idx = int(camera_name.split("_")[-1])
            video_path = video_dir / str(episode) / f"{camera_idx}.mp4"
            stat = video_path.stat()
            videos.append(
                {
                    "episode": int(episode),
                    "camera": int(camera_idx),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    lengths = np.diff(np.concatenate([[0], np.asarray(episode_ends, dtype=np.int64)]))
    return {
        "n_steps": int(n_steps),
        "n_episodes": int(len(episode_ends)),
        "episode_ends": [int(x) for x in episode_ends],
        "videos": videos,
    }


def act_cache_path(cache_dir: Path, dataset_sig, camera_names, image_size, target_frequency, state_mode):
    config = {
        "camera_names": list(camera_names),
        "image_size": tuple(image_size),
        "target_frequency": target_frequency,
        "state_mode": state_mode,
        "safety_frames": VIDEO_ALIGNMENT_SAFETY_FRAMES,
        "dataset": dataset_sig,
        "version": 3,
    }
    digest = hashlib.md5(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / f"act_{state_mode}_cache_{digest}.zarr"


def write_act_cache(dp_group, output_path: Path, camera_names, state_mode: str, overwrite: bool):
    data = dp_group["data"]
    episode_ends = np.asarray(dp_group["meta"]["episode_ends"][:], dtype=np.int64)
    n_steps = int(data[camera_names[0]].shape[0])
    height = int(data[camera_names[0]].shape[1])
    width = int(data[camera_names[0]].shape[2])
    n_cameras = len(camera_names)

    if output_path.exists():
        root = zarr.open(str(output_path), mode="r")
        complete = bool(root.attrs.get("complete", False))
        if complete and not overwrite:
            print(f"ACT {state_mode} cache already exists: {output_path}")
            return
        shutil.rmtree(output_path)

    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
    root = zarr.open(str(output_path), mode="w")
    images = root.create_dataset(
        "images",
        shape=(n_steps, n_cameras, height, width, 3),
        chunks=(1, n_cameras, height, width, 3),
        dtype="uint8",
        compressor=compressor,
    )
    root.create_dataset("episode_valid_ends", data=episode_ends, dtype=np.int64)
    root.attrs["camera_names"] = list(camera_names)
    root.attrs["image_size"] = [width, height]
    root.attrs["state_mode"] = state_mode
    root.attrs["safety_frames"] = int(VIDEO_ALIGNMENT_SAFETY_FRAMES)
    root.attrs["complete"] = False

    batch_size = 128
    for start in tqdm(range(0, n_steps, batch_size), desc=f"Writing ACT {state_mode} cache"):
        end = min(n_steps, start + batch_size)
        batch = np.stack([data[name][start:end] for name in camera_names], axis=1)
        images[start:end] = batch
    root.attrs["complete"] = True
    print(f"Saved ACT {state_mode} cache: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dp-cache", required=True)
    parser.add_argument("--act-cache-dir", required=True)
    parser.add_argument("--target-frequency", type=float, default=20.0)
    parser.add_argument("--camera-names", nargs="+", default=["camera_0", "camera_1", "camera_2"])
    parser.add_argument("--state-mode", choices=["eef", "joint", "both"], default="both")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    register_codecs()
    dataset_path = Path(args.dataset_path)
    dp_cache = Path(args.dp_cache)
    act_cache_dir = Path(args.act_cache_dir)
    act_cache_dir.mkdir(parents=True, exist_ok=True)

    store = zarr.ZipStore(str(dp_cache), mode="r")
    try:
        dp_group = zarr.group(store=store)
        episode_ends = np.asarray(dp_group["meta"]["episode_ends"][:], dtype=np.int64)
        n_steps = int(dp_group["data"][args.camera_names[0]].shape[0])
        image_size = (
            int(dp_group["data"][args.camera_names[0]].shape[2]),
            int(dp_group["data"][args.camera_names[0]].shape[1]),
        )
        sig = dataset_signature(
            dataset_path=dataset_path,
            episode_ends=episode_ends,
            n_steps=n_steps,
            camera_names=args.camera_names,
        )
        modes = ["eef", "joint"] if args.state_mode == "both" else [args.state_mode]
        for mode in modes:
            output_path = act_cache_path(
                cache_dir=act_cache_dir,
                dataset_sig=sig,
                camera_names=args.camera_names,
                image_size=image_size,
                target_frequency=args.target_frequency,
                state_mode=mode,
            )
            write_act_cache(
                dp_group=dp_group,
                output_path=output_path,
                camera_names=args.camera_names,
                state_mode=mode,
                overwrite=args.overwrite,
            )
    finally:
        store.close()


if __name__ == "__main__":
    main()
