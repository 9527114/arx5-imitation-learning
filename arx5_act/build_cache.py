import argparse
from pathlib import Path

from arx5_act.dataset import Arx5ActDataset, compute_norm_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--camera-names", nargs="+", default=["camera_0", "camera_1", "camera_2"])
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--target-frequency", type=float, default=20.0)
    parser.add_argument("--state-mode", default="eef", choices=["eef", "joint"])
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    dataset_path = str(Path(args.dataset_path).absolute())
    cache_dir = None if args.cache_dir is None else str(Path(args.cache_dir).absolute())
    stats = compute_norm_stats(
        dataset_path,
        target_frequency=args.target_frequency,
        camera_names=args.camera_names,
        state_mode=args.state_mode,
    )
    dataset = Arx5ActDataset(
        dataset_path=dataset_path,
        norm_stats=stats,
        camera_names=args.camera_names,
        chunk_size=args.chunk_size,
        image_size=(args.image_width, args.image_height),
        episode_indices=None,
        use_cache=True,
        target_frequency=args.target_frequency,
        state_mode=args.state_mode,
        cache_dir=cache_dir,
    )
    print(f"ACT cache ready: {dataset._cache_path()}")
    print(f"State mode: {args.state_mode}")
    print(f"Samples: {len(dataset)}")
    dataset.close()


if __name__ == "__main__":
    main()
