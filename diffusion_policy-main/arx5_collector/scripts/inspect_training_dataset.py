import click
import numpy as np
from omegaconf import OmegaConf

from diffusion_policy.dataset.arx5_image_dataset import Arx5ImageDataset


def static_action_ratio(replay):
    if "action" not in replay or replay.n_steps < 2:
        return None
    action = np.asarray(replay["action"][:], dtype=np.float64)
    if action.shape[1] < 7:
        return None
    pose_delta = np.linalg.norm(np.diff(action[:, :6], axis=0), axis=1)
    grip_delta = np.abs(np.diff(action[:, 6], axis=0))
    return float(np.mean((pose_delta < 1e-4) & (grip_delta < 1e-4)))


def print_training_readiness(replay, horizon):
    required_keys = [
        "camera_0",
        "camera_1",
        "camera_2",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
        "action",
    ]
    problems = []
    warnings = []
    missing = [key for key in required_keys if key not in replay]
    if missing:
        problems.append(f"missing keys: {missing}")
    if replay.n_episodes == 0:
        problems.append("no episodes")
    if replay.n_steps == 0:
        problems.append("no steps")

    if replay.n_episodes > 0:
        lengths = replay.episode_lengths[:]
        short = int(np.sum(lengths < horizon))
        if short:
            problems.append(f"{short} episode(s) shorter than horizon={horizon}")
        if replay.n_episodes < 10:
            warnings.append(
                f"only {replay.n_episodes} episode(s); use this for pipeline tests, not final training"
            )

    for key in required_keys:
        if key in replay:
            arr = np.asarray(replay[key][:])
            if not np.all(np.isfinite(arr)):
                problems.append(f"{key} contains NaN or Inf")

    ratio = static_action_ratio(replay)
    if ratio is not None and ratio > 0.4:
        warnings.append(f"high static action ratio: {ratio:.1%}")

    print("\nTraining readiness:")
    if problems:
        print("  FAIL")
        for item in problems:
            print(f"  - {item}")
    else:
        print("  PASS")
    for item in warnings:
        print(f"  warning: {item}")


@click.command()
@click.option(
    "--config",
    default="diffusion_policy/config/task/arx5_image.yaml",
    show_default=True,
)
@click.option("--dataset-path", default=None)
@click.option("--no-cache", is_flag=True)
@click.option("--cache-dir", default=None, type=click.Path(file_okay=False))
@click.option("--cache-path", default=None, type=click.Path(dir_okay=False))
@click.option("--target-frequency", default=None, type=float)
@click.option("--trim-static", is_flag=True)
def main(config, dataset_path, no_cache, cache_dir, cache_path, target_frequency, trim_static):
    raw_cfg = OmegaConf.load(config)
    cfg = OmegaConf.create({"task": raw_cfg}).task
    dataset_cfg = cfg.dataset
    if dataset_path is None:
        dataset_path = cfg.dataset_path
    if target_frequency is None:
        target_frequency = dataset_cfg.get("target_frequency", None)

    dataset = Arx5ImageDataset(
        shape_meta=cfg.shape_meta,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        cache_path=cache_path,
        horizon=16,
        pad_before=1,
        pad_after=7,
        n_obs_steps=2,
        n_latency_steps=0,
        use_cache=not no_cache,
        seed=42,
        val_ratio=dataset_cfg.get("val_ratio", 0.0),
        max_train_episodes=dataset_cfg.get("max_train_episodes", None),
        target_frequency=target_frequency,
        delta_action=dataset_cfg.get("delta_action", False),
        trim_static_start_end=trim_static
        or dataset_cfg.get("trim_static_start_end", False),
        static_pos_threshold=dataset_cfg.get("static_pos_threshold", 1e-4),
        static_rot_threshold=dataset_cfg.get("static_rot_threshold", 1e-4),
        static_gripper_threshold=dataset_cfg.get("static_gripper_threshold", 1e-4),
        static_pad_before=dataset_cfg.get("static_pad_before", 2),
        static_pad_after=dataset_cfg.get("static_pad_after", 2),
        min_episode_steps=dataset_cfg.get("min_episode_steps", 16),
    )

    replay = dataset.replay_buffer
    print(f"Dataset path: {dataset_path}")
    print(f"Samples: {len(dataset)}")
    print(f"Episodes: {replay.n_episodes}")
    print(f"Steps: {replay.n_steps}")
    lengths = replay.episode_lengths[:]
    if len(lengths):
        print(
            "Episode length summary:",
            f"min={int(lengths.min())}",
            f"mean={float(lengths.mean()):.1f}",
            f"max={int(lengths.max())}",
        )
    else:
        print("Episode length summary: empty")
    print("Keys:", list(replay.keys()))
    for key in [
        "action",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
    ]:
        if key not in replay:
            continue
        arr = np.asarray(replay[key][:])
        flat = arr.reshape(arr.shape[0], -1)
        print(
            f"{key}: shape={arr.shape} "
            f"min={np.array2string(flat.min(axis=0), precision=4)} "
            f"max={np.array2string(flat.max(axis=0), precision=4)} "
            f"mean={np.array2string(flat.mean(axis=0), precision=4)}"
        )
    print_training_readiness(replay, horizon=16)


if __name__ == "__main__":
    main()
