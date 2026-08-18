import numpy as np
import click

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.arx5_image_dataset import (
    _copy_selected_steps,
    _get_episode_sample_indices,
    _make_joint_action,
    _trim_sample_indices_to_available_video,
)


@click.command()
@click.option("--dataset-path", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--target-frequency", default=20.0, show_default=True, type=float)
@click.option("--num-examples", default=5, show_default=True, type=int)
def main(dataset_path, target_frequency, num_examples):
    replay = ReplayBuffer.create_from_path(f"{dataset_path}/replay_buffer.zarr", mode="r")
    print(f"Dataset: {dataset_path}")
    print(f"Episodes: {replay.n_episodes}")
    print(f"Steps: {replay.n_steps}")
    print(f"Keys: {list(replay.keys())}")

    required = ["robot_joint", "robot_gripper", "timestamp"]
    missing = [key for key in required if key not in replay]
    if missing:
        raise RuntimeError(f"Missing required keys for DP-Joint: {missing}")

    raw_action = _make_joint_action(replay)
    robot_joint = np.asarray(replay["robot_joint"][:], dtype=np.float32)
    robot_gripper = np.asarray(replay["robot_gripper"][:], dtype=np.float32)
    if robot_gripper.ndim == 1:
        robot_gripper = robot_gripper[:, None]
    expected = np.concatenate([robot_joint[:, :6], robot_gripper[:, :1]], axis=-1)
    max_abs_error = float(np.max(np.abs(raw_action - expected))) if len(raw_action) else 0.0

    print("")
    print("Raw DP-Joint action check")
    print(f"  action shape: {raw_action.shape}")
    print(f"  expected shape: {expected.shape}")
    print(f"  max_abs_error(action - [robot_joint, robot_gripper]): {max_abs_error:.8f}")
    print(f"  action min: {np.array2string(raw_action.min(axis=0), precision=5)}")
    print(f"  action max: {np.array2string(raw_action.max(axis=0), precision=5)}")
    print(f"  action mean: {np.array2string(raw_action.mean(axis=0), precision=5)}")

    episode_sample_indices = _get_episode_sample_indices(
        dataset_path=dataset_path,
        target_frequency=target_frequency,
    )
    episode_sample_indices = _trim_sample_indices_to_available_video(
        dataset_path=dataset_path,
        image_keys=["camera_0", "camera_1", "camera_2"],
        episode_sample_indices=episode_sample_indices,
    )
    starts = replay.episode_ends[:] - replay.episode_lengths[:]
    episode_slices = []
    for start, end, indices in zip(starts, replay.episode_ends[:], episode_sample_indices):
        episode_slices.append((int(start), int(end), np.asarray(indices, dtype=np.int64)))
    sampled = _copy_selected_steps(replay, episode_slices)
    sampled_action = _make_joint_action(sampled)

    print("")
    print("Training-frequency DP-Joint action check")
    print(f"  target_frequency: {target_frequency:g}Hz")
    print(f"  sampled episodes: {sampled.n_episodes}")
    print(f"  sampled steps: {sampled.n_steps}")
    print(f"  sampled action shape: {sampled_action.shape}")
    print(f"  sampled action min: {np.array2string(sampled_action.min(axis=0), precision=5)}")
    print(f"  sampled action max: {np.array2string(sampled_action.max(axis=0), precision=5)}")

    print("")
    print("Example training sequences")
    horizon = 16
    n_obs_steps = 2
    n = min(int(num_examples), max(0, sampled.n_steps - horizon))
    for idx in np.linspace(0, max(0, sampled.n_steps - horizon - 1), num=max(1, n), dtype=int):
        obs_joint = np.asarray(sampled["robot_joint"][idx : idx + n_obs_steps], dtype=np.float32)
        obs_gripper = np.asarray(sampled["robot_gripper"][idx : idx + n_obs_steps], dtype=np.float32)
        action = sampled_action[idx : idx + horizon]
        print(
            f"  idx={idx}",
            f"obs.robot_joint={obs_joint.shape}",
            f"obs.robot_gripper={obs_gripper.shape}",
            f"action={action.shape}",
            f"action0={np.array2string(action[0], precision=4)}",
        )

    if max_abs_error > 1e-6:
        raise RuntimeError("DP-Joint action construction mismatch.")
    print("")
    print("PASS: DP-Joint action is exactly [robot_joint(6), robot_gripper(1)].")


if __name__ == "__main__":
    main()
