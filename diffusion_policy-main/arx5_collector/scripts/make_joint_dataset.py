import shutil
from pathlib import Path

import click
import numpy as np
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer


def make_joint_action(replay):
    if "robot_joint" not in replay:
        raise click.ClickException("Input dataset is missing robot_joint.")
    if "robot_gripper" in replay:
        gripper = np.asarray(replay["robot_gripper"][:], dtype=np.float32)
    elif "robot0_gripper_width" in replay:
        gripper = np.asarray(replay["robot0_gripper_width"][:], dtype=np.float32)
    else:
        raise click.ClickException("Input dataset is missing robot_gripper/robot0_gripper_width.")
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    joint = np.asarray(replay["robot_joint"][:], dtype=np.float32)
    return np.concatenate([joint[:, :6], gripper[:, :1]], axis=-1)


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output", "output_path", required=True, type=click.Path(file_okay=False))
@click.option("--overwrite", is_flag=True)
def main(input_path, output_path, overwrite):
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_zarr = input_path / "replay_buffer.zarr"
    output_zarr = output_path / "replay_buffer.zarr"

    if not input_zarr.is_dir():
        raise click.ClickException(f"Missing replay buffer: {input_zarr}")
    if output_path.exists():
        if not overwrite:
            raise click.ClickException(f"Output exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    src = ReplayBuffer.create_from_path(str(input_zarr.absolute()), mode="r")
    joint_action = make_joint_action(src)

    shutil.copytree(input_path, output_path)
    root = zarr.open(str(output_zarr.absolute()), mode="a")
    action = root["data"]["action"]
    if action.shape != joint_action.shape:
        raise click.ClickException(
            f"Action shape mismatch: output action {action.shape}, joint action {joint_action.shape}"
        )
    action[:] = joint_action

    out = ReplayBuffer.create_from_path(str(output_zarr.absolute()), mode="r")
    action_np = np.asarray(out["action"][:], dtype=np.float32)
    max_abs_error = float(np.max(np.abs(action_np - joint_action))) if len(action_np) else 0.0
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Episodes: {out.n_episodes}")
    click.echo(f"Steps: {out.n_steps}")
    click.echo("Action: [robot_joint(6), robot_gripper(1)]")
    click.echo(f"max_abs_error: {max_abs_error:.8f}")


if __name__ == "__main__":
    main()
