import pathlib

import click
import numpy as np
import zarr


OLD_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
)


def write_array(data_group, name, data, overwrite):
    if name in data_group:
        if not overwrite:
            return False
        del data_group[name]
    data_group.array(
        name=name,
        data=data,
        shape=data.shape,
        chunks=data.shape,
        dtype=data.dtype,
        compressor=None,
    )
    return True


@click.command()
@click.argument("dataset_path")
@click.option("--overwrite", is_flag=True, help="Replace old-schema arrays if they already exist.")
def main(dataset_path, overwrite):
    """Add old ARX5 DP low-dimensional keys to a current collector dataset."""
    dataset_path = pathlib.Path(dataset_path)
    zarr_path = dataset_path / "replay_buffer.zarr"
    if not zarr_path.is_dir():
        raise click.ClickException(f"Missing replay buffer: {zarr_path}")

    root = zarr.open(str(zarr_path), mode="a")
    data_group = root["data"]
    keys = set(data_group.keys())
    if set(OLD_KEYS).issubset(keys) and not overwrite:
        print("Old DP schema keys already exist. Nothing changed.")
        return

    if "robot_eef_pose" not in data_group:
        raise click.ClickException("Missing robot_eef_pose; cannot create robot0_eef_* keys.")
    if "robot_gripper" not in data_group:
        raise click.ClickException("Missing robot_gripper; cannot create robot0_gripper_width.")

    eef_pose = np.asarray(data_group["robot_eef_pose"][:], dtype=np.float64)
    gripper = np.asarray(data_group["robot_gripper"][:], dtype=np.float64)
    if eef_pose.ndim != 2 or eef_pose.shape[1] != 6:
        raise click.ClickException(f"robot_eef_pose should be (T, 6), got {eef_pose.shape}")
    if gripper.ndim == 1:
        gripper = gripper[:, None]

    changed = []
    if write_array(data_group, "robot0_eef_pos", eef_pose[:, :3], overwrite):
        changed.append("robot0_eef_pos")
    if write_array(data_group, "robot0_eef_rot_axis_angle", eef_pose[:, 3:], overwrite):
        changed.append("robot0_eef_rot_axis_angle")
    if write_array(data_group, "robot0_gripper_width", gripper, overwrite):
        changed.append("robot0_gripper_width")

    if changed:
        print("Added old DP schema keys:")
        for key in changed:
            print(f"  {key}: {data_group[key].shape}")
    else:
        print("No arrays changed.")


if __name__ == "__main__":
    main()
