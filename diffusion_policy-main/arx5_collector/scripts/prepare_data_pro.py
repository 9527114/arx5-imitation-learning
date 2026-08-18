import shutil
import subprocess
import sys
from pathlib import Path

import click
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from arx5_collector.scripts.filter_episodes import parse_episode_ids


def copy_selected_episodes(input_path: Path, output_path: Path, keep_text: str, drop_text: str, overwrite: bool):
    input_zarr = input_path / "replay_buffer.zarr"
    input_videos = input_path / "videos"
    output_zarr = output_path / "replay_buffer.zarr"
    output_videos = output_path / "videos"

    if not input_zarr.is_dir():
        raise click.ClickException(f"Missing replay buffer: {input_zarr}")
    if not input_videos.is_dir():
        raise click.ClickException(f"Missing videos dir: {input_videos}")
    if output_path.exists():
        if not overwrite:
            raise click.ClickException(f"Output exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    output_videos.mkdir(parents=True, exist_ok=True)

    src = ReplayBuffer.create_from_path(str(input_zarr.absolute()), mode="r")
    keep_ids = parse_episode_ids(keep_text)
    drop_ids = parse_episode_ids(drop_text)
    if keep_ids:
        selected = [idx for idx in range(src.n_episodes) if idx in keep_ids]
    else:
        selected = [idx for idx in range(src.n_episodes) if idx not in drop_ids]
    if not selected:
        raise click.ClickException("No episodes selected.")

    dst = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(output_zarr.absolute())))
    starts = src.episode_ends[:] - src.episode_lengths[:]
    for new_id, old_id in enumerate(selected):
        start = int(starts[old_id])
        end = int(src.episode_ends[old_id])
        episode = {key: src[key][start:end] for key in src.keys()}
        dst.add_episode(episode, compressors="disk")

        old_video_dir = input_videos / str(old_id)
        new_video_dir = output_videos / str(new_id)
        if old_video_dir.is_dir():
            shutil.copytree(old_video_dir, new_video_dir)
        else:
            click.echo(f"WARNING: missing video dir for episode {old_id}: {old_video_dir}", err=True)
    return selected, dst.n_episodes, dst.n_steps


def run_command(args, cwd: Path):
    click.echo("")
    click.echo("+ " + " ".join(str(x) for x in args))
    subprocess.run([str(x) for x in args], cwd=str(cwd), check=True)


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output", "output_path", required=True, type=click.Path(file_okay=False))
@click.option("--drop", default="", help="Episode ids to drop, e.g. 7 or 3,7,12-14.")
@click.option("--keep", default="", help="Episode ids to keep. If set, --drop is ignored.")
@click.option("--overwrite", is_flag=True)
@click.option("--target-frequency", default=20.0, show_default=True, type=float)
@click.option("--act-chunk-size", default=50, show_default=True, type=int)
@click.option("--image-width", default=320, show_default=True, type=int)
@click.option("--image-height", default=240, show_default=True, type=int)
@click.option("--skip-analyze", is_flag=True)
@click.option("--skip-dp-cache", is_flag=True)
@click.option("--skip-act-cache", is_flag=True)
def main(
    input_path,
    output_path,
    drop,
    keep,
    overwrite,
    target_frequency,
    act_chunk_size,
    image_width,
    image_height,
    skip_analyze,
    skip_dp_cache,
    skip_act_cache,
):
    repo_root = Path(__file__).resolve().parents[3]
    dp_root = Path(__file__).resolve().parents[2]
    input_path = Path(input_path)
    output_path = Path(output_path)

    selected, n_episodes, n_steps = copy_selected_episodes(
        input_path=input_path,
        output_path=output_path,
        keep_text=keep,
        drop_text=drop,
        overwrite=overwrite,
    )
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Selected episodes: {selected}")
    click.echo(f"Output episodes: {n_episodes}")
    click.echo(f"Output steps: {n_steps}")

    if not skip_analyze:
        run_command(
            [
                sys.executable,
                "-m",
                "arx5_collector.scripts.analyze_recordings",
                str(output_path),
            ],
            cwd=dp_root,
        )
    if not skip_dp_cache:
        run_command(
            [
                sys.executable,
                "-m",
                "arx5_collector.scripts.inspect_training_dataset",
                "--dataset-path",
                str(output_path),
                "--target-frequency",
                str(target_frequency),
            ],
            cwd=dp_root,
        )
    if not skip_act_cache:
        run_command(
            [
                sys.executable,
                "-m",
                "arx5_act.build_cache",
                "--dataset-path",
                str(output_path),
                "--target-frequency",
                str(target_frequency),
                "--chunk-size",
                str(act_chunk_size),
                "--image-width",
                str(image_width),
                "--image-height",
                str(image_height),
            ],
            cwd=repo_root,
        )


if __name__ == "__main__":
    main()
