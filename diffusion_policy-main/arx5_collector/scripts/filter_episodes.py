import shutil
from pathlib import Path

import click
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer


def parse_episode_ids(text):
    if text is None or text.strip() == "":
        return set()
    result = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output", "output_path", required=True, type=click.Path(file_okay=False))
@click.option("--drop", default="", help="Episode ids to drop, e.g. 7 or 3,7,12-14.")
@click.option("--keep", default="", help="Episode ids to keep. If set, --drop is ignored.")
@click.option("--overwrite", is_flag=True)
def main(input_path, output_path, drop, keep, overwrite):
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_zarr = input_path / "replay_buffer.zarr"
    input_videos = input_path / "videos"
    output_zarr = output_path / "replay_buffer.zarr"
    output_videos = output_path / "videos"

    if not input_zarr.is_dir():
        raise click.ClickException(f"Missing replay buffer: {input_zarr}")
    if output_path.exists():
        if not overwrite:
            raise click.ClickException(f"Output exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_videos.mkdir(parents=True, exist_ok=True)

    src = ReplayBuffer.create_from_path(str(input_zarr.absolute()), mode="r")
    keep_ids = parse_episode_ids(keep)
    drop_ids = parse_episode_ids(drop)
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

    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Original episodes: {src.n_episodes}")
    click.echo(f"Selected episodes: {selected}")
    click.echo(f"Output episodes: {dst.n_episodes}")
    click.echo(f"Output steps: {dst.n_steps}")


if __name__ == "__main__":
    main()
