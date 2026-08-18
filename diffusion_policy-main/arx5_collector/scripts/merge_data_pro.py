import shutil
import subprocess
import sys
from pathlib import Path

import click
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from arx5_collector.scripts.filter_episodes import parse_episode_ids


def parse_per_input_filters(values):
    """Parse filters like data_a=7,12 or data_b=0-3.

    The key may be either the dataset folder name or its full path string.
    """
    filters = {}
    for value in values:
        if "=" not in value:
            raise click.ClickException(
                f"Invalid filter '{value}'. Expected NAME=EPISODES, e.g. glue_motion_new=7,12."
            )
        key, text = value.split("=", 1)
        filters[key.strip()] = parse_episode_ids(text)
    return filters


def get_filter_for_path(filters, path: Path):
    return filters.get(str(path), filters.get(path.name, set()))


def select_episodes(src, input_path: Path, keep_filters, drop_filters):
    keep_ids = get_filter_for_path(keep_filters, input_path)
    drop_ids = get_filter_for_path(drop_filters, input_path)
    if keep_ids:
        return [idx for idx in range(src.n_episodes) if idx in keep_ids]
    return [idx for idx in range(src.n_episodes) if idx not in drop_ids]


def run_command(args, cwd: Path):
    click.echo("")
    click.echo("+ " + " ".join(str(x) for x in args))
    subprocess.run([str(x) for x in args], cwd=str(cwd), check=True)


def merge_datasets(input_paths, output_path: Path, keep_filters, drop_filters, overwrite: bool):
    if output_path.exists():
        if not overwrite:
            raise click.ClickException(f"Output exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)
    output_videos = output_path / "videos"
    output_zarr = output_path / "replay_buffer.zarr"
    output_path.mkdir(parents=True, exist_ok=True)
    output_videos.mkdir(parents=True, exist_ok=True)
    dst = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(output_zarr.absolute())))

    mapping = []
    total_steps = 0
    new_episode = 0
    for input_path in input_paths:
        input_path = Path(input_path)
        input_zarr = input_path / "replay_buffer.zarr"
        input_videos = input_path / "videos"
        if not input_zarr.is_dir():
            raise click.ClickException(f"Missing replay buffer: {input_zarr}")
        if not input_videos.is_dir():
            raise click.ClickException(f"Missing videos dir: {input_videos}")

        src = ReplayBuffer.create_from_path(str(input_zarr.absolute()), mode="r")
        selected = select_episodes(src, input_path, keep_filters, drop_filters)
        if not selected:
            click.echo(f"WARNING: no episodes selected from {input_path}", err=True)
            continue

        starts = src.episode_ends[:] - src.episode_lengths[:]
        for old_episode in selected:
            start = int(starts[old_episode])
            end = int(src.episode_ends[old_episode])
            episode = {key: src[key][start:end] for key in src.keys()}
            dst.add_episode(episode, compressors="disk")

            old_video_dir = input_videos / str(old_episode)
            new_video_dir = output_videos / str(new_episode)
            if not old_video_dir.is_dir():
                raise click.ClickException(f"Missing video dir: {old_video_dir}")
            shutil.copytree(old_video_dir, new_video_dir)
            mapping.append(
                {
                    "new": new_episode,
                    "source": str(input_path),
                    "source_episode": int(old_episode),
                    "steps": int(end - start),
                }
            )
            total_steps += int(end - start)
            new_episode += 1
        click.echo(
            f"Merged {input_path}: selected {len(selected)}/{src.n_episodes} episodes"
        )

    if new_episode == 0:
        raise click.ClickException("No episodes were merged.")

    mapping_path = output_path / "episode_source_map.tsv"
    with mapping_path.open("w", encoding="utf-8") as f:
        f.write("new_episode\tsource\tsource_episode\tsteps\n")
        for row in mapping:
            f.write(
                f"{row['new']}\t{row['source']}\t{row['source_episode']}\t{row['steps']}\n"
            )
    return new_episode, total_steps, mapping_path


@click.command()
@click.option(
    "--input",
    "input_paths",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Input dataset folder. Pass multiple times.",
)
@click.option("--output", "output_path", required=True, type=click.Path(file_okay=False))
@click.option(
    "--drop",
    "drop_values",
    multiple=True,
    help="Per-input drops, e.g. --drop glue_motion_new=7,12 or --drop data_local/foo=1-3.",
)
@click.option(
    "--keep",
    "keep_values",
    multiple=True,
    help="Per-input keeps. If set for an input, its drops are ignored.",
)
@click.option("--overwrite", is_flag=True)
@click.option("--target-frequency", default=20.0, show_default=True, type=float)
@click.option("--act-chunk-size", default=50, show_default=True, type=int)
@click.option("--image-width", default=320, show_default=True, type=int)
@click.option("--image-height", default=240, show_default=True, type=int)
@click.option("--skip-quality", is_flag=True)
@click.option("--skip-analyze", is_flag=True)
@click.option("--skip-dp-cache", is_flag=True)
@click.option("--skip-act-cache", is_flag=True)
def main(
    input_paths,
    output_path,
    drop_values,
    keep_values,
    overwrite,
    target_frequency,
    act_chunk_size,
    image_width,
    image_height,
    skip_quality,
    skip_analyze,
    skip_dp_cache,
    skip_act_cache,
):
    repo_root = Path(__file__).resolve().parents[3]
    dp_root = Path(__file__).resolve().parents[2]
    input_paths = [Path(x) for x in input_paths]
    output_path = Path(output_path)
    keep_filters = parse_per_input_filters(keep_values)
    drop_filters = parse_per_input_filters(drop_values)

    n_episodes, n_steps, mapping_path = merge_datasets(
        input_paths=input_paths,
        output_path=output_path,
        keep_filters=keep_filters,
        drop_filters=drop_filters,
        overwrite=overwrite,
    )
    click.echo("")
    click.echo(f"Output: {output_path}")
    click.echo(f"Output episodes: {n_episodes}")
    click.echo(f"Output steps: {n_steps}")
    click.echo(f"Episode source map: {mapping_path}")

    if not skip_quality:
        run_command(
            [
                sys.executable,
                "-m",
                "arx5_collector.scripts.inspect_episode_quality",
                str(output_path),
            ],
            cwd=dp_root,
        )
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
