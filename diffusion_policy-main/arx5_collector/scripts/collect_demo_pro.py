import subprocess
import sys
from pathlib import Path

import click


def run_command(args, cwd: Path):
    click.echo("")
    click.echo("+ " + " ".join(str(x) for x in args))
    subprocess.run([str(x) for x in args], cwd=str(cwd), check=True)


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--output", "-o", required=True)
@click.option("--target-frequency", default=20.0, show_default=True, type=float)
@click.option("--act-chunk-size", default=50, show_default=True, type=int)
@click.option("--image-width", default=320, show_default=True, type=int)
@click.option("--image-height", default=240, show_default=True, type=int)
@click.option("--skip-dp-cache", is_flag=True)
@click.option("--skip-act-cache", is_flag=True)
@click.pass_context
def main(
    ctx,
    output,
    target_frequency,
    act_chunk_size,
    image_width,
    image_height,
    skip_dp_cache,
    skip_act_cache,
):
    """Collect ARX5 demos, then build DP and ACT caches.

    Unknown arguments are forwarded to arx5_collector.scripts.collect_demo.
    """
    repo_root = Path(__file__).resolve().parents[3]
    dp_root = Path(__file__).resolve().parents[2]
    output_path = Path(output)

    collect_cmd = [
        sys.executable,
        "-m",
        "arx5_collector.scripts.collect_demo",
        "--output",
        str(output_path),
        "--data-frequency",
        str(target_frequency),
    ]
    collect_cmd.extend(ctx.args)
    run_command(collect_cmd, cwd=dp_root)

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
