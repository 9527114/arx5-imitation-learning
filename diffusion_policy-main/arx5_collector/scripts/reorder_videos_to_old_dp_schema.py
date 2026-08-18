import pathlib

import click


@click.command()
@click.argument("dataset_path")
@click.option("--apply", "apply_changes", is_flag=True, help="Actually rename video files.")
def main(dataset_path, apply_changes):
    """Reorder current collector videos to the old ARX5 DP camera convention.

    Current collector convention before the schema cleanup:
      0.mp4 = RealSense 0
      1.mp4 = RealSense 1
      2.mp4 = USB

    Old ARX5 DP convention:
      0.mp4 = USB
      1.mp4 = RealSense 0
      2.mp4 = RealSense 1
    """
    dataset_path = pathlib.Path(dataset_path)
    video_root = dataset_path / "videos"
    if not video_root.is_dir():
        raise click.ClickException(f"Missing videos dir: {video_root}")

    episode_dirs = sorted(
        [path for path in video_root.iterdir() if path.is_dir()],
        key=lambda path: int(path.name) if path.name.isdigit() else path.name,
    )
    if not episode_dirs:
        raise click.ClickException(f"No episode video dirs found in {video_root}")

    print("Reorder mapping:")
    print("  old 2.mp4 -> new 0.mp4  (USB)")
    print("  old 0.mp4 -> new 1.mp4  (RealSense 0)")
    print("  old 1.mp4 -> new 2.mp4  (RealSense 1)")
    print("Mode:", "APPLY" if apply_changes else "DRY-RUN")

    missing = []
    for ep_dir in episode_dirs:
        paths = [ep_dir / f"{idx}.mp4" for idx in range(3)]
        if not all(path.exists() for path in paths):
            missing.append(str(ep_dir))
            continue
        if not apply_changes:
            print(f"{ep_dir}: would reorder")
            continue

        tmp_paths = [ep_dir / f"__tmp_camera_{idx}.mp4" for idx in range(3)]
        for src, tmp in zip(paths, tmp_paths):
            if tmp.exists():
                raise click.ClickException(f"Temporary file already exists: {tmp}")
            src.rename(tmp)
        tmp_paths[2].rename(ep_dir / "0.mp4")
        tmp_paths[0].rename(ep_dir / "1.mp4")
        tmp_paths[1].rename(ep_dir / "2.mp4")
        print(f"{ep_dir}: reordered")

    if missing:
        print("Skipped episodes with missing 0/1/2.mp4:")
        for item in missing:
            print(f"  {item}")
    if not apply_changes:
        print("No files changed. Add --apply to rename files.")


if __name__ == "__main__":
    main()
