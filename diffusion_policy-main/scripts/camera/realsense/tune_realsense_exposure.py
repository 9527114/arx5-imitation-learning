import os
import sys
import time

import click
import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.common.cv2_util import get_image_transform, optimal_row_cols
from diffusion_policy.real_world.multi_realsense import MultiRealsense


def make_grid(frames, row, col):
    h, w, c = frames[0].shape
    canvas = np.zeros((row * h, col * w, c), dtype=frames[0].dtype)
    for i, frame in enumerate(frames):
        r = i // col
        cidx = i % col
        canvas[r * h:(r + 1) * h, cidx * w:(cidx + 1) * w] = frame
    return canvas


def write_yaml(path, data):
    lines = [
        "realsense:",
        f"  mode: {data['mode']}",
        f"  resolution: [{data['resolution'][0]}, {data['resolution'][1]}]",
        f"  fps: {data['fps']}",
        f"  exposure: {data['exposure']}",
        f"  gain: {data['gain']}",
        f"  white_balance: {data['white_balance']}",
        f"  tuned_at: \"{data['tuned_at']}\"",
        "  camera_serial_numbers:",
    ]
    for serial in data["camera_serial_numbers"]:
        lines.append(f"    - \"{serial}\"")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


@click.command()
@click.option("--resolution", default="1280x720", show_default=True)
@click.option("--display-resolution", default="640x360", show_default=True)
@click.option("--fps", default=30, show_default=True, type=int)
@click.option("--exposure", default=250, show_default=True, type=int)
@click.option("--gain", default=16, show_default=True, type=int)
@click.option("--white-balance", default=4500, show_default=True, type=int)
@click.option("--exposure-step", default=25, show_default=True, type=int)
@click.option("--gain-step", default=2, show_default=True, type=int)
@click.option("--white-balance-step", default=100, show_default=True, type=int)
@click.option(
    "--output-dir",
    default="scripts/camera/realsense",
    show_default=True,
    help="Directory for HHMM RealSense setting YAML files.",
)
def main(
    resolution,
    display_resolution,
    fps,
    exposure,
    gain,
    white_balance,
    exposure_step,
    gain_step,
    white_balance_step,
    output_dir,
):
    capture_w, capture_h = [int(x) for x in resolution.lower().split("x")]
    display_w, display_h = [int(x) for x in display_resolution.lower().split("x")]

    color_tf = get_image_transform(
        input_res=(capture_w, capture_h),
        output_res=(display_w, display_h),
        bgr_to_rgb=False,
    )

    def transform(data):
        data["color"] = color_tf(data["color"])
        return data

    mode = "auto"
    camera_serial_numbers = []
    cv2.setNumThreads(1)

    with MultiRealsense(
        resolution=(capture_w, capture_h),
        capture_fps=fps,
        put_fps=fps,
        put_downsample=False,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        transform=transform,
        verbose=True,
    ) as realsense:
        camera_serial_numbers = list(realsense.cameras.keys())
        realsense.set_exposure(exposure=None, gain=None)
        realsense.set_white_balance(white_balance=None)

        rw, rh, col, row = optimal_row_cols(
            n_cameras=realsense.n_cameras,
            in_wh_ratio=display_w / display_h,
            max_resolution=(
                display_w * realsense.n_cameras,
                display_h * realsense.n_cameras,
            ),
        )
        del rw, rh

        print("")
        print("Controls:")
        print("  a: auto exposure + auto white balance")
        print("  m: manual exposure/gain/white balance")
        print("  q: quit, print settings, and save YAML")
        print("  exposure: [ / ]")
        print("  gain: - / =")
        print("  white balance: , / .")
        print("")

        out = None
        while True:
            out = realsense.get(out=out)
            frames = [value["color"] for value in out.values()]
            vis = make_grid(frames, row=row, col=col)
            text = (
                f"mode={mode} exposure={exposure} gain={gain} "
                f"white_balance={white_balance}"
            )
            cv2.putText(
                vis,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow("realsense_tuning", vis)

            key = cv2.pollKey()
            if key == ord("q"):
                break
            elif key == ord("a"):
                mode = "auto"
                realsense.set_exposure(exposure=None, gain=None)
                realsense.set_white_balance(white_balance=None)
            elif key == ord("m"):
                mode = "manual"
                realsense.set_exposure(exposure=exposure, gain=gain)
                realsense.set_white_balance(white_balance=white_balance)
            elif key == ord("["):
                exposure = max(1, exposure - exposure_step)
                if mode == "manual":
                    realsense.set_exposure(exposure=exposure, gain=gain)
            elif key == ord("]"):
                exposure = min(10000, exposure + exposure_step)
                if mode == "manual":
                    realsense.set_exposure(exposure=exposure, gain=gain)
            elif key == ord("-"):
                gain = max(0, gain - gain_step)
                if mode == "manual":
                    realsense.set_exposure(exposure=exposure, gain=gain)
            elif key == ord("="):
                gain = min(128, gain + gain_step)
                if mode == "manual":
                    realsense.set_exposure(exposure=exposure, gain=gain)
            elif key == ord(","):
                white_balance = max(2800, white_balance - white_balance_step)
                if mode == "manual":
                    realsense.set_white_balance(white_balance=white_balance)
            elif key == ord("."):
                white_balance = min(6500, white_balance + white_balance_step)
                if mode == "manual":
                    realsense.set_white_balance(white_balance=white_balance)

            time.sleep(1 / 60)

    cv2.destroyAllWindows()

    output_dir = os.path.join(ROOT_DIR, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tuned_at = time.strftime("%Y-%m-%d %H:%M:%S")
    output_path = os.path.join(output_dir, f"{time.strftime('%H%M')}.yaml")
    settings = {
        "mode": mode,
        "resolution": [capture_w, capture_h],
        "fps": fps,
        "exposure": exposure,
        "gain": gain,
        "white_balance": white_balance,
        "camera_serial_numbers": camera_serial_numbers,
        "tuned_at": tuned_at,
    }
    write_yaml(output_path, settings)

    print("")
    print("Recommended fixed RealSense settings:")
    print(f"  exposure: {exposure}")
    print(f"  gain: {gain}")
    print(f"  white_balance: {white_balance}")
    print(f"  camera_serial_numbers: {camera_serial_numbers}")
    print(f"  saved: {output_path}")
    print("")
    print("Use in collection:")
    print(f"  env.realsense.set_exposure(exposure={exposure}, gain={gain})")
    print(f"  env.realsense.set_white_balance(white_balance={white_balance})")


if __name__ == "__main__":
    main()
