import os
import sys
import time

import click
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.real_world.camera_config_util import apply_usb_camera_config


@click.command()
@click.option("--device", default=0, show_default=True, type=int)
@click.option("--duration", default=3.0, show_default=True, type=float)
def main(device, duration):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise click.ClickException(f"Failed to open /dev/video{device}")

    apply_usb_camera_config(cap, {"mode": "auto"})
    print(f"Reset /dev/video{device} to auto exposure and auto white balance.")
    print("Reading a few frames so the driver can settle...")

    start_time = time.time()
    n_frames = 0
    try:
        while time.time() - start_time < duration:
            ok, _ = cap.read()
            if not ok:
                raise click.ClickException("Failed to read frame from USB camera.")
            n_frames += 1
            time.sleep(1 / 30)
    finally:
        cap.release()

    print(f"Done. Frames read: {n_frames}")


if __name__ == "__main__":
    main()
