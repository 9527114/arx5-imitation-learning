import os
import sys
import time

import click
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.real_world.camera_config_util import apply_usb_camera_config


def write_yaml(path, data):
    lines = [
        "usb_camera:",
        f"  mode: {data['mode']}",
        f"  device: {data['device']}",
        f"  resolution: [{data['resolution'][0]}, {data['resolution'][1]}]",
        f"  fps: {data['fps']}",
        f"  exposure: {data['exposure']}",
        f"  gain: {data['gain']}",
        f"  white_balance: {data['white_balance']}",
        f"  brightness: {data['brightness']}",
        f"  contrast: {data['contrast']}",
        f"  saturation: {data['saturation']}",
        f"  tuned_at: \"{data['tuned_at']}\"",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def readback(cap):
    return {
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "gain": cap.get(cv2.CAP_PROP_GAIN),
        "white_balance": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
        "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
        "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
        "saturation": cap.get(cv2.CAP_PROP_SATURATION),
    }


@click.command()
@click.option("--device", default=0, show_default=True, type=int)
@click.option("--width", default=1280, show_default=True, type=int)
@click.option("--height", default=720, show_default=True, type=int)
@click.option("--fps", default=30, show_default=True, type=int)
@click.option("--exposure", default=-6.0, show_default=True, type=float)
@click.option("--gain", default=0.0, show_default=True, type=float)
@click.option("--white-balance", default=4500.0, show_default=True, type=float)
@click.option("--brightness", default=0.0, show_default=True, type=float)
@click.option("--contrast", default=32.0, show_default=True, type=float)
@click.option("--saturation", default=64.0, show_default=True, type=float)
@click.option("--exposure-step", default=1.0, show_default=True, type=float)
@click.option("--gain-step", default=1.0, show_default=True, type=float)
@click.option("--white-balance-step", default=100.0, show_default=True, type=float)
@click.option("--image-step", default=2.0, show_default=True, type=float)
@click.option(
    "--output-dir",
    default="scripts/camera/usb",
    show_default=True,
    help="Directory for HHMM USB camera setting YAML files.",
)
@click.option(
    "--current-config",
    default="scripts/camera/usb/current.yaml",
    show_default=True,
    help="Also write the latest settings to this stable config path.",
)
def main(
    device,
    width,
    height,
    fps,
    exposure,
    gain,
    white_balance,
    brightness,
    contrast,
    saturation,
    exposure_step,
    gain_step,
    white_balance_step,
    image_step,
    output_dir,
    current_config,
):
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise click.ClickException(f"Failed to open /dev/video{device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    mode = "auto"
    apply_usb_camera_config(cap, {"mode": "auto"})

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opened /dev/video{device}")
    print(f"Requested: {width}x{height}@{fps}")
    print(f"Actual: {actual_width}x{actual_height}@{actual_fps:.2f}")
    print("")
    print("Controls:")
    print("  a: auto exposure + auto white balance")
    print("  m: manual exposure/gain/white balance")
    print("  q: quit, print settings, and save YAML")
    print("  exposure: [ / ]")
    print("  gain: - / =")
    print("  white balance: , / .")
    print("  brightness: 1 / 2")
    print("  contrast: 3 / 4")
    print("  saturation: 5 / 6")
    print("")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise click.ClickException("Failed to read frame from USB camera.")

            actual = readback(cap)
            text = (
                f"mode={mode} exp={exposure:g} gain={gain:g} wb={white_balance:g} "
                f"br={brightness:g} ct={contrast:g} sat={saturation:g}"
            )
            actual_text = (
                f"actual exp={actual['exposure']:.2f} gain={actual['gain']:.2f} "
                f"wb={actual['white_balance']:.2f}"
            )
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, actual_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("usb_camera_tuning", frame)

            key = cv2.pollKey()
            if key == ord("q"):
                break
            elif key == ord("a"):
                mode = "auto"
                apply_usb_camera_config(cap, {"mode": "auto"})
            elif key == ord("m"):
                mode = "manual"
                apply_usb_camera_config(cap, {
                    "mode": mode,
                    "exposure": exposure,
                    "gain": gain,
                    "white_balance": white_balance,
                    "brightness": brightness,
                    "contrast": contrast,
                    "saturation": saturation,
                })
            elif key == ord("["):
                exposure -= exposure_step
            elif key == ord("]"):
                exposure += exposure_step
            elif key == ord("-"):
                gain = max(0.0, gain - gain_step)
            elif key == ord("="):
                gain += gain_step
            elif key == ord(","):
                white_balance = max(2000.0, white_balance - white_balance_step)
            elif key == ord("."):
                white_balance = min(8000.0, white_balance + white_balance_step)
            elif key == ord("1"):
                brightness -= image_step
            elif key == ord("2"):
                brightness += image_step
            elif key == ord("3"):
                contrast = max(0.0, contrast - image_step)
            elif key == ord("4"):
                contrast += image_step
            elif key == ord("5"):
                saturation = max(0.0, saturation - image_step)
            elif key == ord("6"):
                saturation += image_step
            else:
                time.sleep(1 / 60)
                continue

            if mode == "manual":
                apply_usb_camera_config(cap, {
                    "mode": mode,
                    "exposure": exposure,
                    "gain": gain,
                    "white_balance": white_balance,
                    "brightness": brightness,
                    "contrast": contrast,
                    "saturation": saturation,
                })
            time.sleep(1 / 60)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    output_dir = os.path.join(ROOT_DIR, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tuned_at = time.strftime("%Y-%m-%d %H:%M:%S")
    output_path = os.path.join(output_dir, f"{time.strftime('%H%M')}.yaml")
    settings = {
        "mode": mode,
        "device": device,
        "resolution": [actual_width, actual_height],
        "fps": fps,
        "exposure": exposure,
        "gain": gain,
        "white_balance": white_balance,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "tuned_at": tuned_at,
    }
    write_yaml(output_path, settings)
    current_config_path = os.path.join(ROOT_DIR, current_config)
    os.makedirs(os.path.dirname(current_config_path), exist_ok=True)
    write_yaml(current_config_path, settings)

    print("")
    print("Recommended fixed USB camera settings:")
    for key in ["device", "exposure", "gain", "white_balance", "brightness", "contrast", "saturation"]:
        print(f"  {key}: {settings[key]}")
    print(f"  saved: {output_path}")
    print(f"  current: {current_config_path}")


if __name__ == "__main__":
    main()
