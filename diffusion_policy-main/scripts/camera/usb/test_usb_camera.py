import os
import sys
import time

import click
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.real_world.camera_config_util import load_and_apply_usb_camera_config


@click.command()
@click.option("--device", default=0, show_default=True, type=int)
@click.option("--width", default=1280, show_default=True, type=int)
@click.option("--height", default=720, show_default=True, type=int)
@click.option("--fps", default=30, show_default=True, type=int)
@click.option("--duration", default=10.0, show_default=True, type=float)
@click.option("--output", default="data_local/usb_camera_test.mp4", show_default=True)
@click.option("--camera-config", default=None, help="Path to usb_camera YAML config.")
@click.option("--no-auto-reset", is_flag=True, help="Do not force auto exposure/WB on startup.")
@click.option("--print-interval", default=1.0, show_default=True, type=float)
@click.option("--warmup", default=3.0, show_default=True, type=float)
@click.option("--lock-after-warmup", is_flag=True, help="Switch to manual using readback values after warmup.")
@click.option("--no-record", is_flag=True, help="Preview only without writing mp4.")
def main(
    device,
    width,
    height,
    fps,
    duration,
    output,
    camera_config,
    no_auto_reset,
    print_interval,
    warmup,
    lock_after_warmup,
    no_record,
):
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise click.ClickException(f"Failed to open /dev/video{device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if camera_config is not None:
        load_and_apply_usb_camera_config(cap, camera_config)
    elif not no_auto_reset:
        load_and_apply_usb_camera_config(cap, None)
        from diffusion_policy.real_world.camera_config_util import apply_usb_camera_config
        apply_usb_camera_config(cap, {"mode": "auto"})

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opened /dev/video{device}")
    print(f"Requested: {width}x{height}@{fps}")
    print(f"Actual: {actual_width}x{actual_height}@{actual_fps:.2f}")
    print(
        "Camera props: "
        f"auto_exposure={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.3f}, "
        f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.3f}, "
        f"gain={cap.get(cv2.CAP_PROP_GAIN):.3f}, "
        f"auto_wb={cap.get(cv2.CAP_PROP_AUTO_WB):.3f}, "
        f"white_balance={cap.get(cv2.CAP_PROP_WB_TEMPERATURE):.3f}, "
        f"brightness={cap.get(cv2.CAP_PROP_BRIGHTNESS):.3f}, "
        f"contrast={cap.get(cv2.CAP_PROP_CONTRAST):.3f}, "
        f"saturation={cap.get(cv2.CAP_PROP_SATURATION):.3f}"
    )

    warmup_start = time.time()
    warmup_frames = 0
    while time.time() - warmup_start < warmup:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise click.ClickException("Failed to read frame during warmup.")
        warmup_frames += 1
        cv2.putText(
            frame,
            f"warming up auto exposure... {time.time() - warmup_start:.1f}/{warmup:.1f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("usb_camera_test", frame)
        if cv2.pollKey() == ord("q"):
            break

    if lock_after_warmup:
        from diffusion_policy.real_world.camera_config_util import apply_usb_camera_config
        apply_usb_camera_config(cap, {
            "mode": "manual",
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "gain": cap.get(cv2.CAP_PROP_GAIN),
            "white_balance": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
            "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": cap.get(cv2.CAP_PROP_SATURATION),
        })
        print("Locked USB camera to readback values after warmup.")
    print(f"Warmup frames: {warmup_frames}")

    writer = None
    if not no_record:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        writer = cv2.VideoWriter(
            output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (actual_width, actual_height),
        )
        if not writer.isOpened():
            cap.release()
            raise click.ClickException(f"Failed to open video writer: {output}")

    start_time = time.time()
    last_print_time = start_time
    n_frames = 0
    try:
        while time.time() - start_time < duration:
            ok, frame = cap.read()
            if not ok:
                raise click.ClickException("Failed to read frame from USB camera.")
            now = time.time()
            props = {
                "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
                "gain": cap.get(cv2.CAP_PROP_GAIN),
                "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
                "white_balance": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
                "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
                "saturation": cap.get(cv2.CAP_PROP_SATURATION),
            }
            if now - last_print_time >= print_interval:
                last_print_time = now
                print(
                    "Camera props: "
                    f"auto_exposure={props['auto_exposure']:.3f}, "
                    f"exposure={props['exposure']:.3f}, "
                    f"gain={props['gain']:.3f}, "
                    f"auto_wb={props['auto_wb']:.3f}, "
                    f"white_balance={props['white_balance']:.3f}, "
                    f"brightness={props['brightness']:.3f}, "
                    f"contrast={props['contrast']:.3f}, "
                    f"saturation={props['saturation']:.3f}"
                )
            text = (
                f"AE={props['auto_exposure']:.2f} EXP={props['exposure']:.2f} "
                f"GAIN={props['gain']:.2f} AWB={props['auto_wb']:.2f} "
                f"WB={props['white_balance']:.0f}"
            )
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            n_frames += 1
            cv2.imshow("usb_camera_test", frame)
            if writer is not None:
                writer.write(frame.copy())
            if cv2.pollKey() == ord("q"):
                break
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    if writer is not None:
        print(f"Saved: {output}")
    else:
        print("Preview only; no video saved.")
    print(f"Frames: {n_frames}, observed FPS: {n_frames / max(elapsed, 1e-6):.2f}")


if __name__ == "__main__":
    main()
