import os
import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np

from arx5_collector.camera import ThreeCameraRecorder


def resize_frame(frame, width=426, height=240):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def make_preview(realsense_data, usb_data):
    frames = []
    for idx in sorted(realsense_data.keys()):
        frame = realsense_data[idx]["color"]
        frames.append(resize_frame(frame))
    frames.append(resize_frame(usb_data["color"]))
    return np.concatenate(frames, axis=1)


@click.command()
@click.option("--output", "-o", default="data_local/arx5_camera_test", show_default=True)
@click.option("--realsense-config", default=None)
@click.option("--usb-config", default=None)
@click.option("--usb-device", default=0, show_default=True, type=int)
@click.option("--duration", default=0.0, show_default=True, type=float, help="0 means record until q.")
@click.option("--fps", default=30, show_default=True, type=int)
@click.option("--width", default=1280, show_default=True, type=int)
@click.option("--height", default=720, show_default=True, type=int)
def main(output, realsense_config, usb_config, usb_device, duration, fps, width, height):
    cv2.setNumThreads(1)
    episode_dir = os.path.abspath(os.path.join(output, "videos", "0"))
    os.makedirs(episode_dir, exist_ok=True)

    with SharedMemoryManager() as shm_manager:
        cameras = ThreeCameraRecorder(
            shm_manager=shm_manager,
            realsense_config=realsense_config,
            usb_config=usb_config,
            usb_device=usb_device,
            resolution=(width, height),
            fps=fps,
            verbose=True,
        )
        cameras.start()
        n_usb_frames = 0
        try:
            start_time = time.time()
            cameras.start_recording(episode_dir, start_time=start_time)
            print(f"Recording to: {episode_dir}")
            if duration > 0:
                print(f"Duration: {duration}s")
            else:
                print("Duration: until q is pressed in preview window.")
            print("Preview order: realsense_0 | realsense_1 | usb")
            print("Press q in the preview window to stop and save.")

            end_time = start_time + duration if duration > 0 else None
            rs_out = None
            while True:
                rs_out = cameras.realsense.get(out=rs_out)
                usb_data = cameras.pump_usb_frame()
                n_usb_frames += 1

                preview = make_preview(rs_out, usb_data)
                cv2.putText(
                    preview,
                    f"recording frames_usb={n_usb_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("arx5_three_camera_test", preview)
                key = cv2.pollKey()
                if key == ord("q"):
                    print("Stop requested by q.")
                    break
                if end_time is not None and time.time() >= end_time:
                    print("Stop requested by duration.")
                    break
                time.sleep(1 / max(fps, 1))
        except KeyboardInterrupt:
            print("Interrupted by Ctrl+C. Stopping recorders and flushing files...")
        finally:
            cameras.stop_recording()
            time.sleep(0.5)
            cameras.stop()
            cv2.destroyAllWindows()

    print("Done.")
    print(f"USB frames written: {n_usb_frames}")
    print("Expected files:")
    for idx in range(3):
        path = os.path.join(episode_dir, f"{idx}.mp4")
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"  {path} exists={exists} size={size}")


if __name__ == "__main__":
    main()
