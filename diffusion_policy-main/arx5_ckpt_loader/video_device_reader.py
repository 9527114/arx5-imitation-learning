import time
from typing import Dict, Iterable, Optional

import cv2

from diffusion_policy.real_world.camera_config_util import load_and_apply_usb_camera_config


class VideoDeviceReader:
    """Reads camera_0/camera_1/camera_2 from explicit /dev/video indices."""

    def __init__(
        self,
        devices: Iterable[int],
        resolution=(640, 480),
        fps: int = 30,
        config_path: Optional[str] = None,
    ):
        self.devices = [int(x) for x in devices]
        self.resolution = tuple(resolution)
        self.fps = int(fps)
        self.config_path = config_path
        self.caps = []

    def start(self):
        self.stop()
        for device in self.devices:
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open /dev/video{device}")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if self.config_path:
                load_and_apply_usb_camera_config(cap, self.config_path)
            self.caps.append(cap)

    def stop(self):
        for cap in self.caps:
            cap.release()
        self.caps = []

    def get(self) -> Dict[str, dict]:
        frames = {}
        now = time.time()
        for idx, cap in enumerate(self.caps):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read /dev/video{self.devices[idx]}")
            frames[f"camera_{idx}"] = {
                "color": frame,
                "timestamp": now,
            }
        return frames

