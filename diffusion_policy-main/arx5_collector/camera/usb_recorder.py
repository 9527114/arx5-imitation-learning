import time
from typing import Optional

import cv2

from diffusion_policy.real_world.camera_config_util import load_and_apply_usb_camera_config
from diffusion_policy.real_world.video_recorder import VideoRecorder


class UsbRecorder:
    """Minimal USB camera recorder wrapper. Uses OpenCV capture + PyAV H264 writing."""

    def __init__(
        self,
        device: int = 0,
        config_path: Optional[str] = None,
        resolution=(1280, 720),
        fps: int = 30,
        auto_config: bool = True,
    ):
        self.device = device
        self.config_path = config_path
        self.resolution = tuple(resolution)
        self.fps = fps
        self.auto_config = auto_config
        self.cap = None
        self.video_recorder = VideoRecorder.create_h264(
            fps=fps,
            codec="h264",
            input_pix_fmt="bgr24",
            crf=23,
            thread_type="AUTO",
            thread_count=2,
        )
        self.video_path = None
        self.recording = False

    @property
    def is_ready(self):
        return self.cap is not None and self.cap.isOpened()

    def start(self):
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open /dev/video{self.device}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.config_path is not None:
            load_and_apply_usb_camera_config(self.cap, self.config_path)
        elif self.auto_config:
            from diffusion_policy.real_world.camera_config_util import apply_usb_camera_config
            apply_usb_camera_config(self.cap, {"mode": "auto"})

    def stop(self):
        self.stop_recording()
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get(self):
        assert self.cap is not None
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read USB camera frame.")
        return {
            "color": frame,
            "timestamp": time.time(),
        }

    def start_recording(self, video_path: str, start_time=None):
        assert self.cap is not None
        self.video_recorder.start(video_path, start_time=start_time)
        self.video_path = video_path
        self.recording = True

    def write_frame(self, frame, frame_time=None):
        if self.recording and self.video_recorder.is_ready():
            if frame_time is None:
                frame_time = time.time()
            self.video_recorder.write_frame(frame.copy(), frame_time=frame_time)

    def stop_recording(self):
        self.recording = False
        self.video_recorder.stop()
