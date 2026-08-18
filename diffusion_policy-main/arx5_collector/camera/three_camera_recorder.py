import os
from typing import Optional, Sequence

from multiprocessing.managers import SharedMemoryManager

from arx5_collector.camera.realsense_recorder import RealsenseRecorder
from arx5_collector.camera.usb_recorder import UsbRecorder


class ThreeCameraRecorder:
    """Owns two RealSense cameras plus one USB camera.

    Recording uses the old ARX5 DP camera convention:
      0.mp4: USB wrist camera
      1.mp4: first RealSense
      2.mp4: second RealSense
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        realsense_config: Optional[str],
        usb_config: Optional[str] = None,
        usb_device: int = 0,
        realsense_serial_numbers: Optional[Sequence[str]] = None,
        resolution=(1280, 720),
        usb_resolution=None,
        fps: int = 30,
        realsense_video_crf: int = 23,
        verbose: bool = False,
    ):
        self.realsense = RealsenseRecorder(
            shm_manager=shm_manager,
            serial_numbers=realsense_serial_numbers,
            config_path=realsense_config,
            resolution=resolution,
            fps=fps,
            video_crf=realsense_video_crf,
            verbose=verbose,
        )
        self.usb = UsbRecorder(
            device=usb_device,
            config_path=usb_config,
            resolution=resolution if usb_resolution is None else usb_resolution,
            fps=fps,
        )

    @property
    def n_cameras(self):
        return self.realsense.n_cameras + 1

    @property
    def is_ready(self):
        return self.realsense.is_ready and self.usb.is_ready

    def start(self):
        self.realsense.start(wait=True)
        self.usb.start()

    def stop(self):
        self.usb.stop()
        self.realsense.stop(wait=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start_recording(self, episode_video_dir: str, start_time: float, record_usb: bool = True):
        os.makedirs(episode_video_dir, exist_ok=True)
        rs_offset = 1 if record_usb else 0
        rs_paths = [
            os.path.join(episode_video_dir, f"{idx + rs_offset}.mp4")
            for idx in range(self.realsense.n_cameras)
        ]
        self.realsense.restart_put(start_time)
        self.realsense.start_recording(rs_paths, start_time=start_time)
        if record_usb:
            self.usb.start_recording(
                os.path.join(episode_video_dir, "0.mp4"),
                start_time=start_time,
            )

    def stop_recording(self):
        self.usb.stop_recording()
        self.realsense.stop_recording()

    def pump_usb_frame(self):
        data = self.usb.get()
        self.usb.write_frame(data["color"], frame_time=data["timestamp"])
        return data
