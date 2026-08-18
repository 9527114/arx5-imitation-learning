from typing import Optional, Sequence

from multiprocessing.managers import SharedMemoryManager

from diffusion_policy.real_world.camera_config_util import load_and_apply_realsense_config
from diffusion_policy.real_world.multi_realsense import MultiRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder


class RealsenseRecorder:
    """Small ownership wrapper around DP's MultiRealsense."""

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        serial_numbers: Optional[Sequence[str]] = None,
        config_path: Optional[str] = None,
        resolution=(1280, 720),
        fps: int = 30,
        transform=None,
        video_recorder=None,
        video_crf: int = 23,
        verbose: bool = False,
    ):
        self.config_path = config_path
        if video_recorder is None:
            video_recorder = VideoRecorder.create_h264(
                fps=fps,
                codec="h264",
                input_pix_fmt="bgr24",
                crf=video_crf,
                thread_type="FRAME",
                thread_count=1,
            )
        self.realsense = MultiRealsense(
            serial_numbers=serial_numbers,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=fps,
            put_fps=fps,
            put_downsample=False,
            record_fps=fps,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            transform=transform,
            recording_transform=None,
            video_recorder=video_recorder,
            verbose=verbose,
        )

    @property
    def n_cameras(self):
        return self.realsense.n_cameras

    @property
    def is_ready(self):
        return self.realsense.is_ready

    def start(self, wait=True):
        self.realsense.start(wait=wait)
        load_and_apply_realsense_config(self.realsense, self.config_path)

    def stop(self, wait=True):
        self.realsense.stop(wait=wait)

    def get(self, *args, **kwargs):
        return self.realsense.get(*args, **kwargs)

    def start_recording(self, video_paths, start_time):
        self.realsense.start_recording(video_paths, start_time=start_time)

    def stop_recording(self):
        self.realsense.stop_recording()

    def restart_put(self, start_time):
        self.realsense.restart_put(start_time)
