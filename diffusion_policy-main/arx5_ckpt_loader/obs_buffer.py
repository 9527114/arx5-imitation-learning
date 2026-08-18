from collections import deque
from typing import Dict, Optional

import numpy as np
import torch

from diffusion_policy.real_world.real_inference_util import get_real_obs_dict


class Arx5ObsBuffer:
    """Builds the n_obs_steps observation dict expected by the trained policy."""

    def __init__(self, shape_meta, n_obs_steps: int):
        self.shape_meta = shape_meta
        self.n_obs_steps = int(n_obs_steps)
        self.frames = deque(maxlen=self.n_obs_steps)

    def clear(self):
        self.frames.clear()

    def append(self, camera_frames: Dict[str, np.ndarray], robot_state: Dict[str, np.ndarray]):
        eef_pose = np.asarray(robot_state["ActualTCPPose"], dtype=np.float32)
        gripper_pos = np.asarray(robot_state["gripper_pos"], dtype=np.float32)
        robot_joint = np.asarray(robot_state["ActualQ"], dtype=np.float32)
        sample = {
            "timestamp": float(robot_state["robot_receive_timestamp"]),
            "robot_eef_pose": eef_pose,
            "robot_gripper": gripper_pos,
            "robot_joint": robot_joint,
            "robot0_eef_pos": eef_pose[:3],
            "robot0_eef_rot_axis_angle": eef_pose[3:],
            "robot0_gripper_width": gripper_pos,
        }
        sample.update(camera_frames)
        self.frames.append(sample)

    @property
    def is_ready(self) -> bool:
        return len(self.frames) >= self.n_obs_steps

    def _stack_env_obs(self) -> Dict[str, np.ndarray]:
        if not self.is_ready:
            raise RuntimeError(
                f"Need {self.n_obs_steps} observations, got {len(self.frames)}."
            )
        result = {}
        keys = self.frames[-1].keys()
        for key in keys:
            result[key] = np.stack([frame[key] for frame in self.frames], axis=0)
        return result

    def as_policy_input(self, device: torch.device) -> Dict[str, torch.Tensor]:
        env_obs = self._stack_env_obs()
        obs_np = get_real_obs_dict(env_obs=env_obs, shape_meta=self.shape_meta)
        return {
            key: torch.from_numpy(value).unsqueeze(0).to(device)
            for key, value in obs_np.items()
        }

    def get_timestamps(self) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError(
                f"Need {self.n_obs_steps} observations, got {len(self.frames)}."
            )
        return np.asarray([frame["timestamp"] for frame in self.frames], dtype=np.float64)


def make_camera_frame_dict(
    realsense_data,
    usb_data: Optional[dict] = None,
    camera_order: str = "old_dp",
) -> Dict[str, np.ndarray]:
    """Map collector camera outputs to policy camera keys.

    old_dp:
      camera_0 is USB wrist, camera_1/camera_2 are RealSense views.
    current_collector:
      camera_0/camera_1 are RealSense views, camera_2 is USB. This matches
      checkpoints trained before the old-DP schema cleanup.
    """

    if camera_order not in ("old_dp", "current_collector"):
        raise ValueError(f"Unsupported camera_order: {camera_order}")
    frames = {}
    if camera_order == "old_dp":
        if usb_data is not None:
            frames["camera_0"] = usb_data["color"]
            rs_offset = 1
        else:
            rs_offset = 0
        for out_idx, idx in enumerate(sorted(realsense_data.keys())):
            frames[f"camera_{out_idx + rs_offset}"] = realsense_data[idx]["color"]
        return frames

    for out_idx, idx in enumerate(sorted(realsense_data.keys())):
        frames[f"camera_{out_idx}"] = realsense_data[idx]["color"]
    if usb_data is not None:
        frames[f"camera_{len(frames)}"] = usb_data["color"]
    return frames


def make_video_device_frame_dict(video_data) -> Dict[str, np.ndarray]:
    return {
        key: value["color"]
        for key, value in video_data.items()
    }


def build_zero_obs(cfg, device: torch.device) -> Dict[str, torch.Tensor]:
    n_obs_steps = int(cfg.n_obs_steps)
    obs_dict = {}
    for key, meta in cfg.task.shape_meta.obs.items():
        shape = tuple(meta.shape)
        tensor_shape = (1, n_obs_steps, *shape)
        obs_dict[key] = torch.zeros(tensor_shape, dtype=torch.float32, device=device)
    return obs_dict
