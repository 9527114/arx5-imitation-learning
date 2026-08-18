import time

import numpy as np
import torch


def policy_supports_prev_action(policy) -> bool:
    return hasattr(policy, "prev_cond_steps") and hasattr(policy, "action_dim")


class PrevActionConditioner:
    """Build latency-aware previous-chunk conditioning for CFG DP policies."""

    def __init__(
        self,
        prev_cond_steps,
        action_dim,
        latency=0.15,
        latency_margin=0.03,
        latency_ema_alpha=0.8,
        max_latency=0.25,
        max_start_idx=None,
        enabled=True,
    ):
        self.prev_cond_steps = int(prev_cond_steps)
        self.action_dim = int(action_dim)
        self.latency = float(latency)
        self.latency_margin = float(latency_margin)
        self.latency_ema_alpha = float(latency_ema_alpha)
        self.max_latency = float(max_latency)
        self.max_start_idx = None if max_start_idx is None else int(max_start_idx)
        self.enabled = bool(enabled) and self.prev_cond_steps > 0
        self.last_chunk = None
        self.last_timestamps = None

    def clear(self):
        self.last_chunk = None
        self.last_timestamps = None

    def record(self, action_chunk, action_timestamps):
        if not self.enabled:
            return
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        action_timestamps = np.asarray(action_timestamps, dtype=np.float64)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != self.action_dim:
            return
        if len(action_chunk) != len(action_timestamps) or len(action_chunk) == 0:
            return
        self.last_chunk = action_chunk.copy()
        self.last_timestamps = action_timestamps.copy()

    def update_latency(self, actual_latency):
        actual_latency = float(actual_latency)
        if actual_latency <= 0 or actual_latency > self.max_latency:
            return
        alpha = min(max(self.latency_ema_alpha, 0.0), 1.0)
        self.latency = alpha * self.latency + (1.0 - alpha) * actual_latency
        self.latency = min(self.latency, self.max_latency)

    def make_tensors(self, now=None, device="cpu", dtype=torch.float32):
        prev_action = np.zeros(
            (self.prev_cond_steps, self.action_dim),
            dtype=np.float32,
        )
        prev_mask = np.zeros((self.prev_cond_steps,), dtype=np.float32)
        start_idx = None

        if self.enabled and self.last_chunk is not None and self.last_timestamps is not None:
            if now is None:
                now = time.time()
            condition_time = float(now) + self.latency + self.latency_margin
            start_idx = int(np.searchsorted(self.last_timestamps, condition_time, side="left"))
            if self.max_start_idx is not None and start_idx > self.max_start_idx:
                raw_start_idx = start_idx
                start_idx = None
                return {
                    "prev_action": torch.as_tensor(prev_action, device=device, dtype=dtype).unsqueeze(0),
                    "prev_action_mask": torch.as_tensor(prev_mask, device=device, dtype=dtype).unsqueeze(0),
                    "debug": {
                        "valid": 0,
                        "start_idx": None,
                        "raw_start_idx": raw_start_idx,
                        "latency": self.latency,
                        "condition_time": condition_time,
                        "first_valid_action": None,
                        "disabled_reason": "start_idx_over_limit",
                    },
                }
            end_idx = min(start_idx + self.prev_cond_steps, len(self.last_chunk))
            if 0 <= start_idx < end_idx:
                window = self.last_chunk[start_idx:end_idx]
                n_valid = len(window)
                prev_action[:n_valid] = window
                prev_mask[:n_valid] = 1.0

        return {
            "prev_action": torch.as_tensor(prev_action, device=device, dtype=dtype).unsqueeze(0),
            "prev_action_mask": torch.as_tensor(prev_mask, device=device, dtype=dtype).unsqueeze(0),
            "debug": {
                "valid": int(prev_mask.sum()),
                "start_idx": start_idx,
                "raw_start_idx": start_idx,
                "latency": self.latency,
                "condition_time": None if start_idx is None else condition_time,
                "first_valid_action": _first_valid_action(prev_action, prev_mask),
                "disabled_reason": None if int(prev_mask.sum()) > 0 else "no_valid_future_chunk",
            },
        }


def _first_valid_action(prev_action, prev_mask):
    valid = np.nonzero(np.asarray(prev_mask) > 0.5)[0]
    if len(valid) == 0:
        return None
    return np.asarray(prev_action[valid[0]], dtype=np.float64)


def axis_angle_to_matrix(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    k = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def pose_tracking_error(actual_pose, desired_pose):
    actual_pose = np.asarray(actual_pose, dtype=np.float64)
    desired_pose = np.asarray(desired_pose, dtype=np.float64)
    pos_error = float(np.linalg.norm(desired_pose[:3] - actual_pose[:3]))
    desired_rot = axis_angle_to_matrix(desired_pose[3:6])
    actual_rot = axis_angle_to_matrix(actual_pose[3:6])
    delta_rot = desired_rot.T @ actual_rot
    cos_theta = (float(np.trace(delta_rot)) - 1.0) / 2.0
    cos_theta = min(1.0, max(-1.0, cos_theta))
    rot_error = float(np.arccos(cos_theta))
    return pos_error, rot_error


def error_adaptive_guidance_weight(
    base_weight,
    actual_pose,
    condition_action,
    pos_threshold=0.02,
    rot_threshold=0.05,
):
    if condition_action is None:
        return 0.0, None, None, False
    pos_error, rot_error = pose_tracking_error(actual_pose, condition_action[:6])
    guided = pos_error <= float(pos_threshold) and rot_error <= float(rot_threshold)
    return (float(base_weight) if guided else 0.0), pos_error, rot_error, guided
