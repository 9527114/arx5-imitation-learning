from dataclasses import dataclass

import numpy as np


@dataclass
class Arx5PolicyAction:
    pose: np.ndarray
    gripper: float
    raw: np.ndarray


def select_action(action_sequence, index: int = 0) -> Arx5PolicyAction:
    action = np.asarray(action_sequence, dtype=np.float64)
    if action.ndim == 2:
        action = action[index]
    if action.shape[-1] != 7:
        raise ValueError(f"Expected 7D action, got shape {action.shape}.")

    pose = action[:6].copy()
    gripper = float(action[6])
    return Arx5PolicyAction(pose=pose, gripper=gripper, raw=action.copy())


def clamp_gripper(gripper: float, width: float, margin: float = 0.0) -> float:
    lo = float(margin)
    hi = float(width) - float(margin)
    return float(np.clip(gripper, lo, hi))

