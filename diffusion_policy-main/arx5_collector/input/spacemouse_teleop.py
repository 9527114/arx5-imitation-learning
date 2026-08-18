from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from queue import Queue
from typing import Optional

import numpy as np

from arx5_collector.sdk_path import ensure_arx5_sdk_path
ensure_arx5_sdk_path()

from arx5_local_config import apply_local_spacemouse_mapping
from peripherals.spacemouse_shared_memory import Spacemouse


@dataclass
class TeleopCommand:
    action: np.ndarray
    gripper_action: float
    stage: int
    raw_motion: np.ndarray
    reset_requested: bool = False
    left_pressed: bool = False
    right_pressed: bool = False


class SpaceMouseTeleop:
    """Converts SpaceMouse motion into target EEF pose actions."""

    def __init__(
        self,
        shm_manager: Optional[SharedMemoryManager] = None,
        pos_speed: float = 0.8,
        rot_speed: float = 1.5,
        gripper_speed: float = 0.08,
        gripper_margin: float = 0.06,
        deadzone: float = 0.1,
        max_value: int = 500,
        smoothing_window: int = 3,
    ):
        self.shm_manager = shm_manager
        self._own_shm_manager = shm_manager is None
        self.pos_speed = pos_speed
        self.rot_speed = rot_speed
        self.gripper_speed = gripper_speed
        self.gripper_margin = gripper_margin
        self.deadzone = deadzone
        self.max_value = max_value
        self.queue = Queue(smoothing_window)
        self.spacemouse = None

    def start(self):
        if self.shm_manager is None:
            self.shm_manager = SharedMemoryManager()
            self.shm_manager.start()
        self.spacemouse = Spacemouse(
            shm_manager=self.shm_manager,
            deadzone=self.deadzone,
            max_value=self.max_value,
        )
        self.spacemouse.start()

    def stop(self):
        if self.spacemouse is not None:
            self.spacemouse.stop()
        self.spacemouse = None
        if self._own_shm_manager and self.shm_manager is not None:
            self.shm_manager.shutdown()
        self.shm_manager = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_motion(self) -> np.ndarray:
        assert self.spacemouse is not None
        state = self.spacemouse.get_motion_state_transformed()
        state = apply_local_spacemouse_mapping(state)
        positive_idx = state >= self.deadzone
        negative_idx = state <= -self.deadzone
        state[positive_idx] = (state[positive_idx] - self.deadzone) / (
            1 - self.deadzone
        )
        state[negative_idx] = (state[negative_idx] + self.deadzone) / (
            1 - self.deadzone
        )
        if self.queue.maxsize <= 0:
            return state
        if self.queue._qsize() == self.queue.maxsize:
            self.queue._get()
        self.queue.put_nowait(state)
        return np.mean(np.array(list(self.queue.queue)), axis=0)

    def update(
        self,
        target_pose: np.ndarray,
        target_gripper_pos: float,
        dt: float,
        gripper_width: float,
        stage: int = 0,
    ) -> TeleopCommand:
        assert self.spacemouse is not None
        motion = self.get_motion()

        action = np.asarray(target_pose, dtype=np.float64).copy()
        action[:3] += motion[:3] * self.pos_speed * dt
        action[3:] += motion[3:] * self.rot_speed * dt

        gripper_delta = 0
        left_pressed = self.spacemouse.is_button_pressed(0)
        right_pressed = self.spacemouse.is_button_pressed(1)
        reset_requested = left_pressed and right_pressed
        if left_pressed and not right_pressed:
            gripper_delta = -1
        elif right_pressed and not left_pressed:
            gripper_delta = 1
        gripper_next = target_gripper_pos + gripper_delta * self.gripper_speed * dt
        if gripper_delta > 0:
            gripper_next = min(gripper_next, gripper_width - self.gripper_margin)
        elif gripper_delta < 0:
            gripper_next = max(gripper_next, 0.0)
        gripper_action = float(np.clip(gripper_next, 0.0, gripper_width))

        return TeleopCommand(
            action=action,
            gripper_action=gripper_action,
            stage=stage,
            raw_motion=motion.copy(),
            reset_requested=reset_requested,
            left_pressed=left_pressed,
            right_pressed=right_pressed,
        )
