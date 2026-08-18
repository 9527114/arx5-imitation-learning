import threading
import time
from typing import Optional, Tuple

import numpy as np

from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


class ActionTrajectoryBuffer:
    """Thread-safe 7D action trajectory buffer.

    Actions are absolute ARX5 targets:
      [x, y, z, rx, ry, rz, gripper_width]

    The buffer stores timestamped future actions and interpolates them into a
    smooth single target for the SDK execution loop.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._times = np.empty((0,), dtype=np.float64)
        self._actions = np.empty((0, 7), dtype=np.float64)

    def clear(self):
        with self._lock:
            self._times = np.empty((0,), dtype=np.float64)
            self._actions = np.empty((0, 7), dtype=np.float64)

    def set_hold(self, action, timestamp: Optional[float] = None):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"Expected 7D hold action, got {action.shape}.")
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            self._times = np.asarray([float(timestamp)], dtype=np.float64)
            self._actions = action[None].copy()

    def add_chunk(
        self,
        actions,
        timestamps,
        current_action,
        now: Optional[float] = None,
        min_lead_time: float = 0.01,
        blend_time: float = 0.0,
        keep_old_until_new: bool = True,
    ) -> int:
        """Insert a new future chunk.

        The old future trajectory is kept only until the first new timestamp.
        When blend_time > 0, the beginning of the new chunk is also blended
        against the old buffered trajectory. This preserves the old-DP style
        receding-horizon handoff without letting stale far-future commands
        dominate the latest policy output.
        """
        if now is None:
            now = time.time()
        actions = np.asarray(actions, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        current_action = np.asarray(current_action, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected actions shape (N, 7), got {actions.shape}.")
        if timestamps.shape != (len(actions),):
            raise ValueError(f"Expected timestamps shape {(len(actions),)}, got {timestamps.shape}.")
        if current_action.shape != (7,):
            raise ValueError(f"Expected current_action shape (7,), got {current_action.shape}.")

        keep = timestamps > (float(now) + float(min_lead_time))
        actions = actions[keep]
        timestamps = timestamps[keep]
        if len(actions) == 0:
            return 0

        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        actions = actions[order]
        unique = np.concatenate([[True], np.diff(timestamps) > 1e-5])
        timestamps = timestamps[unique]
        actions = actions[unique]
        if len(actions) == 0:
            return 0

        with self._lock:
            anchor_action = current_action.copy()
            anchor_time = float(now)
            if len(self._times) > 0 and self._times[0] <= now <= self._times[-1]:
                anchor_action = self._sample_locked(now)

            if keep_old_until_new and len(self._times) > 0:
                first_new_time = float(timestamps[0])
                old_keep = (self._times > now) & (self._times < first_new_time)
                old_times = self._times[old_keep]
                old_actions = self._actions[old_keep]
            else:
                old_times = np.empty((0,), dtype=np.float64)
                old_actions = np.empty((0, 7), dtype=np.float64)

            if blend_time > 0 and len(self._times) > 1:
                first_new_time = float(timestamps[0])
                blend_end_time = first_new_time + float(blend_time)
                blended_actions = actions.copy()
                for idx, timestamp in enumerate(timestamps):
                    if timestamp > blend_end_time:
                        break
                    old_action = self._sample_locked(float(timestamp))
                    alpha = np.clip(
                        (float(timestamp) - first_new_time) / max(float(blend_time), 1e-6),
                        0.0,
                        1.0,
                    )
                    blended_actions[idx] = (1.0 - alpha) * old_action + alpha * blended_actions[idx]
                actions = blended_actions

            all_times = np.concatenate([[anchor_time], old_times, timestamps])
            all_actions = np.concatenate([[anchor_action], old_actions, actions], axis=0)
            order = np.argsort(all_times)
            all_times = all_times[order]
            all_actions = all_actions[order]
            unique = np.concatenate([[True], np.diff(all_times) > 1e-5])
            self._times = all_times[unique]
            self._actions = all_actions[unique]
        return len(actions)

    def sample(self, timestamp: Optional[float] = None) -> Optional[np.ndarray]:
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            if len(self._times) == 0:
                return None
            return self._sample_locked(float(timestamp))

    def horizon(self, now: Optional[float] = None) -> Tuple[float, float]:
        if now is None:
            now = time.time()
        with self._lock:
            if len(self._times) == 0:
                return 0.0, 0.0
            return float(self._times[0] - now), float(self._times[-1] - now)

    def _sample_locked(self, timestamp: float) -> np.ndarray:
        if len(self._times) == 1:
            return self._actions[0].copy()
        pose_interp = PoseTrajectoryInterpolator(
            times=self._times,
            poses=self._actions[:, :6],
        )
        pose = pose_interp(timestamp)
        gripper = np.interp(
            np.clip(timestamp, self._times[0], self._times[-1]),
            self._times,
            self._actions[:, 6],
        )
        return np.concatenate([pose, [gripper]]).astype(np.float64)


class BufferedActionExecutor:
    """High-rate local executor for ARX5 policy chunks."""

    def __init__(
        self,
        robot,
        buffer: ActionTrajectoryBuffer,
        frequency: float,
        gripper_margin: float = 0.0,
        command_latency: float = 0.02,
        log_interval: float = 0.1,
        logger=None,
        tracking_guard: bool = True,
    ):
        self.robot = robot
        self.buffer = buffer
        self.frequency = float(frequency)
        self.gripper_margin = float(gripper_margin)
        self.command_latency = float(command_latency)
        self.log_interval = float(log_interval)
        self.logger = logger
        self.tracking_guard = bool(tracking_guard)
        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._last_log_time = 0.0

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="arx5-buffered-executor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._enabled.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def enable(self):
        self._enabled.set()

    def disable(self):
        self._enabled.clear()

    def _run(self):
        dt = 1.0 / max(self.frequency, 1e-6)
        next_t = time.monotonic()
        while not self._stop.is_set():
            if self._enabled.is_set():
                wall_now = time.time()
                action = self.buffer.sample(wall_now)
                if action is not None:
                    if self.tracking_guard and not self.robot.tracking_is_healthy():
                        pos_error, rot_error, state = self.robot.tracking_error()
                        action[:6] = state["ActualTCPPose"]
                        action[6] = float(state["gripper_pos"][0])
                        if (
                            self.logger is not None
                            and wall_now - self._last_log_time >= self.log_interval
                        ):
                            self._last_log_time = wall_now
                            self.logger.log(
                                "tracking_guard_hold",
                                timestamp=wall_now,
                                pos_error=pos_error,
                                rot_error=rot_error,
                                action=action,
                            )
                    if self.robot.robot_config is not None:
                        gripper_lo = self.gripper_margin
                        gripper_hi = (
                            float(self.robot.robot_config.gripper_width)
                            - self.gripper_margin
                        )
                        action[6] = np.clip(action[6], gripper_lo, gripper_hi)
                    self.robot.schedule_waypoint(
                        action[:6],
                        target_time=wall_now + self.command_latency,
                        gripper_pos=float(action[6]),
                    )
                    if (
                        self.logger is not None
                        and wall_now - self._last_log_time >= self.log_interval
                    ):
                        self._last_log_time = wall_now
                        start_dt, end_dt = self.buffer.horizon(wall_now)
                        self.logger.log(
                            "buffer_sample",
                            timestamp=wall_now,
                            action=action,
                            horizon=[start_dt, end_dt],
                        )
            next_t += dt
            sleep_time = next_t - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.monotonic()
