import threading
import time
from typing import Optional

import numpy as np


def wall_to_monotonic(timestamp: float) -> float:
    return time.monotonic() - time.time() + float(timestamp)


class _LinearActionInterpolator:
    def __init__(self, times, actions):
        times = np.asarray(times, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        if times.ndim != 1 or actions.ndim != 2 or len(times) != len(actions):
            raise ValueError("Invalid linear interpolator inputs.")
        if actions.shape[1] != 7:
            raise ValueError(f"Expected 7D actions, got {actions.shape}.")
        order = np.argsort(times)
        self.times = times[order]
        self.actions = actions[order]

    def __call__(self, t):
        is_scalar = np.isscalar(t)
        t_arr = np.asarray([t] if is_scalar else t, dtype=np.float64)
        t_arr = np.clip(t_arr, self.times[0], self.times[-1])
        out = np.empty((len(t_arr), self.actions.shape[1]), dtype=np.float64)
        for dim in range(self.actions.shape[1]):
            out[:, dim] = np.interp(t_arr, self.times, self.actions[:, dim])
        return out[0] if is_scalar else out

    def trim(self, start_t: float, end_t: float):
        start_t = float(start_t)
        end_t = float(end_t)
        if end_t < start_t:
            end_t = start_t
        keep = (self.times > start_t) & (self.times < end_t)
        times = np.concatenate([[start_t], self.times[keep], [end_t]])
        times = np.unique(times)
        actions = self(times)
        return _LinearActionInterpolator(times, actions)


class JointContinuousWaypointExecutor:
    """Joint-space continuous executor for DP-Joint.

    It mirrors the EEF continuous executor but linearly interpolates absolute
    joint actions [q1..q6, gripper] and streams set_joint_cmd at a fixed rate.
    This avoids repeatedly replacing SDK joint trajectories.
    """

    def __init__(
        self,
        robot,
        frequency: float = 200.0,
        gripper_margin: float = 0.0,
        command_latency: float = 0.01,
        replace_blend_time: float = 0.08,
        replace_min_lead_time: float = 0.06,
        replace_future: bool = False,
        logger=None,
    ):
        self.robot = robot
        self.frequency = float(frequency)
        self.gripper_margin = float(gripper_margin)
        self.command_latency = float(command_latency)
        self.replace_blend_time = float(replace_blend_time)
        self.replace_min_lead_time = float(replace_min_lead_time)
        self.replace_future = bool(replace_future)
        self.logger = logger

        self._lock = threading.Lock()
        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._interp = None
        self._last_waypoint_time = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="arx5-joint-continuous-executor",
            daemon=True,
        )
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

    def clear(self):
        with self._lock:
            self._interp = None
            self._last_waypoint_time = None

    def _current_action(self):
        state = self.robot.get_state()
        return np.concatenate(
            [
                np.asarray(state["ActualQ"], dtype=np.float64)[:6],
                np.asarray([float(state["gripper_pos"][0])], dtype=np.float64),
            ]
        )

    def set_hold(self, action=None, timestamp: Optional[float] = None):
        if action is None:
            action = self._current_action()
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"Expected 7D hold action, got {action.shape}.")
        if timestamp is None:
            timestamp = time.time()
        mono_t = wall_to_monotonic(timestamp)
        with self._lock:
            self._interp = _LinearActionInterpolator([mono_t], action[None])
            self._last_waypoint_time = mono_t

    def add_chunk(self, actions, timestamps, now: Optional[float] = None) -> int:
        if now is None:
            now = time.time()
        actions = np.asarray(actions, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected actions shape (N, 7), got {actions.shape}.")
        if timestamps.shape != (len(actions),):
            raise ValueError(f"Expected timestamps shape {(len(actions),)}, got {timestamps.shape}.")

        keep = timestamps > now
        actions = actions[keep]
        timestamps = timestamps[keep]
        if len(actions) == 0:
            return 0

        order = np.argsort(timestamps)
        actions = actions[order]
        target_times = np.asarray([wall_to_monotonic(timestamps[i]) for i in order], dtype=np.float64)

        with self._lock:
            curr_t = time.monotonic()
            if self._interp is None:
                hold_t = wall_to_monotonic(time.time())
                self._interp = _LinearActionInterpolator([hold_t], self._current_action()[None])
                self._last_waypoint_time = hold_t

            min_step = 1.0 / max(self.frequency, 1e-6)
            min_replace_lead = max(min_step, self.replace_min_lead_time)
            adjusted_times = []
            if self.replace_future:
                last_t = curr_t
            else:
                last_t = max(curr_t, float(self._last_waypoint_time or curr_t))
            for target_t in target_times:
                if self.replace_future:
                    action_t = max(float(target_t), curr_t + min_replace_lead, last_t + 1e-5)
                else:
                    action_t = max(float(target_t), curr_t + min_step, last_t + 1e-5)
                adjusted_times.append(action_t)
                last_t = action_t
            adjusted_times = np.asarray(adjusted_times, dtype=np.float64)
            adjusted_actions = actions.copy()

            first_new_t = float(adjusted_times[0])
            old_end_t = float(self._interp.times[-1])
            old_at_new = self._interp(adjusted_times)
            boundary_jump = float(np.linalg.norm(adjusted_actions[0, :6] - old_at_new[0, :6]))
            boundary_gripper_jump = float(abs(adjusted_actions[0, 6] - old_at_new[0, 6]))

            if self.replace_future and self.replace_blend_time > 0:
                blend_end_t = first_new_t + self.replace_blend_time
                for idx, action_t in enumerate(adjusted_times):
                    if action_t > blend_end_t:
                        break
                    alpha = np.clip(
                        (float(action_t) - first_new_t) / max(self.replace_blend_time, 1e-6),
                        0.0,
                        1.0,
                    )
                    adjusted_actions[idx] = (1.0 - alpha) * old_at_new[idx] + alpha * adjusted_actions[idx]

            old_prefix_end = first_new_t if self.replace_future else max(old_end_t, curr_t)
            old_prefix = self._interp.trim(curr_t, old_prefix_end)
            prefix_keep = old_prefix.times < (first_new_t - 1e-6)
            if not self.replace_future:
                prefix_keep = old_prefix.times < (adjusted_times[0] - 1e-6)
            prefix_times = old_prefix.times[prefix_keep]
            prefix_actions = old_prefix.actions[prefix_keep]
            if len(prefix_times) == 0:
                prefix_times = np.asarray([curr_t], dtype=np.float64)
                prefix_actions = self._interp(curr_t)[None]

            self._interp = _LinearActionInterpolator(
                np.concatenate([prefix_times, adjusted_times]),
                np.concatenate([prefix_actions, adjusted_actions], axis=0),
            )
            self._last_waypoint_time = float(adjusted_times[-1])
            if self.logger is not None:
                self.logger.log(
                    "joint_continuous_replace" if self.replace_future else "joint_continuous_append",
                    timestamp=now,
                    inserted=len(actions),
                    first_new_lead=float(first_new_t - curr_t),
                    old_horizon=float(old_end_t - curr_t),
                    new_horizon=float(adjusted_times[-1] - curr_t),
                    boundary_joint_jump=boundary_jump,
                    boundary_gripper_jump=boundary_gripper_jump,
                    replace_blend_time=self.replace_blend_time,
                    replace_min_lead_time=self.replace_min_lead_time,
                    replace_future=self.replace_future,
                )
        return len(actions)

    def _sample(self, timestamp: float):
        with self._lock:
            if self._interp is None:
                return None
            return self._interp(timestamp).astype(np.float64)

    def _run(self):
        dt = 1.0 / max(self.frequency, 1e-6)
        next_t = time.monotonic()
        while not self._stop.is_set():
            if self._enabled.is_set():
                mono_now = time.monotonic()
                wall_now = time.time()
                action = self._sample(mono_now)
                if action is not None:
                    if self.robot.robot_config is not None:
                        lo = self.gripper_margin
                        hi = float(self.robot.robot_config.gripper_width) - self.gripper_margin
                        action[6] = np.clip(action[6], lo, hi)
                    self.robot.send_joint_cmd(
                        action[:6],
                        target_time=wall_now + self.command_latency,
                        gripper_pos=float(action[6]),
                    )
            next_t += dt
            sleep_time = next_t - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.monotonic()
