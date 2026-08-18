import threading
import time
from typing import Optional

import numpy as np

from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


def wall_to_monotonic(timestamp: float) -> float:
    return time.monotonic() - time.time() + float(timestamp)


class ContinuousWaypointExecutor:
    """Old-DP style ARX5 waypoint executor.

    Policy chunks are stored as timestamped waypoints. A control thread samples
    the resulting trajectory at a fixed rate and streams single EEF commands.
    This avoids repeatedly replacing SDK trajectories with one-point trajs.
    """

    def __init__(
        self,
        robot,
        frequency: float = 200.0,
        gripper_margin: float = 0.0,
        command_latency: float = 0.01,
        log_interval: float = 0.1,
        logger=None,
        tracking_guard: bool = True,
        max_pos_speed: float = 0.45,
        max_rot_speed: float = 1.05,
        replace_future: bool = False,
        replace_blend_time: float = 0.0,
        replace_min_lead_time: float = 0.0,
    ):
        self.robot = robot
        self.frequency = float(frequency)
        self.gripper_margin = float(gripper_margin)
        self.command_latency = float(command_latency)
        self.log_interval = float(log_interval)
        self.logger = logger
        self.tracking_guard = bool(tracking_guard)
        self.max_pos_speed = float(max_pos_speed)
        self.max_rot_speed = float(max_rot_speed)
        self.replace_future = bool(replace_future)
        self.replace_blend_time = float(replace_blend_time)
        self.replace_min_lead_time = float(replace_min_lead_time)

        self._lock = threading.Lock()
        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._pose_interp = None
        self._gripper_interp = None
        self._last_waypoint_time = None
        self._last_log_time = 0.0

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="arx5-continuous-executor",
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
            self._pose_interp = None
            self._gripper_interp = None
            self._last_waypoint_time = None

    def set_hold(self, action, timestamp: Optional[float] = None):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"Expected 7D hold action, got {action.shape}.")
        if timestamp is None:
            timestamp = time.time()
        t = wall_to_monotonic(timestamp)
        with self._lock:
            self._pose_interp = PoseTrajectoryInterpolator(
                times=np.asarray([t], dtype=np.float64),
                poses=action[None, :6],
            )
            self._gripper_interp = PoseTrajectoryInterpolator(
                times=np.asarray([t], dtype=np.float64),
                poses=np.asarray([[action[6], 0, 0, 0, 0, 0]], dtype=np.float64),
            )
            self._last_waypoint_time = t

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
        timestamps = timestamps[order]
        target_times = np.asarray([wall_to_monotonic(t) for t in timestamps], dtype=np.float64)

        with self._lock:
            curr_t = time.monotonic()
            if self._pose_interp is None:
                state = self.robot.get_state()
                current_action = np.concatenate(
                    [
                        np.asarray(state["ActualTCPPose"], dtype=np.float64),
                        np.asarray([state["gripper_pos"][0]], dtype=np.float64),
                    ]
                )
                hold_t = wall_to_monotonic(time.time())
                self._pose_interp = PoseTrajectoryInterpolator(
                    times=np.asarray([hold_t], dtype=np.float64),
                    poses=current_action[None, :6],
                )
                self._gripper_interp = PoseTrajectoryInterpolator(
                    times=np.asarray([hold_t], dtype=np.float64),
                    poses=np.asarray([[current_action[6], 0, 0, 0, 0, 0]], dtype=np.float64),
                )
                self._last_waypoint_time = hold_t

            if self.replace_future:
                min_step = 1.0 / max(self.frequency, 1e-6)
                min_replace_lead = max(min_step, self.replace_min_lead_time)
                adjusted_times = []
                last_t = curr_t
                for target_t in target_times:
                    pose_time = max(float(target_t), curr_t + min_replace_lead, last_t + 1e-5)
                    adjusted_times.append(pose_time)
                    last_t = pose_time
                adjusted_times = np.asarray(adjusted_times, dtype=np.float64)
                adjusted_actions = actions.copy()

                first_new_t = float(adjusted_times[0])
                old_end_t = float(self._pose_interp.times[-1])
                old_pose_at_first = self._pose_interp(first_new_t)
                old_gripper_at_first = float(self._gripper_interp(first_new_t)[0])
                boundary_pos_jump = float(np.linalg.norm(adjusted_actions[0, :3] - old_pose_at_first[:3]))
                boundary_rot_jump = float(np.linalg.norm(adjusted_actions[0, 3:6] - old_pose_at_first[3:6]))
                boundary_gripper_jump = float(abs(adjusted_actions[0, 6] - old_gripper_at_first))
                eps = 1e-6
                old_pose_prefix = self._pose_interp.trim(curr_t, first_new_t)
                old_gripper_prefix = self._gripper_interp.trim(curr_t, first_new_t)
                if self.replace_blend_time > 0:
                    blend_end_t = first_new_t + self.replace_blend_time
                    old_poses_at_new = self._pose_interp(adjusted_times)
                    old_gripper_at_new = self._gripper_interp(adjusted_times)[:, 0]
                    for idx, pose_time in enumerate(adjusted_times):
                        if pose_time > blend_end_t:
                            break
                        alpha = np.clip(
                            (float(pose_time) - first_new_t)
                            / max(self.replace_blend_time, 1e-6),
                            0.0,
                            1.0,
                        )
                        adjusted_actions[idx, :6] = (
                            (1.0 - alpha) * old_poses_at_new[idx]
                            + alpha * adjusted_actions[idx, :6]
                        )
                        adjusted_actions[idx, 6] = (
                            (1.0 - alpha) * old_gripper_at_new[idx]
                            + alpha * adjusted_actions[idx, 6]
                        )
                prefix_keep = old_pose_prefix.times < (first_new_t - eps)
                prefix_times = old_pose_prefix.times[prefix_keep]
                prefix_poses = old_pose_prefix.poses[prefix_keep]
                prefix_gripper_poses = old_gripper_prefix.poses[prefix_keep]
                if len(prefix_times) == 0:
                    prefix_times = np.asarray([curr_t], dtype=np.float64)
                    prefix_poses = self._pose_interp(curr_t)[None]
                    prefix_gripper_poses = self._gripper_interp(curr_t)[None]

                self._pose_interp = PoseTrajectoryInterpolator(
                    times=np.concatenate([prefix_times, adjusted_times]),
                    poses=np.concatenate([prefix_poses, adjusted_actions[:, :6]], axis=0),
                )
                self._gripper_interp = PoseTrajectoryInterpolator(
                    times=np.concatenate([prefix_times, adjusted_times]),
                    poses=np.concatenate(
                        [
                            prefix_gripper_poses,
                            np.column_stack(
                                [
                                    adjusted_actions[:, 6],
                                    np.zeros((len(actions), 5), dtype=np.float64),
                                ]
                            ),
                        ],
                        axis=0,
                    ),
                )
                self._last_waypoint_time = float(adjusted_times[-1])
                if self.logger is not None:
                    self.logger.log(
                        "continuous_replace",
                        timestamp=now,
                        inserted=len(actions),
                        first_new_lead=float(first_new_t - curr_t),
                        old_horizon=float(old_end_t - curr_t),
                        new_horizon=float(adjusted_times[-1] - curr_t),
                        replace_blend_time=self.replace_blend_time,
                        replace_min_lead_time=self.replace_min_lead_time,
                        boundary_pos_jump=boundary_pos_jump,
                        boundary_rot_jump=boundary_rot_jump,
                        boundary_gripper_jump=boundary_gripper_jump,
                    )
                return len(actions)

            for action, target_t in zip(actions, target_times):
                if self.replace_future:
                    pose_time = max(
                        float(target_t),
                        curr_t + 1.0 / max(self.frequency, 1e-6),
                    )
                else:
                    pose_time = max(
                        float(target_t),
                        curr_t + 1.0 / max(self.frequency, 1e-6),
                        float(self._last_waypoint_time or curr_t) + 1e-5,
                    )
                self._pose_interp = self._pose_interp.schedule_waypoint(
                    pose=action[:6],
                    time=pose_time,
                    max_pos_speed=self.max_pos_speed,
                    max_rot_speed=self.max_rot_speed,
                    curr_time=curr_t,
                    last_waypoint_time=self._last_waypoint_time,
                )
                self._gripper_interp = self._gripper_interp.schedule_waypoint(
                    pose=[action[6], 0, 0, 0, 0, 0],
                    time=pose_time,
                    curr_time=curr_t,
                    last_waypoint_time=self._last_waypoint_time,
                )
                self._last_waypoint_time = float(self._pose_interp.times[-1])
        return len(actions)

    def _sample(self, timestamp: float):
        with self._lock:
            if self._pose_interp is None or self._gripper_interp is None:
                return None
            pose = self._pose_interp(timestamp)
            gripper = float(self._gripper_interp(timestamp)[0])
        return np.concatenate([pose, [gripper]]).astype(np.float64)

    def sample_wall_time(self, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        return self._sample(wall_to_monotonic(float(timestamp)))

    def sample_future_window_wall(self, start_time: float, steps: int, dt: float):
        steps = int(steps)
        if steps <= 0:
            return None
        start_t = wall_to_monotonic(float(start_time))
        sample_times = start_t + np.arange(steps, dtype=np.float64) * float(dt)
        with self._lock:
            if self._pose_interp is None or self._gripper_interp is None:
                return None
            end_t = float(self._pose_interp.times[-1])
            valid = sample_times <= (end_t + 1e-6)
            poses = self._pose_interp(sample_times)
            gripper = self._gripper_interp(sample_times)[:, 0]
        actions = np.concatenate([poses, gripper[:, None]], axis=1).astype(np.float64)
        return actions, valid.astype(np.float32), float(end_t - start_t)

    def _run(self):
        dt = 1.0 / max(self.frequency, 1e-6)
        next_t = time.monotonic()
        while not self._stop.is_set():
            if self._enabled.is_set():
                mono_now = time.monotonic()
                wall_now = time.time()
                action = self._sample(mono_now)
                if action is not None:
                    if self.tracking_guard and not self.robot.tracking_is_healthy():
                        pos_error, rot_error, state = self.robot.tracking_error()
                        action[:6] = state["ActualTCPPose"]
                        action[6] = float(state["gripper_pos"][0])
                        self.set_hold(action, timestamp=wall_now)
                        if (
                            self.logger is not None
                            and wall_now - self._last_log_time >= self.log_interval
                        ):
                            self._last_log_time = wall_now
                            self.logger.log(
                                "continuous_tracking_hold",
                                timestamp=wall_now,
                                pos_error=pos_error,
                                rot_error=rot_error,
                                action=action,
                            )

                    if self.robot.robot_config is not None:
                        lo = self.gripper_margin
                        hi = float(self.robot.robot_config.gripper_width) - self.gripper_margin
                        action[6] = np.clip(action[6], lo, hi)

                    self.robot.send_eef_cmd(
                        action[:6],
                        target_time=wall_now + self.command_latency,
                        gripper_pos=float(action[6]),
                    )
                    if (
                        self.logger is not None
                        and wall_now - self._last_log_time >= self.log_interval
                    ):
                        self._last_log_time = wall_now
                        self.logger.log(
                            "continuous_sample",
                            timestamp=wall_now,
                            action=action,
                        )
            next_t += dt
            sleep_time = next_t - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.monotonic()
