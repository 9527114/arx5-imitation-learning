import time
from typing import Dict, Optional

import numpy as np

from arx5_collector.sdk_path import ensure_arx5_sdk_path
ensure_arx5_sdk_path()

import arx5_interface as arx5
from arx5_local_config import apply_local_controller_gain, apply_local_robot_config


class Arx5Robot:
    """Thin ARX5 SDK wrapper that exposes DP-style robot state keys."""

    def __init__(
        self,
        model: str,
        interface: str,
        log_level=None,
        reset_to_home: bool = False,
        preview_time: float = 0.05,
        command_mode: str = "cmd",
        arm_gain_mode: str = "default",
        arm_kp_scale: float = 1.0,
        arm_kd_scale: float = 1.0,
        gripper_safe_torque: float = 0.75,
        gripper_safe_margin: float = 0.002,
        tracking_pos_error_limit: float = 0.08,
        tracking_rot_error_limit: float = 0.6,
    ):
        self.model = model
        self.interface = interface
        self.controller: Optional[arx5.Arx5CartesianController] = None
        self.robot_config = None
        self.controller_config = None
        self.log_level = log_level
        self.reset_to_home_on_start = reset_to_home
        self.preview_time = preview_time
        self.arm_gain_mode = arm_gain_mode
        self.arm_kp_scale = float(arm_kp_scale)
        self.arm_kd_scale = float(arm_kd_scale)
        self.gripper_safe_torque = None if gripper_safe_torque is None else float(gripper_safe_torque)
        self.gripper_safe_margin = float(gripper_safe_margin)
        self.tracking_pos_error_limit = float(tracking_pos_error_limit)
        self.tracking_rot_error_limit = float(tracking_rot_error_limit)
        self._last_gripper_safety_print = 0.0
        self._last_tracking_guard_print = 0.0
        self.start_gain = None
        if command_mode not in ("cmd", "traj"):
            raise ValueError(f"Unsupported command_mode: {command_mode}")
        self.command_mode = command_mode
        self._last_state = None
        self._last_target_pose = None
        self._last_target_time = None

    @property
    def is_ready(self) -> bool:
        return self.controller is not None

    def start(self):
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(self.model)
        apply_local_robot_config(robot_config)
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "cartesian_controller", robot_config.joint_dof
        )
        controller = arx5.Arx5CartesianController(
            robot_config, controller_config, self.interface
        )
        self.start_gain = apply_local_controller_gain(
            controller,
            arm_gain_mode=self.arm_gain_mode,
            arm_kp_scale=self.arm_kp_scale,
            arm_kd_scale=self.arm_kd_scale,
        )
        if self.log_level is not None:
            controller.set_log_level(self.log_level)
        if self.reset_to_home_on_start:
            controller.reset_to_home()

        self.robot_config = robot_config
        self.controller_config = controller_config
        self.controller = controller
        state = self.get_state()
        self._last_target_pose = state["TargetTCPPose"].copy()
        self._last_target_time = state["robot_receive_timestamp"]

    def tracking_error(self):
        state = self.get_state()
        error = state["TargetTCPPose"] - state["ActualTCPPose"]
        return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:6])), state

    def tracking_is_healthy(self):
        pos_error, rot_error, _ = self.tracking_error()
        pos_ok = self.tracking_pos_error_limit <= 0 or pos_error <= self.tracking_pos_error_limit
        rot_ok = self.tracking_rot_error_limit <= 0 or rot_error <= self.tracking_rot_error_limit
        return pos_ok and rot_ok

    def get_gain_summary(self):
        assert self.controller is not None
        gain = self.controller.get_gain()
        return {
            "kp": np.asarray(gain.kp(), dtype=np.float64).copy(),
            "kd": np.asarray(gain.kd(), dtype=np.float64).copy(),
            "gripper_kp": float(gain.gripper_kp),
            "gripper_kd": float(gain.gripper_kd),
        }

    def apply_gain(
        self,
        arm_gain_mode: Optional[str] = None,
        arm_kp_scale: Optional[float] = None,
        arm_kd_scale: Optional[float] = None,
    ):
        assert self.controller is not None
        return apply_local_controller_gain(
            self.controller,
            arm_gain_mode=self.arm_gain_mode if arm_gain_mode is None else arm_gain_mode,
            arm_kp_scale=self.arm_kp_scale if arm_kp_scale is None else arm_kp_scale,
            arm_kd_scale=self.arm_kd_scale if arm_kd_scale is None else arm_kd_scale,
        )

    def restore_runtime_gain(self):
        return self.apply_gain(
            arm_gain_mode=self.arm_gain_mode,
            arm_kp_scale=self.arm_kp_scale,
            arm_kd_scale=self.arm_kd_scale,
        )

    def stop(self):
        if self.controller is not None:
            self.controller.set_to_damping()
        self.controller = None

    def reset_to_home(
        self,
        reset_gain_mode: Optional[str] = None,
        reset_arm_kp_scale: Optional[float] = None,
        reset_arm_kd_scale: Optional[float] = None,
        restore_gain: bool = False,
    ):
        assert self.controller is not None
        if reset_gain_mode is not None:
            self.apply_gain(
                arm_gain_mode=reset_gain_mode,
                arm_kp_scale=reset_arm_kp_scale,
                arm_kd_scale=reset_arm_kd_scale,
            )
        try:
            self.controller.reset_to_home()
            state = self.get_state()
            self._last_target_pose = state["TargetTCPPose"].copy()
            self._last_target_time = state["robot_receive_timestamp"]
            return state
        finally:
            if restore_gain:
                self.restore_runtime_gain()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_state(self) -> Dict[str, np.ndarray]:
        assert self.controller is not None
        eef_state = self.controller.get_eef_state()
        eef_cmd = self.controller.get_eef_cmd()
        joint_state = self.controller.get_joint_state()
        joint_cmd = self.controller.get_joint_cmd()
        now = time.time()

        actual_pose = np.asarray(eef_state.pose_6d(), dtype=np.float64).copy()
        target_pose = np.asarray(eef_cmd.pose_6d(), dtype=np.float64).copy()
        actual_joint = np.asarray(joint_state.pos(), dtype=np.float64).copy()
        actual_joint_vel = np.asarray(joint_state.vel(), dtype=np.float64).copy()
        target_joint = np.asarray(joint_cmd.pos(), dtype=np.float64).copy()
        target_joint_vel = np.asarray(joint_cmd.vel(), dtype=np.float64).copy()

        actual_pose_vel = np.zeros(6, dtype=np.float64)
        target_pose_vel = np.zeros(6, dtype=np.float64)
        if self._last_state is not None:
            dt = now - float(self._last_state["robot_receive_timestamp"])
            if dt > 1e-6:
                actual_pose_vel = (actual_pose - self._last_state["ActualTCPPose"]) / dt
                target_pose_vel = (target_pose - self._last_state["TargetTCPPose"]) / dt

        state = {
            "ActualTCPPose": actual_pose,
            "ActualTCPSpeed": actual_pose_vel,
            "ActualQ": actual_joint,
            "ActualQd": actual_joint_vel,
            "TargetTCPPose": target_pose,
            "TargetTCPSpeed": target_pose_vel,
            "TargetQ": target_joint,
            "TargetQd": target_joint_vel,
            "robot_receive_timestamp": now,
            "gripper_pos": np.array([eef_state.gripper_pos], dtype=np.float64),
            "target_gripper_pos": np.array([eef_cmd.gripper_pos], dtype=np.float64),
            "gripper_torque": np.array([joint_state.gripper_torque], dtype=np.float64),
        }
        self._last_state = state
        return state

    def get_all_state(self) -> Dict[str, np.ndarray]:
        state = self.get_state()
        return {
            key: value[None] if isinstance(value, np.ndarray) else np.array([value])
            for key, value in state.items()
        }

    def _safe_gripper_target(self, requested_pos: float) -> float:
        """Clamp closing commands when gripper torque is already high.

        ARX5 reports gripper width in meters. Smaller width means closing.
        When the fingers are blocked by an object, continuing to command a
        smaller width can trip the SDK over-current protection. In that case
        hold near the current actual width and still allow opening commands.
        """
        assert self.controller is not None
        width = float(self.robot_config.gripper_width)
        requested_pos = float(np.clip(requested_pos, 0.0, width))
        if self.gripper_safe_torque is None or self.gripper_safe_torque <= 0:
            return requested_pos

        eef_state = self.controller.get_eef_state()
        joint_state = self.controller.get_joint_state()
        actual_pos = float(np.clip(eef_state.gripper_pos, 0.0, width))
        torque = float(joint_state.gripper_torque)

        is_closing = requested_pos < actual_pos - 1e-4
        if is_closing and abs(torque) >= self.gripper_safe_torque:
            safe_pos = float(np.clip(actual_pos + self.gripper_safe_margin, 0.0, width))
            now = time.time()
            if now - self._last_gripper_safety_print > 0.5:
                print(
                    "Gripper safety hold:",
                    f"requested={requested_pos:.5f}",
                    f"actual={actual_pos:.5f}",
                    f"safe={safe_pos:.5f}",
                    f"torque={torque:.3f}",
                    f"limit={self.gripper_safe_torque:.3f}",
                )
                self._last_gripper_safety_print = now
            return safe_pos
        return requested_pos

    def _make_eef_cmd(self, pose, target_time: Optional[float] = None, gripper_pos=None):
        assert self.controller is not None
        pose = np.asarray(pose, dtype=np.float64)
        assert pose.shape == (6,)

        cmd = arx5.EEFState()
        cmd.pose_6d()[:] = pose
        if gripper_pos is None:
            gripper_pos = self.controller.get_eef_cmd().gripper_pos
        cmd.gripper_pos = self._safe_gripper_target(float(gripper_pos))
        now_wall = time.time()
        now_robot = self.controller.get_timestamp()
        if target_time is None:
            dt = self.preview_time
        else:
            dt = max(float(target_time) - now_wall, self.preview_time)
        cmd.timestamp = now_robot + dt
        return cmd, pose

    def send_eef_cmd(self, pose, target_time: Optional[float] = None, gripper_pos=None):
        """Send one immediate EEF command without SDK trajectory replacement.

        The old ARX5 DP deployment keeps its own trajectory interpolator and
        streams the interpolated pose each control tick. This method supports
        that control mode by forcing set_eef_cmd regardless of command_mode.
        """
        assert self.controller is not None
        cmd, pose = self._make_eef_cmd(pose, target_time=target_time, gripper_pos=gripper_pos)
        self.controller.set_eef_cmd(cmd)
        self._last_target_pose = pose.copy()
        self._last_target_time = float(time.time() if target_time is None else target_time)

    def schedule_waypoint(self, pose, target_time: Optional[float] = None, gripper_pos=None):
        """Send one end-effector target to the ARX5 SDK.

        This is the single-action control path used by human SpaceMouse mode
        and by policy dry/fallback modes.
        """
        cmd, pose = self._make_eef_cmd(pose, target_time=target_time, gripper_pos=gripper_pos)
        if self.command_mode == "traj":
            self.controller.set_eef_traj([cmd])
        else:
            self.controller.set_eef_cmd(cmd)
        self._last_target_pose = pose.copy()
        self._last_target_time = float(time.time() if target_time is None else target_time)

    def schedule_waypoints(self, actions, target_times, gripper_margin: float = 0.0):
        """Schedule a chunk of absolute 7D actions.

        Each action is [x, y, z, rx, ry, rz, gripper_width]. target_times are
        wall-clock timestamps. In trajectory mode this submits the whole chunk
        to the SDK; in command mode it falls back to the latest action.

        This is the main DP/ACT deployment control path. If online policy
        motion feels jerky, inspect the action chunk and timestamps before they
        arrive here, then inspect the generated SDK EEFState list below.
        """
        assert self.controller is not None
        actions = np.asarray(actions, dtype=np.float64)
        target_times = np.asarray(target_times, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected actions with shape (N, 7), got {actions.shape}.")
        if target_times.shape != (len(actions),):
            raise ValueError(
                f"Expected target_times shape {(len(actions),)}, got {target_times.shape}."
            )
        if not self.tracking_is_healthy():
            pos_error, rot_error, state = self.tracking_error()
            now = time.time()
            if now - self._last_tracking_guard_print > 0.5:
                print(
                    "Tracking guard hold:",
                    f"pos_error={pos_error:.4f}",
                    f"rot_error={rot_error:.4f}",
                )
                self._last_tracking_guard_print = now
            self.schedule_waypoint(
                state["ActualTCPPose"],
                target_time=float(time.time() + self.preview_time),
                gripper_pos=float(state["gripper_pos"][0]),
            )
            return
        if len(actions) == 0:
            return

        keep = np.concatenate([[True], np.diff(target_times) > 1e-5])
        if not np.all(keep):
            actions = actions[keep]
            target_times = target_times[keep]
            if len(actions) == 0:
                return

        gripper_lo = float(gripper_margin)
        gripper_hi = float(self.robot_config.gripper_width) - float(gripper_margin)
        now_wall = time.time()
        now_robot = self.controller.get_timestamp()

        if self.command_mode != "traj":
            last = actions[-1]
            self.schedule_waypoint(
                last[:6],
                target_time=float(target_times[-1]),
                gripper_pos=float(np.clip(last[6], gripper_lo, gripper_hi)),
            )
            return

        cmds = []
        for action, target_time in zip(actions, target_times):
            cmd = arx5.EEFState()
            cmd.pose_6d()[:] = action[:6]
            requested_gripper = float(np.clip(action[6], gripper_lo, gripper_hi))
            cmd.gripper_pos = self._safe_gripper_target(requested_gripper)
            dt = max(float(target_time) - now_wall, self.preview_time)
            cmd.timestamp = now_robot + dt
            cmds.append(cmd)
        self.controller.set_eef_traj(cmds)
        self._last_target_pose = actions[-1, :6].copy()
        self._last_target_time = float(target_times[-1])
