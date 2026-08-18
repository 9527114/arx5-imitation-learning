import time
from typing import Dict, Optional

import numpy as np

from arx5_collector.sdk_path import ensure_arx5_sdk_path

ensure_arx5_sdk_path()

import arx5_interface as arx5
from arx5_local_config import apply_local_controller_gain, apply_local_robot_config


def _attempt_iter(attempts):
    attempts = int(attempts)
    if attempts <= 0:
        idx = 0
        while True:
            yield idx, "until_success"
            idx += 1
    else:
        for idx in range(attempts):
            yield idx, str(attempts)


class JointActRobot:
    """ARX5 joint-space runtime for ACT joint checkpoints.

    ACT joint actions are absolute [q1..q6, gripper_width] targets. They must
    be sent through Arx5JointController rather than the Cartesian EEF API.
    """

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
        tracking_joint_error_limit: float = 0.25,
    ):
        if command_mode not in ("cmd", "traj"):
            raise ValueError(f"Unsupported command_mode: {command_mode}")
        self.model = model
        self.interface = interface
        self.log_level = log_level
        self.reset_to_home_on_start = reset_to_home
        self.preview_time = float(preview_time)
        self.command_mode = command_mode
        self.arm_gain_mode = arm_gain_mode
        self.arm_kp_scale = float(arm_kp_scale)
        self.arm_kd_scale = float(arm_kd_scale)
        self.gripper_safe_torque = None if gripper_safe_torque is None else float(gripper_safe_torque)
        self.gripper_safe_margin = float(gripper_safe_margin)
        self.tracking_pos_error_limit = float(tracking_pos_error_limit)
        self.tracking_rot_error_limit = float(tracking_rot_error_limit)
        self.tracking_joint_error_limit = float(tracking_joint_error_limit)
        self.controller = None
        self.robot_config = None
        self.controller_config = None
        self.start_gain = None
        self._last_state = None
        self._last_target_joint = None
        self._last_target_time = None
        self._last_gripper_safety_print = 0.0
        self._last_tracking_guard_print = 0.0

    def start(self):
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(self.model)
        apply_local_robot_config(robot_config)
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", robot_config.joint_dof
        )
        controller = arx5.Arx5JointController(robot_config, controller_config, self.interface)
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
        self._last_target_joint = state["TargetQ"].copy()
        self._last_target_time = state["robot_receive_timestamp"]

    def stop(self):
        if self.controller is not None:
            self.controller.set_to_damping()
        self.controller = None

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

    def get_gain_summary(self):
        assert self.controller is not None
        gain = self.controller.get_gain()
        return {
            "kp": np.asarray(gain.kp(), dtype=np.float64).copy(),
            "kd": np.asarray(gain.kd(), dtype=np.float64).copy(),
            "gripper_kp": float(gain.gripper_kp),
            "gripper_kd": float(gain.gripper_kd),
        }

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
            self._last_target_joint = state["TargetQ"].copy()
            self._last_target_time = state["robot_receive_timestamp"]
            return state
        finally:
            if restore_gain:
                self.restore_runtime_gain()

    def reset_to_joints(
        self,
        joints,
        gripper: float,
        duration: float = 2.0,
        settle_time: float = 0.35,
        attempts: int = 1,
        joint_tolerance: Optional[float] = 0.08,
    ):
        joints = np.asarray(joints, dtype=np.float64)
        final_state = None
        for attempt, attempts_label in _attempt_iter(attempts):
            self.schedule_waypoint(joints, target_time=time.time() + float(duration), gripper_pos=gripper)
            time.sleep(max(0.0, float(duration)) + max(0.0, float(settle_time)))
            final_state = self.get_state()
            joint_error = float(np.linalg.norm(final_state["ActualQ"] - joints))
            print(
                "reset joint check",
                f"attempt={attempt + 1}/{attempts_label}",
                f"joint_error={joint_error:.5f}",
            )
            if joint_tolerance is None or joint_error <= float(joint_tolerance):
                return final_state, True
        return final_state, False

    def get_state(self) -> Dict[str, np.ndarray]:
        assert self.controller is not None
        joint_state = self.controller.get_joint_state()
        joint_cmd = self.controller.get_joint_cmd()
        eef_state = self.controller.get_eef_state()
        now = time.time()

        actual_joint = np.asarray(joint_state.pos(), dtype=np.float64).copy()
        actual_joint_vel = np.asarray(joint_state.vel(), dtype=np.float64).copy()
        target_joint = np.asarray(joint_cmd.pos(), dtype=np.float64).copy()
        target_joint_vel = np.asarray(joint_cmd.vel(), dtype=np.float64).copy()
        actual_pose = np.asarray(eef_state.pose_6d(), dtype=np.float64).copy()

        actual_pose_vel = np.zeros(6, dtype=np.float64)
        if self._last_state is not None:
            dt = now - float(self._last_state["robot_receive_timestamp"])
            if dt > 1e-6:
                actual_pose_vel = (actual_pose - self._last_state["ActualTCPPose"]) / dt

        state = {
            "ActualTCPPose": actual_pose,
            "ActualTCPSpeed": actual_pose_vel,
            "ActualQ": actual_joint,
            "ActualQd": actual_joint_vel,
            "TargetTCPPose": actual_pose.copy(),
            "TargetTCPSpeed": np.zeros(6, dtype=np.float64),
            "TargetQ": target_joint,
            "TargetQd": target_joint_vel,
            "robot_receive_timestamp": now,
            "gripper_pos": np.array([joint_state.gripper_pos], dtype=np.float64),
            "target_gripper_pos": np.array([joint_cmd.gripper_pos], dtype=np.float64),
            "gripper_torque": np.array([joint_state.gripper_torque], dtype=np.float64),
        }
        self._last_state = state
        return state

    def tracking_error(self):
        state = self.get_state()
        joint_error = state["TargetQ"] - state["ActualQ"]
        return float(np.linalg.norm(joint_error)), 0.0, state

    def tracking_is_healthy(self):
        if self.tracking_joint_error_limit <= 0:
            return True
        joint_error, _, _ = self.tracking_error()
        return joint_error <= self.tracking_joint_error_limit

    def _safe_gripper_target(self, requested_pos: float) -> float:
        if self.gripper_safe_torque is None:
            return requested_pos
        state = self.get_state()
        actual_pos = float(state["gripper_pos"][0])
        torque = abs(float(state["gripper_torque"][0]))
        is_closing = requested_pos < actual_pos
        if is_closing and torque >= self.gripper_safe_torque:
            safe_pos = max(actual_pos - self.gripper_safe_margin, 0.0)
            now = time.time()
            if now - self._last_gripper_safety_print > 0.5:
                print(
                    "Joint gripper safety hold:",
                    f"requested={requested_pos:.5f}",
                    f"actual={actual_pos:.5f}",
                    f"safe={safe_pos:.5f}",
                    f"torque={torque:.3f}",
                )
                self._last_gripper_safety_print = now
            return safe_pos
        return requested_pos

    def _make_joint_cmd(self, joints, target_time: Optional[float] = None, gripper_pos=None):
        assert self.controller is not None
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (self.robot_config.joint_dof,):
            raise ValueError(f"Expected joints shape {(self.robot_config.joint_dof,)}, got {joints.shape}.")
        if gripper_pos is None:
            gripper_pos = self.controller.get_joint_cmd().gripper_pos

        now_wall = time.time()
        now_robot = self.controller.get_timestamp()
        dt = self.preview_time if target_time is None else max(float(target_time) - now_wall, self.preview_time)
        cmd = arx5.JointState(self.robot_config.joint_dof)
        cmd.pos()[:] = joints
        cmd.gripper_pos = self._safe_gripper_target(float(gripper_pos))
        cmd.timestamp = now_robot + dt
        return cmd, joints

    def schedule_waypoint(self, joints, target_time: Optional[float] = None, gripper_pos=None):
        cmd, joints = self._make_joint_cmd(joints, target_time=target_time, gripper_pos=gripper_pos)
        if self.command_mode == "traj":
            self.controller.set_joint_traj([cmd])
        else:
            self.controller.set_joint_cmd(cmd)
        self._last_target_joint = joints.copy()
        self._last_target_time = float(time.time() if target_time is None else target_time)

    def send_joint_cmd(self, joints, target_time: Optional[float] = None, gripper_pos=None):
        """Stream one joint-space command without replacing an SDK trajectory."""
        cmd, joints = self._make_joint_cmd(joints, target_time=target_time, gripper_pos=gripper_pos)
        self.controller.set_joint_cmd(cmd)
        self._last_target_joint = joints.copy()
        self._last_target_time = float(time.time() if target_time is None else target_time)

    def schedule_waypoints(self, actions, target_times, gripper_margin: float = 0.0):
        assert self.controller is not None
        actions = np.asarray(actions, dtype=np.float64)
        target_times = np.asarray(target_times, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected joint actions with shape (N, 7), got {actions.shape}.")
        if target_times.shape != (len(actions),):
            raise ValueError(f"Expected target_times shape {(len(actions),)}, got {target_times.shape}.")
        if len(actions) == 0:
            return

        keep = np.concatenate([[True], np.diff(target_times) > 1e-5])
        actions = actions[keep]
        target_times = target_times[keep]
        if len(actions) == 0:
            return

        gripper_lo = float(gripper_margin)
        gripper_hi = float(self.robot_config.gripper_width) - float(gripper_margin)
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
            cmd, _ = self._make_joint_cmd(
                action[:6],
                target_time=float(target_time),
                gripper_pos=float(np.clip(action[6], gripper_lo, gripper_hi)),
            )
            cmds.append(cmd)
        self.controller.set_joint_traj(cmds)
        self._last_target_joint = actions[-1, :6].copy()
        self._last_target_time = float(target_times[-1])
