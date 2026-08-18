import os

import numpy as np


GRIPPER_OPEN_READOUT = float(os.environ.get("ARX5_GRIPPER_OPEN_READOUT", "5.04635"))
GRIPPER_WIDTH = float(os.environ.get("ARX5_GRIPPER_WIDTH", "0.088"))
GRIPPER_TORQUE_MAX = float(os.environ.get("ARX5_GRIPPER_TORQUE_MAX", "2.0"))
GRIPPER_KP = float(os.environ.get("ARX5_GRIPPER_KP", "5.2"))
GRIPPER_KD = float(os.environ.get("ARX5_GRIPPER_KD", "0.8"))
ARM_GAIN_MODE = os.environ.get("ARX5_ARM_GAIN_MODE", "default")
ARM_KP_SCALE = float(os.environ.get("ARX5_ARM_KP_SCALE", "1.0"))
ARM_KD_SCALE = float(os.environ.get("ARX5_ARM_KD_SCALE", "1.0"))
SPACEMOUSE_TRANSLATION_SIGN = np.array(
    [
        float(os.environ.get("ARX5_SPACEMOUSE_X_SIGN", "-1")),
        float(os.environ.get("ARX5_SPACEMOUSE_Y_SIGN", "-1")),
        float(os.environ.get("ARX5_SPACEMOUSE_Z_SIGN", "1")),
    ],
    dtype=np.float64,
)
SPACEMOUSE_ROTATION_SIGN = np.array(
    [
        float(os.environ.get("ARX5_SPACEMOUSE_ROLL_SIGN", "1")),
        float(os.environ.get("ARX5_SPACEMOUSE_PITCH_SIGN", "1")),
        float(os.environ.get("ARX5_SPACEMOUSE_YAW_SIGN", "1")),
    ],
    dtype=np.float64,
)


def apply_local_robot_config(robot_config):
    """Apply lab-local calibration values before controller creation."""
    if robot_config.robot_model == "X5":
        robot_config.gripper_open_readout = GRIPPER_OPEN_READOUT
        robot_config.gripper_width = GRIPPER_WIDTH
        robot_config.gripper_torque_max = GRIPPER_TORQUE_MAX
    return robot_config


def apply_local_controller_gain(
    controller,
    arm_gain_mode=None,
    arm_kp_scale=None,
    arm_kd_scale=None,
):
    if arm_gain_mode is None:
        arm_gain_mode = ARM_GAIN_MODE
    if arm_kp_scale is None:
        arm_kp_scale = ARM_KP_SCALE
    if arm_kd_scale is None:
        arm_kd_scale = ARM_KD_SCALE

    gain = controller.get_gain()
    controller_config = controller.get_controller_config()
    if arm_gain_mode == "default":
        gain.kp()[:] = controller_config.default_kp * float(arm_kp_scale)
        gain.kd()[:] = controller_config.default_kd * float(arm_kd_scale)
    elif arm_gain_mode == "pro":
        gain.kp()[:] = (
            np.array([160.0, 160.0, 180.0, 60.0, 60.0, 20.0])
            * float(arm_kp_scale)
        )
        gain.kd()[:] = controller_config.default_kd * float(arm_kd_scale)
        if gain.kd().shape[0] > 3:
            gain.kd()[3] = 0.5 * float(arm_kd_scale)
    elif arm_gain_mode == "damping":
        gain.kp()[:] = 0.0
        gain.kd()[:] = controller_config.default_kd * float(arm_kd_scale)
    else:
        raise ValueError(f"Unsupported arm_gain_mode: {arm_gain_mode}")
    gain.gripper_kp = GRIPPER_KP
    gain.gripper_kd = GRIPPER_KD
    controller.set_gain(gain)
    return gain


def apply_local_spacemouse_mapping(state):
    mapped = np.asarray(state).copy()
    mapped[:3] *= SPACEMOUSE_TRANSLATION_SIGN
    mapped[3:] *= SPACEMOUSE_ROTATION_SIGN
    return mapped
