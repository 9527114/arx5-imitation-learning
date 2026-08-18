import time

import numpy as np


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


def reset_to_home_checked(
    robot,
    attempts=3,
    settle_time=0.35,
    pos_tolerance=0.006,
    rot_tolerance=0.05,
    joint_tolerance=0.08,
):
    final_state = None
    for attempt, attempts_label in _attempt_iter(attempts):
        final_state = robot.reset_to_home()
        if settle_time > 0:
            time.sleep(float(settle_time))
            final_state = robot.get_state()

        pos_error = float(
            np.linalg.norm(final_state["TargetTCPPose"][:3] - final_state["ActualTCPPose"][:3])
        )
        rot_error = float(
            np.linalg.norm(final_state["TargetTCPPose"][3:6] - final_state["ActualTCPPose"][3:6])
        )
        joint_error = float(np.linalg.norm(final_state["TargetQ"] - final_state["ActualQ"]))
        print(
            "reset home check",
            f"attempt={attempt + 1}/{attempts_label}",
            f"pos_error={pos_error:.5f}",
            f"rot_error={rot_error:.5f}",
            f"joint_error={joint_error:.5f}",
        )

        pos_ok = pos_tolerance is None or pos_error <= float(pos_tolerance)
        rot_ok = rot_tolerance is None or rot_error <= float(rot_tolerance)
        joint_ok = joint_tolerance is None or joint_error <= float(joint_tolerance)
        if pos_ok and rot_ok and joint_ok:
            return final_state, True
    return final_state, False


def reset_to_pose(
    robot,
    pose,
    gripper,
    duration,
    frequency,
    command_latency,
    attempts=1,
    settle_time=0.0,
    pos_tolerance=None,
    rot_tolerance=None,
):
    pose = np.asarray(pose, dtype=np.float64)
    final_state = None
    for attempt, attempts_label in _attempt_iter(attempts):
        state = robot.get_state()
        start_pose = state["ActualTCPPose"].copy()
        start_gripper = float(state["gripper_pos"][0])
        steps = max(2, int(float(duration) * float(frequency)))
        dt = float(duration) / float(steps)
        for idx in range(steps):
            alpha = float(idx + 1) / float(steps)
            target_pose = (1.0 - alpha) * start_pose + alpha * pose
            target_gripper = (1.0 - alpha) * start_gripper + alpha * float(gripper)
            robot.schedule_waypoint(
                target_pose,
                target_time=time.time() + max(command_latency, dt),
                gripper_pos=target_gripper,
            )
            time.sleep(dt)
        hold_until = time.time() + max(0.0, float(settle_time))
        while time.time() < hold_until:
            robot.schedule_waypoint(
                pose,
                target_time=time.time() + max(command_latency, 0.05),
                gripper_pos=float(gripper),
            )
            time.sleep(0.05)
        final_state = robot.get_state()
        if pos_tolerance is None and rot_tolerance is None:
            break
        pos_error = float(np.linalg.norm(final_state["ActualTCPPose"][:3] - pose[:3]))
        rot_error = float(np.linalg.norm(final_state["ActualTCPPose"][3:6] - pose[3:6]))
        print(
            "reset check",
            f"attempt={attempt + 1}/{attempts_label}",
            f"pos_error={pos_error:.5f}",
            f"rot_error={rot_error:.5f}",
        )
        pos_ok = pos_tolerance is None or pos_error <= float(pos_tolerance)
        rot_ok = rot_tolerance is None or rot_error <= float(rot_tolerance)
        if pos_ok and rot_ok:
            break
    if final_state is None:
        final_state = robot.get_state()
    pos_error = float(np.linalg.norm(final_state["ActualTCPPose"][:3] - pose[:3]))
    rot_error = float(np.linalg.norm(final_state["ActualTCPPose"][3:6] - pose[3:6]))
    pos_ok = pos_tolerance is None or pos_error <= float(pos_tolerance)
    rot_ok = rot_tolerance is None or rot_error <= float(rot_tolerance)
    return final_state, bool(pos_ok and rot_ok)
