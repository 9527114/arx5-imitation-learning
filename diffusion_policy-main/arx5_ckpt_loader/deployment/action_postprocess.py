import numpy as np


def clamp_action_chunk_delta(
    action_chunk,
    anchor_action,
    max_pos_step,
    max_rot_step,
    max_gripper_step,
):
    if max_pos_step is None and max_rot_step is None and max_gripper_step is None:
        return action_chunk, False
    action_chunk = np.asarray(action_chunk, dtype=np.float64).copy()
    prev_action = np.asarray(anchor_action, dtype=np.float64).copy()
    clipped = False
    for idx in range(len(action_chunk)):
        action = action_chunk[idx].copy()
        if max_pos_step is not None:
            dpos = action[:3] - prev_action[:3]
            norm = float(np.linalg.norm(dpos))
            if norm > max_pos_step:
                action[:3] = prev_action[:3] + dpos / max(norm, 1e-9) * max_pos_step
                clipped = True
        if max_rot_step is not None:
            drot = action[3:6] - prev_action[3:6]
            norm = float(np.linalg.norm(drot))
            if norm > max_rot_step:
                action[3:6] = prev_action[3:6] + drot / max(norm, 1e-9) * max_rot_step
                clipped = True
        if max_gripper_step is not None:
            dgrip = float(action[6] - prev_action[6])
            if abs(dgrip) > max_gripper_step:
                action[6] = prev_action[6] + np.sign(dgrip) * max_gripper_step
                clipped = True
        action_chunk[idx] = action
        prev_action = action
    return action_chunk, clipped


def blend_chunk_start(action_chunk, anchor_action, blend_steps):
    action_chunk = np.asarray(action_chunk, dtype=np.float64).copy()
    blend_steps = max(0, int(blend_steps))
    if blend_steps <= 0 or len(action_chunk) == 0:
        return action_chunk
    n = min(blend_steps, len(action_chunk))
    anchor = np.asarray(anchor_action, dtype=np.float64)
    for i in range(n):
        alpha = float(i + 1) / float(n + 1)
        action_chunk[i] = (1.0 - alpha) * anchor + alpha * action_chunk[i]
    return action_chunk


def apply_action_deadband(action_chunk, anchor_action, pos_deadband, rot_deadband, gripper_deadband):
    if pos_deadband <= 0 and rot_deadband <= 0 and gripper_deadband <= 0:
        return action_chunk, False
    action_chunk = np.asarray(action_chunk, dtype=np.float64).copy()
    prev = np.asarray(anchor_action, dtype=np.float64).copy()
    applied = False
    for idx in range(len(action_chunk)):
        action = action_chunk[idx].copy()
        if pos_deadband > 0 and np.linalg.norm(action[:3] - prev[:3]) < pos_deadband:
            action[:3] = prev[:3]
            applied = True
        if rot_deadband > 0 and np.linalg.norm(action[3:6] - prev[3:6]) < rot_deadband:
            action[3:6] = prev[3:6]
            applied = True
        if gripper_deadband > 0 and abs(float(action[6] - prev[6])) < gripper_deadband:
            action[6] = prev[6]
            applied = True
        action_chunk[idx] = action
        prev = action
    return action_chunk, applied


def smooth_action_chunk(
    action_chunk,
    anchor_action,
    pos_alpha=1.0,
    rot_alpha=1.0,
    gripper_alpha=1.0,
):
    """Apply causal exponential smoothing over policy waypoints.

    alpha=1.0 leaves the chunk unchanged. Smaller values track the new policy
    chunk more slowly from the previous waypoint, reducing 20Hz waypoint jitter
    at the cost of lag.
    """
    action_chunk = np.asarray(action_chunk, dtype=np.float64).copy()
    if len(action_chunk) == 0:
        return action_chunk, False
    pos_alpha = float(pos_alpha)
    rot_alpha = float(rot_alpha)
    gripper_alpha = float(gripper_alpha)
    if pos_alpha >= 1.0 and rot_alpha >= 1.0 and gripper_alpha >= 1.0:
        return action_chunk, False
    pos_alpha = np.clip(pos_alpha, 0.0, 1.0)
    rot_alpha = np.clip(rot_alpha, 0.0, 1.0)
    gripper_alpha = np.clip(gripper_alpha, 0.0, 1.0)
    prev = np.asarray(anchor_action, dtype=np.float64).copy()
    for idx in range(len(action_chunk)):
        raw = action_chunk[idx].copy()
        action_chunk[idx, :3] = (1.0 - pos_alpha) * prev[:3] + pos_alpha * raw[:3]
        action_chunk[idx, 3:6] = (1.0 - rot_alpha) * prev[3:6] + rot_alpha * raw[3:6]
        action_chunk[idx, 6] = (1.0 - gripper_alpha) * prev[6] + gripper_alpha * raw[6]
        prev = action_chunk[idx]
    return action_chunk, True


def make_anchor_action(robot_state, target_pose, target_gripper, action_anchor):
    if action_anchor == "target":
        return np.concatenate(
            [
                np.asarray(target_pose, dtype=np.float64),
                np.asarray([target_gripper], dtype=np.float64),
            ]
        )
    return np.concatenate(
        [
            np.asarray(robot_state["ActualTCPPose"], dtype=np.float64),
            np.asarray([robot_state["gripper_pos"][0]], dtype=np.float64),
        ]
    )
