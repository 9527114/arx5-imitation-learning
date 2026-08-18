import time

import numpy as np


def summarize_policy_alignment(
    obs_timestamps,
    raw_action_chunk,
    action_chunk,
    action_timestamps,
    curr_time=None,
):
    if curr_time is None:
        curr_time = time.time()
    obs_timestamps = np.asarray(obs_timestamps, dtype=np.float64)
    action_timestamps = np.asarray(action_timestamps, dtype=np.float64)
    raw_action_chunk = np.asarray(raw_action_chunk, dtype=np.float64)
    action_chunk = np.asarray(action_chunk, dtype=np.float64)

    summary = {
        "obs_count": int(len(obs_timestamps)),
        "obs_latest_age": float(curr_time - obs_timestamps[-1]) if len(obs_timestamps) else None,
        "raw_action_len": int(len(raw_action_chunk)),
        "scheduled_action_len": int(len(action_chunk)),
        "dropped_action_len": int(max(0, len(raw_action_chunk) - len(action_chunk))),
        "timestamp_dt0": float(action_timestamps[0] - curr_time) if len(action_timestamps) else None,
        "timestamp_dtN": float(action_timestamps[-1] - curr_time) if len(action_timestamps) else None,
    }
    if len(action_chunk) > 1:
        dpos = np.linalg.norm(np.diff(action_chunk[:, :3], axis=0), axis=1)
        drot = np.linalg.norm(np.diff(action_chunk[:, 3:6], axis=0), axis=1)
        summary.update(
            {
                "chunk_pos_step_max": float(dpos.max()),
                "chunk_pos_step_mean": float(dpos.mean()),
                "chunk_rot_step_max": float(drot.max()),
                "chunk_rot_step_mean": float(drot.mean()),
            }
        )
    return summary
