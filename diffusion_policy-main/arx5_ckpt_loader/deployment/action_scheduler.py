from dataclasses import dataclass

import numpy as np


@dataclass
class TimedActionCandidate:
    action_chunk: np.ndarray
    action_timestamps: np.ndarray
    action_indices: np.ndarray
    action_start_idx: int
    submit_steps: int
    timestamp_base: float


def get_policy_action_sequence(result, cfg):
    """Return the full deployable action sequence aligned to the policy action start."""
    if "action_pred" in result:
        pred_start = int(cfg.n_obs_steps) - 1
        return result["action_pred"][0, pred_start:].detach().to("cpu").numpy()
    return result["action"][0].detach().to("cpu").numpy()


def make_timed_action_candidate(
    action_sequence,
    obs_timestamps,
    curr_time: float,
    dt: float,
    steps_per_inference: int,
    submit_extra_steps: int,
    command_latency: float,
    action_exec_latency: float,
    timestamp_mode: str,
):
    """Select an action_pred slice and assign wall-clock execution timestamps.

    `compensated` mode is latency-aware: it maps current execution time back to
    the action_pred index instead of taking the first N actions and dropping
    stale points afterwards.
    """
    action_sequence = np.asarray(action_sequence, dtype=np.float64)
    obs_timestamps = np.asarray(obs_timestamps, dtype=np.float64)
    if action_sequence.ndim != 2:
        raise ValueError(f"Expected action_sequence shape (T, Da), got {action_sequence.shape}.")
    if len(action_sequence) == 0:
        raise ValueError("action_sequence is empty.")
    if len(obs_timestamps) == 0:
        raise ValueError("obs_timestamps is empty.")

    submit_steps = int(steps_per_inference) + max(0, int(submit_extra_steps))
    submit_steps = max(1, submit_steps)
    action_start_idx = 0

    if timestamp_mode == "obs":
        timestamp_base = float(obs_timestamps[-1]) + float(command_latency)
    elif timestamp_mode == "compensated":
        timestamp_base = float(obs_timestamps[-1]) + float(command_latency)
        min_exec_time = float(curr_time) + max(float(command_latency), float(action_exec_latency))
        action_start_idx = int(np.ceil((min_exec_time - timestamp_base) / max(float(dt), 1e-6)))
        action_start_idx = max(0, action_start_idx)
    elif timestamp_mode == "now":
        timestamp_base = float(curr_time) + float(command_latency)
    else:
        raise ValueError(f"Unknown timestamp_mode: {timestamp_mode}")

    action_start_idx = min(action_start_idx, max(0, len(action_sequence) - 1))
    action_end_idx = min(action_start_idx + submit_steps, len(action_sequence))
    action_indices = np.arange(action_start_idx, action_end_idx, dtype=np.int64)
    action_chunk = action_sequence[action_indices].copy()
    action_timestamps = action_indices.astype(np.float64) * float(dt) + timestamp_base

    return TimedActionCandidate(
        action_chunk=action_chunk,
        action_timestamps=action_timestamps,
        action_indices=action_indices,
        action_start_idx=int(action_start_idx),
        submit_steps=int(submit_steps),
        timestamp_base=float(timestamp_base),
    )


def filter_future_actions(
    action_chunk,
    action_timestamps,
    curr_time: float,
    action_exec_latency: float,
    command_latency: float,
    dt: float,
):
    action_chunk = np.asarray(action_chunk, dtype=np.float64)
    action_timestamps = np.asarray(action_timestamps, dtype=np.float64)
    is_new = action_timestamps > (float(curr_time) + float(action_exec_latency))
    if np.sum(is_new) == 0:
        return (
            action_chunk[[-1]],
            np.asarray([float(curr_time) + max(float(command_latency), float(dt))], dtype=np.float64),
            True,
        )
    return action_chunk[is_new], action_timestamps[is_new], False
