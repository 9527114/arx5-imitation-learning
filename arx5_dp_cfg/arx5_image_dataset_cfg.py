from typing import Dict
import copy
import hashlib
import json
import os
import shutil

import cv2
import numpy as np
import torch
import zarr
from filelock import FileLock
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.real_world.video_recorder import read_video

VIDEO_ALIGNMENT_SAFETY_FRAMES = 2


class Arx5ImageDataset(BaseImageDataset):
    """Dataset for ARX5 collector output.

    Expected input directory:
      replay_buffer.zarr/
      videos/<episode_id>/<camera_id>.mp4

    Video stems become DP image keys: 0.mp4 -> camera_0, 1.mp4 -> camera_1.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        use_cache=True,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        target_frequency=None,
        delta_action=False,
        trim_static_start_end=False,
        static_pos_threshold=1e-4,
        static_rot_threshold=1e-4,
        static_gripper_threshold=1e-4,
        static_pad_before=2,
        static_pad_after=2,
        min_episode_steps=16,
        cache_dir=None,
        cache_path=None,
        prev_cond_steps=8,
        prev_action_mode="future",
    ):
        assert os.path.isdir(dataset_path), dataset_path

        if use_cache:
            processing_config = {
                "shape_meta": OmegaConf.to_container(shape_meta),
                "target_frequency": target_frequency,
                "delta_action": delta_action,
                "trim_static_start_end": trim_static_start_end,
                "static_pos_threshold": static_pos_threshold,
                "static_rot_threshold": static_rot_threshold,
                "static_gripper_threshold": static_gripper_threshold,
                "static_pad_before": static_pad_before,
                "static_pad_after": static_pad_after,
                "min_episode_steps": min_episode_steps,
                "trim_to_video_frames": True,
            }
            shape_meta_json = json.dumps(
                processing_config, sort_keys=True
            )
            shape_meta_hash = hashlib.md5(shape_meta_json.encode("utf-8")).hexdigest()
            if cache_path is not None and str(cache_path).strip() != "":
                cache_zarr_path = str(cache_path)
                cache_base_dir = os.path.dirname(cache_zarr_path)
            else:
                cache_base_dir = dataset_path if cache_dir is None else cache_dir
                cache_zarr_path = os.path.join(cache_base_dir, f"arx5_{shape_meta_hash}.zarr.zip")
            os.makedirs(cache_base_dir, exist_ok=True)
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on ARX5 dataset cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    if cache_path is not None and str(cache_path).strip() != "":
                        raise FileNotFoundError(f"Explicit ARX5 cache not found: {cache_zarr_path}")
                    try:
                        print("ARX5 cache does not exist. Creating.")
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore(),
                            target_frequency=target_frequency,
                        )
                        replay_buffer = _process_replay_buffer(
                            replay_buffer=replay_buffer,
                            target_frequency=None,
                            delta_action=delta_action,
                            trim_static_start_end=trim_static_start_end,
                            static_pos_threshold=static_pos_threshold,
                            static_rot_threshold=static_rot_threshold,
                            static_gripper_threshold=static_gripper_threshold,
                            static_pad_before=static_pad_before,
                            static_pad_after=static_pad_after,
                            min_episode_steps=min_episode_steps,
                        )
                        print("Saving ARX5 cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as exc:
                        if os.path.exists(cache_zarr_path):
                            if os.path.isdir(cache_zarr_path):
                                shutil.rmtree(cache_zarr_path)
                            else:
                                os.remove(cache_zarr_path)
                        raise exc
                else:
                    print("Loading cached ARX5 ReplayBuffer from disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store,
                            store=zarr.MemoryStore(),
                        )
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore(),
                target_frequency=target_frequency,
            )
            replay_buffer = _process_replay_buffer(
                replay_buffer=replay_buffer,
                target_frequency=None,
                delta_action=delta_action,
                trim_static_start_end=trim_static_start_end,
                static_pos_threshold=static_pos_threshold,
                static_rot_threshold=static_rot_threshold,
                static_gripper_threshold=static_gripper_threshold,
                static_pad_before=static_pad_before,
                static_pad_after=static_pad_after,
                min_episode_steps=min_episode_steps,
            )

        rgb_keys = list()
        lowdim_keys = list()
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.prev_cond_steps = int(prev_cond_steps)
        self.prev_action_mode = str(prev_action_mode)
        if self.prev_action_mode not in ("future", "past"):
            raise ValueError(
                f"prev_action_mode must be 'future' or 'past', got {self.prev_action_mode!r}"
            )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
        obs_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][obs_slice], -1, 1).astype(
                np.float32
            ) / 255.0
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][obs_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        action_dim = action.shape[-1]
        prev_action = np.zeros(
            (self.prev_cond_steps, action_dim),
            dtype=np.float32,
        )
        prev_action_mask = np.zeros((self.prev_cond_steps,), dtype=np.float32)
        if self.prev_cond_steps > 0 and self.prev_action_mode == "past":
            episode_starts = self.replay_buffer.episode_ends[:] - self.replay_buffer.episode_lengths[:]
            episode_idx = int(np.searchsorted(self.replay_buffer.episode_ends[:], buffer_start_idx, side="right"))
            episode_start = int(episode_starts[episode_idx])
            prev_start = max(episode_start, int(buffer_start_idx) - self.prev_cond_steps)
            prev_end = int(buffer_start_idx)
            if prev_end > prev_start:
                prev = np.asarray(self.replay_buffer["action"][prev_start:prev_end], dtype=np.float32)
                n_prev = len(prev)
                prev_action[-n_prev:] = prev
                prev_action_mask[-n_prev:] = 1.0
        elif self.prev_cond_steps > 0 and self.prev_action_mode == "future":
            n_valid = min(self.prev_cond_steps, len(action))
            if n_valid > 0:
                prev_action[:n_valid] = action[:n_valid]
                prev_action_mask[:n_valid] = 1.0
            # Mask actions that come from SequenceSampler padding at episode boundaries.
            for i in range(self.prev_cond_steps):
                sample_i = i + int(self.n_latency_steps)
                if sample_i < int(sample_start_idx) or sample_i >= int(sample_end_idx):
                    prev_action_mask[i] = 0.0

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
            "prev_action": torch.from_numpy(prev_action),
            "prev_action_mask": torch.from_numpy(prev_action_mask),
        }


def _get_replay_buffer(dataset_path, shape_meta, store, target_frequency=None):
    rgb_keys = list()
    lowdim_keys = list()
    out_resolutions = dict()

    for key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        shape = tuple(attr.get("shape"))
        if obs_type == "rgb":
            rgb_keys.append(key)
            c, h, w = shape
            assert c == 3
            out_resolutions[key] = (w, h)
        elif obs_type == "low_dim":
            lowdim_keys.append(key)

    action_shape = tuple(shape_meta["action"]["shape"])
    assert len(action_shape) == 1

    cv2.setNumThreads(1)
    episode_sample_indices = _get_episode_sample_indices(
        dataset_path=dataset_path,
        target_frequency=target_frequency,
    )
    episode_sample_indices = _trim_sample_indices_to_available_video(
        dataset_path=dataset_path,
        image_keys=rgb_keys,
        episode_sample_indices=episode_sample_indices,
    )
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys + ["action", "timestamp"],
            image_keys=rgb_keys,
            episode_sample_indices=episode_sample_indices,
            n_decoding_threads=4,
            n_encoding_threads=4,
            max_inflight_tasks=32,
        )

    if replay_buffer["action"].shape[1:] != action_shape:
        raise RuntimeError(
            "Action shape mismatch. "
            f"Dataset has {replay_buffer['action'].shape[1:]}, "
            f"shape_meta expects {action_shape}. "
            "Old ARX5 data may have 6D action; new data should use 7D pose+gripper."
        )

    return replay_buffer


def _get_episode_sample_indices(dataset_path, target_frequency):
    if target_frequency is None:
        return None
    target_frequency = float(target_frequency)
    if target_frequency <= 0:
        raise ValueError(f"target_frequency must be positive, got {target_frequency}")

    replay_path = os.path.join(dataset_path, "replay_buffer.zarr")
    replay_buffer = ReplayBuffer.create_from_path(replay_path, mode="r")
    timestamps = np.asarray(replay_buffer["timestamp"][:], dtype=np.float64)
    starts = replay_buffer.episode_ends[:] - replay_buffer.episode_lengths[:]
    target_dt = 1.0 / target_frequency
    result = []
    old_steps = 0
    new_steps = 0
    for start, end in zip(starts, replay_buffer.episode_ends[:]):
        ep_timestamps = timestamps[start:end]
        if len(ep_timestamps) == 0:
            result.append(np.array([], dtype=np.int64))
            continue
        if np.any(np.diff(ep_timestamps) < 0):
            raise RuntimeError(
                "Episode timestamps are not monotonic. "
                f"dataset_path={dataset_path}, episode_start={start}, episode_end={end}"
            )
        old_steps += len(ep_timestamps)
        if len(ep_timestamps) <= 1:
            idxs = np.arange(len(ep_timestamps), dtype=np.int64)
        else:
            grid = np.arange(ep_timestamps[0], ep_timestamps[-1] + 1e-9, target_dt)
            idxs = np.searchsorted(ep_timestamps, grid, side="right") - 1
            idxs = np.clip(idxs, 0, len(ep_timestamps) - 1)
            idxs = np.unique(idxs).astype(np.int64)
        result.append(idxs)
        new_steps += len(idxs)
    print(
        "ARX5 preselected training samples:",
        f"{old_steps} -> {new_steps} steps",
        f"target_frequency={target_frequency:g}Hz",
    )
    return result


def _count_decodable_training_frames(video_path, dt, max_needed):
    count = 0
    for _ in read_video(
        video_path=str(video_path),
        dt=dt,
        thread_type="FRAME",
        thread_count=1,
    ):
        count += 1
        if count >= max_needed:
            break
    return count


def _trim_sample_indices_to_available_video(
    dataset_path,
    image_keys,
    episode_sample_indices,
):
    if episode_sample_indices is None or len(image_keys) == 0:
        return episode_sample_indices

    camera_idxs = sorted(int(key.split("_")[-1]) for key in image_keys)
    replay_path = os.path.join(dataset_path, "replay_buffer.zarr")
    video_dir = os.path.join(dataset_path, "videos")
    replay_buffer = ReplayBuffer.create_from_path(replay_path, mode="r")
    timestamps = np.asarray(replay_buffer["timestamp"][:], dtype=np.float64)
    starts = replay_buffer.episode_ends[:] - replay_buffer.episode_lengths[:]

    trimmed = []
    old_steps = 0
    new_steps = 0
    trimmed_episodes = []
    dropped_episodes = []

    for episode_idx, idxs in enumerate(episode_sample_indices):
        idxs = np.asarray(idxs, dtype=np.int64)
        old_steps += len(idxs)
        if len(idxs) == 0:
            trimmed.append(idxs)
            continue

        start = starts[episode_idx]
        length = replay_buffer.episode_lengths[episode_idx]
        selected_timestamps = timestamps[start : start + length][idxs]
        if len(selected_timestamps) > 1:
            dt = float(np.median(np.diff(selected_timestamps)))
        elif len(timestamps) > 1:
            dt = float(np.median(np.diff(timestamps)))
        else:
            dt = 1.0 / 20.0

        frame_counts = []
        for camera_idx in camera_idxs:
            video_path = os.path.join(video_dir, str(episode_idx), f"{camera_idx}.mp4")
            if not os.path.exists(video_path):
                raise RuntimeError(
                    f"Missing video {video_path} for episode {episode_idx}."
                )
            frame_counts.append(
                _count_decodable_training_frames(
                    video_path=video_path,
                    dt=dt,
                    max_needed=len(idxs),
                )
            )

        keep_steps = min(len(idxs), min(frame_counts))
        if keep_steps < len(idxs):
            keep_steps = max(0, keep_steps - VIDEO_ALIGNMENT_SAFETY_FRAMES)
        if keep_steps <= 0:
            dropped_episodes.append(episode_idx)
            idxs = idxs[:0]
        elif keep_steps < len(idxs):
            trimmed_episodes.append(
                (
                    episode_idx,
                    len(idxs),
                    keep_steps,
                    frame_counts,
                )
            )
            idxs = idxs[:keep_steps]

        trimmed.append(idxs)
        new_steps += len(idxs)

    if trimmed_episodes or dropped_episodes:
        print(
            "ARX5 video alignment trim:",
            f"{old_steps} -> {new_steps} steps",
            f"trimmed_episodes={len(trimmed_episodes)}",
            f"dropped_episodes={len(dropped_episodes)}",
        )
        for episode_idx, old_len, new_len, frame_counts in trimmed_episodes:
            print(
                f"  episode {episode_idx}: {old_len} -> {new_len} "
                f"min_video_frames={min(frame_counts)} camera_frames={frame_counts}"
            )
        if dropped_episodes:
            print("  dropped episodes:", dropped_episodes)

    return trimmed


def _copy_selected_steps(replay_buffer, episode_slices):
    out = ReplayBuffer.create_empty_numpy()
    keys = list(replay_buffer.keys())
    for start, end, indices in episode_slices:
        if len(indices) == 0:
            continue
        episode = {}
        for key in keys:
            episode[key] = replay_buffer[key][start:end][indices]
        out.add_episode(episode)
    return out


def _resample_replay_buffer_by_time(replay_buffer, target_frequency):
    if target_frequency is None:
        return replay_buffer
    target_frequency = float(target_frequency)
    if target_frequency <= 0:
        raise ValueError(f"target_frequency must be positive, got {target_frequency}")

    episode_slices = []
    starts = replay_buffer.episode_ends[:] - replay_buffer.episode_lengths[:]
    target_dt = 1.0 / target_frequency
    for start, end in zip(starts, replay_buffer.episode_ends[:]):
        timestamps = np.asarray(replay_buffer["timestamp"][start:end], dtype=np.float64)
        if len(timestamps) <= 1:
            episode_slices.append((start, end, np.arange(len(timestamps))))
            continue
        grid = np.arange(timestamps[0], timestamps[-1] + 1e-9, target_dt)
        indices = np.searchsorted(timestamps, grid, side="right") - 1
        indices = np.clip(indices, 0, len(timestamps) - 1)
        indices = np.unique(indices)
        episode_slices.append((start, end, indices))

    out = _copy_selected_steps(replay_buffer, episode_slices)
    print(
        "ARX5 dataset resampled:",
        f"{replay_buffer.n_steps} -> {out.n_steps} steps",
        f"target_frequency={target_frequency:g}Hz",
    )
    return out


def _trim_static_replay_buffer(
    replay_buffer,
    static_pos_threshold,
    static_rot_threshold,
    static_gripper_threshold,
    static_pad_before,
    static_pad_after,
    min_episode_steps,
):
    episode_slices = []
    starts = replay_buffer.episode_ends[:] - replay_buffer.episode_lengths[:]
    dropped = 0
    for start, end in zip(starts, replay_buffer.episode_ends[:]):
        action = np.asarray(replay_buffer["action"][start:end], dtype=np.float64)
        n = len(action)
        if n <= min_episode_steps or n < 2:
            episode_slices.append((start, end, np.arange(n)))
            continue

        pos_delta = np.linalg.norm(np.diff(action[:, :3], axis=0), axis=1)
        rot_delta = np.linalg.norm(np.diff(action[:, 3:6], axis=0), axis=1)
        gripper_delta = np.abs(np.diff(action[:, 6], axis=0))
        active = (
            (pos_delta > static_pos_threshold)
            | (rot_delta > static_rot_threshold)
            | (gripper_delta > static_gripper_threshold)
        )
        active_idxs = np.flatnonzero(active)
        if len(active_idxs) == 0:
            dropped += 1
            continue
        trim_start = max(0, int(active_idxs[0]) - int(static_pad_before))
        trim_end = min(n, int(active_idxs[-1]) + 2 + int(static_pad_after))
        if trim_end - trim_start < min_episode_steps:
            dropped += 1
            continue
        episode_slices.append((start, end, np.arange(trim_start, trim_end)))

    out = _copy_selected_steps(replay_buffer, episode_slices)
    print(
        "ARX5 dataset static trim:",
        f"{replay_buffer.n_steps} -> {out.n_steps} steps",
        f"dropped_episodes={dropped}",
    )
    return out


def _apply_delta_action(replay_buffer):
    actions = replay_buffer["action"][:].copy()
    episode_ends = replay_buffer.episode_ends[:]
    starts = episode_ends - replay_buffer.episode_lengths[:]
    for start, end in zip(starts, episode_ends):
        if end - start <= 1:
            continue
        actions[start + 1 : end, :6] = np.diff(actions[start:end, :6], axis=0)
        actions[start, :6] = 0
    replay_buffer["action"][:] = actions
    return replay_buffer


def _process_replay_buffer(
    replay_buffer,
    target_frequency,
    delta_action,
    trim_static_start_end,
    static_pos_threshold,
    static_rot_threshold,
    static_gripper_threshold,
    static_pad_before,
    static_pad_after,
    min_episode_steps,
):
    replay_buffer = _resample_replay_buffer_by_time(
        replay_buffer=replay_buffer,
        target_frequency=target_frequency,
    )
    if trim_static_start_end:
        replay_buffer = _trim_static_replay_buffer(
            replay_buffer=replay_buffer,
            static_pos_threshold=static_pos_threshold,
            static_rot_threshold=static_rot_threshold,
            static_gripper_threshold=static_gripper_threshold,
            static_pad_before=static_pad_before,
            static_pad_after=static_pad_after,
            min_episode_steps=min_episode_steps,
        )
    if delta_action:
        replay_buffer = _apply_delta_action(replay_buffer)
    return replay_buffer
