from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Dict, List, Sequence, Tuple

import cv2
from filelock import FileLock
import numcodecs
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import zarr

from arx5_act.paths import ensure_project_paths

ensure_project_paths()
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.arx5_image_dataset import (
    _copy_selected_steps,
    _get_episode_sample_indices,
    _trim_sample_indices_to_available_video,
)

VIDEO_ALIGNMENT_SAFETY_FRAMES = 2


@dataclass
class EpisodeIndex:
    episode: int
    start: int
    end: int


def _make_qpos(replay: ReplayBuffer, state_mode: str = "eef") -> np.ndarray:
    if state_mode == "eef":
        pos = np.asarray(replay["robot0_eef_pos"][:], dtype=np.float32)
        rot = np.asarray(replay["robot0_eef_rot_axis_angle"][:], dtype=np.float32)
        grip = np.asarray(replay["robot0_gripper_width"][:], dtype=np.float32)
        return np.concatenate([pos, rot, grip], axis=-1)
    if state_mode == "joint":
        joint = np.asarray(replay["robot_joint"][:], dtype=np.float32)
        grip = np.asarray(replay["robot_gripper"][:], dtype=np.float32)
        if grip.ndim == 1:
            grip = grip[:, None]
        return np.concatenate([joint, grip[:, :1]], axis=-1)
    raise ValueError(f"Unsupported state_mode: {state_mode}")


def _make_action(replay: ReplayBuffer, state_mode: str = "eef") -> np.ndarray:
    if state_mode == "eef":
        return np.asarray(replay["action"][:], dtype=np.float32)
    if state_mode == "joint":
        return _make_qpos(replay, state_mode="joint")
    raise ValueError(f"Unsupported state_mode: {state_mode}")


def _load_replay_buffer(dataset_path: str, target_frequency=None, camera_names=None):
    replay = ReplayBuffer.create_from_path(
        str(Path(dataset_path).joinpath("replay_buffer.zarr").absolute()),
        mode="r",
    )
    if target_frequency is None:
        return replay

    image_keys = list(camera_names or ("camera_0", "camera_1", "camera_2"))
    episode_sample_indices = _get_episode_sample_indices(
        dataset_path=dataset_path,
        target_frequency=target_frequency,
    )
    episode_sample_indices = _trim_sample_indices_to_available_video(
        dataset_path=dataset_path,
        image_keys=image_keys,
        episode_sample_indices=episode_sample_indices,
    )
    starts = replay.episode_ends[:] - replay.episode_lengths[:]
    episode_slices = []
    for start, end, indices in zip(starts, replay.episode_ends[:], episode_sample_indices):
        episode_slices.append((int(start), int(end), np.asarray(indices, dtype=np.int64)))
    return _copy_selected_steps(replay, episode_slices)


def compute_norm_stats(
    dataset_path: str,
    target_frequency=None,
    camera_names=None,
    state_mode: str = "eef",
) -> Dict[str, np.ndarray]:
    replay = _load_replay_buffer(
        dataset_path=dataset_path,
        target_frequency=target_frequency,
        camera_names=camera_names,
    )
    qpos = _make_qpos(replay, state_mode=state_mode)
    action = _make_action(replay, state_mode=state_mode)
    return {
        "qpos_mean": qpos.mean(axis=0),
        "qpos_std": np.clip(qpos.std(axis=0), 1e-2, np.inf),
        "action_mean": action.mean(axis=0),
        "action_std": np.clip(action.std(axis=0), 1e-2, np.inf),
        "example_qpos": qpos[0],
    }


class Arx5ActDataset(Dataset):
    """ACT dataset adapter for ARX5 Diffusion Policy collection folders.

    Expected folder:
      replay_buffer.zarr/
      videos/<episode_id>/0.mp4, 1.mp4, 2.mp4

    The ACT qpos/action convention used here is 7D:
      [x, y, z, rx, ry, rz, gripper_width]
    """

    def __init__(
        self,
        dataset_path: str,
        norm_stats: Dict[str, np.ndarray],
        camera_names: Sequence[str] = ("camera_0", "camera_1", "camera_2"),
        chunk_size: int = 50,
        image_size: Tuple[int, int] = (320, 240),
        episode_indices: Sequence[int] = None,
        use_cache: bool = True,
        target_frequency=None,
        state_mode: str = "eef",
        cache_dir: str = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.video_dir = self.dataset_path / "videos"
        self.camera_names = list(camera_names)
        self.chunk_size = int(chunk_size)
        self.image_size = tuple(image_size)
        self.norm_stats = norm_stats
        self.use_cache = bool(use_cache)
        if state_mode not in ("eef", "joint"):
            raise ValueError(f"Unsupported state_mode: {state_mode}")
        self.state_mode = state_mode
        self.cache_dir = None if cache_dir is None else Path(cache_dir)

        self.target_frequency = target_frequency
        self.replay = _load_replay_buffer(
            dataset_path=str(self.dataset_path),
            target_frequency=target_frequency,
            camera_names=self.camera_names,
        )
        self.qpos = _make_qpos(self.replay, state_mode=self.state_mode)
        self.action = _make_action(self.replay, state_mode=self.state_mode)
        self.timestamps = np.asarray(self.replay["timestamp"][:], dtype=np.float64)
        self.episode_ends = self.replay.episode_ends[:]
        self.episode_lengths = self.replay.episode_lengths[:]
        self.episode_starts = self.episode_ends - self.episode_lengths

        if episode_indices is None:
            episode_indices = range(self.replay.n_episodes)
        self.episode_indices = [int(x) for x in episode_indices]
        self._video_caps = {}
        self._video_fps = {}
        self._video_frame_counts = {}

        episode_valid_ends = self._compute_episode_valid_ends()
        self.episode_valid_ends = episode_valid_ends
        self._cache_images = None
        if self.use_cache:
            self._cache_images = self._load_or_create_cache(episode_valid_ends)

        self.indices: List[EpisodeIndex] = []
        for episode in self.episode_indices:
            start = int(self.episode_starts[episode])
            end = int(self.episode_ends[episode])
            valid_end = min(end, episode_valid_ends[episode])
            for step in range(start, valid_end):
                self.indices.append(EpisodeIndex(episode=episode, start=step, end=end))

    def _dataset_signature(self) -> Dict:
        videos = []
        for episode in range(self.replay.n_episodes):
            for camera_name in self.camera_names:
                camera_idx = int(camera_name.split("_")[-1])
                video_path = self.video_dir / str(episode) / f"{camera_idx}.mp4"
                stat = video_path.stat()
                videos.append(
                    {
                        "episode": int(episode),
                        "camera": int(camera_idx),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
        return {
            "n_steps": int(self.qpos.shape[0]),
            "n_episodes": int(self.replay.n_episodes),
            "episode_ends": [int(x) for x in self.episode_ends],
            "videos": videos,
        }

    def _cache_path(self) -> Path:
        config = {
            "camera_names": self.camera_names,
            "image_size": self.image_size,
            "target_frequency": self.target_frequency,
            "state_mode": self.state_mode,
            "safety_frames": VIDEO_ALIGNMENT_SAFETY_FRAMES,
            "dataset": self._dataset_signature(),
            "version": 3,
        }
        digest = hashlib.md5(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
        cache_base = self.dataset_path if self.cache_dir is None else self.cache_dir
        cache_base.mkdir(parents=True, exist_ok=True)
        return cache_base / f"act_{self.state_mode}_cache_{digest}.zarr"

    def _cache_is_complete(self, cache_path: Path) -> bool:
        try:
            root = zarr.open(str(cache_path), mode="r")
            if not bool(root.attrs.get("complete", False)):
                return False
            images = root["images"]
            n_cameras = len(self.camera_names)
            width, height = self.image_size
            expected_shape = (int(self.qpos.shape[0]), n_cameras, height, width, 3)
            return tuple(images.shape) == expected_shape
        except Exception:
            return False

    def _load_or_create_cache(self, episode_valid_ends: Dict[int, int]):
        cache_path = self._cache_path()
        lock_path = str(cache_path) + ".lock"
        print(f"Acquiring lock on ACT image cache: {cache_path}")
        with FileLock(lock_path):
            if cache_path.is_dir() and not self._cache_is_complete(cache_path):
                print("Found incomplete ACT image cache. Rebuilding.")
                shutil.rmtree(cache_path)
            if not cache_path.is_dir():
                try:
                    self._create_image_cache(cache_path, episode_valid_ends)
                except Exception as exc:
                    if cache_path.exists():
                        shutil.rmtree(cache_path)
                    raise exc
            else:
                print("Loading cached ACT images from disk.")
            root = zarr.open(str(cache_path), mode="r")
            return root["images"]

    def _create_image_cache(self, cache_path: Path, episode_valid_ends: Dict[int, int]):
        print("ACT image cache does not exist. Creating.")
        n_steps = int(self.qpos.shape[0])
        n_cameras = len(self.camera_names)
        width, height = self.image_size
        compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
        root = zarr.open(str(cache_path), mode="w")
        images = root.create_dataset(
            "images",
            shape=(n_steps, n_cameras, height, width, 3),
            chunks=(1, n_cameras, height, width, 3),
            dtype="uint8",
            compressor=compressor,
        )
        root.create_dataset(
            "episode_valid_ends",
            data=np.asarray(
                [episode_valid_ends[int(i)] for i in range(self.replay.n_episodes)],
                dtype=np.int64,
            ),
            dtype=np.int64,
        )
        root.attrs["camera_names"] = list(self.camera_names)
        root.attrs["image_size"] = list(self.image_size)
        root.attrs["state_mode"] = self.state_mode
        root.attrs["safety_frames"] = int(VIDEO_ALIGNMENT_SAFETY_FRAMES)
        root.attrs["complete"] = False

        for episode in tqdm(range(self.replay.n_episodes), desc="Building ACT image cache"):
            episode = int(episode)
            start = int(self.episode_starts[episode])
            valid_end = int(episode_valid_ends[episode])
            if valid_end <= start:
                continue
            episode_timestamps = self.timestamps[start:valid_end]
            t0 = float(self.timestamps[start])
            for camera_pos, camera_name in enumerate(self.camera_names):
                camera_idx = int(camera_name.split("_")[-1])
                video_path = self.video_dir / str(episode) / f"{camera_idx}.mp4"
                fps, frame_count = self._get_video_info(episode, camera_idx)
                frame_indices = np.rint((episode_timestamps - t0) * fps).astype(np.int64)
                frame_indices = np.clip(frame_indices, 0, frame_count - 1)

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video: {video_path}")
                try:
                    last_frame_idx = None
                    last_frame = None
                    for offset, frame_idx in enumerate(frame_indices):
                        frame_idx = int(frame_idx)
                        if last_frame_idx != frame_idx:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ok, frame = cap.read()
                            if not ok:
                                raise RuntimeError(
                                    f"Failed to read frame {frame_idx} from {video_path}"
                                )
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
                            last_frame = frame
                            last_frame_idx = frame_idx
                        images[start + offset, camera_pos] = last_frame
                finally:
                    cap.release()
        root.attrs["complete"] = True
        print(f"Saved ACT image cache: {cache_path}")

    def _get_video_info(self, episode: int, camera_idx: int) -> Tuple[float, int]:
        key = (int(episode), int(camera_idx))
        if key in self._video_fps and key in self._video_frame_counts:
            return self._video_fps[key], self._video_frame_counts[key]

        video_path = self.video_dir / str(episode) / f"{camera_idx}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")
        self._video_fps[key] = fps
        self._video_frame_counts[key] = frame_count
        return fps, frame_count

    def _compute_episode_valid_ends(self) -> Dict[int, int]:
        valid_ends = {}
        camera_idxs = [int(name.split("_")[-1]) for name in self.camera_names]
        trimmed = []
        for episode in range(self.replay.n_episodes):
            start = int(self.episode_starts[episode])
            end = int(self.episode_ends[episode])
            t0 = float(self.timestamps[start])
            valid_end = end
            for camera_idx in camera_idxs:
                fps, frame_count = self._get_video_info(episode, camera_idx)
                max_frame_idx = max(0, frame_count - 1 - VIDEO_ALIGNMENT_SAFETY_FRAMES)
                max_timestamp = t0 + (max_frame_idx + 0.5) / fps
                camera_valid_end = int(
                    np.searchsorted(
                        self.timestamps[start:end],
                        max_timestamp,
                        side="right",
                    )
                ) + start
                valid_end = min(valid_end, camera_valid_end)
            valid_end = max(start, valid_end)
            valid_ends[int(episode)] = valid_end
            if valid_end < end:
                trimmed.append((int(episode), end - start, valid_end - start))
        if trimmed:
            old_steps = sum(old for _, old, _ in trimmed)
            new_steps = sum(new for _, _, new in trimmed)
            print(
                "ACT video alignment trim:",
                f"{old_steps} -> {new_steps} steps across {len(trimmed)} episodes",
            )
            for episode, old_len, new_len in trimmed:
                print(f"  episode {episode}: {old_len} -> {new_len}")
        return valid_ends

    def __len__(self):
        return len(self.indices)

    def _read_camera_frame(self, episode: int, camera_name: str, timestamp: float) -> np.ndarray:
        camera_idx = int(camera_name.split("_")[-1])
        video_path = self.video_dir / str(episode) / f"{camera_idx}.mp4"
        cap_key = (int(episode), int(camera_idx))
        cap = self._video_caps.get(cap_key)
        if cap is None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            self._video_caps[cap_key] = cap
        fps = self._video_fps.get(cap_key)
        if fps is None:
            fps, _ = self._get_video_info(episode, camera_idx)
            self._video_fps[cap_key] = fps
        episode_start = int(self.episode_starts[episode])
        t0 = float(self.timestamps[episode_start])
        frame_idx = max(0, int(round((float(timestamp) - t0) * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
        return frame

    def __getitem__(self, index: int):
        item = self.indices[index]
        timestamp = float(self.timestamps[item.start])

        if self._cache_images is not None:
            image_data = self._cache_images[item.start]
        else:
            images = [
                self._read_camera_frame(item.episode, camera_name, timestamp)
                for camera_name in self.camera_names
            ]
            image_data = np.stack(images, axis=0)
        image_data = torch.from_numpy(image_data).float()
        image_data = torch.einsum("k h w c -> k c h w", image_data) / 255.0

        qpos = self.qpos[item.start].copy()
        actions = np.zeros((self.chunk_size, self.action.shape[-1]), dtype=np.float32)
        is_pad = np.ones((self.chunk_size,), dtype=bool)
        available = min(self.chunk_size, item.end - item.start)
        actions[:available] = self.action[item.start : item.start + available]
        is_pad[:available] = False

        qpos = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        actions = (actions - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        return (
            image_data,
            torch.from_numpy(qpos).float(),
            torch.from_numpy(actions).float(),
            torch.from_numpy(is_pad).bool(),
        )

    def close(self):
        for cap in self._video_caps.values():
            cap.release()
        self._video_caps = {}
        self._video_fps = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_episode_split(n_episodes: int, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_episodes)
    rng.shuffle(indices)
    if val_ratio <= 0 or n_episodes <= 1:
        n_val = 0
    else:
        n_val = max(1, int(round(n_episodes * val_ratio)))
        n_val = min(n_val, n_episodes - 1)
    val = indices[:n_val]
    train = indices[n_val:]
    if len(train) == 0:
        train = val
    return train.tolist(), val.tolist()
