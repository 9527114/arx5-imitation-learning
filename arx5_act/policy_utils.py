import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from arx5_act.paths import ensure_project_paths

ensure_project_paths()
from policy import ACTPolicy


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_policy_config(args) -> Dict:
    state_dim = 7
    return {
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "backbone": args.backbone,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "nheads": args.nheads,
        "camera_names": list(args.camera_names),
        "state_dim": state_dim,
        "pretrained_backbone": args.pretrained_backbone,
    }


def build_policy(policy_config: Dict, device: torch.device) -> ACTPolicy:
    # ACT's original builder parses sys.argv internally. Provide the required
    # dummy CLI args here so arx5_act entrypoints can own their own arguments.
    old_argv = sys.argv
    try:
        sys.argv = [
            old_argv[0],
            "--ckpt_dir",
            ".",
            "--policy_class",
            "ACT",
            "--task_name",
            "arx5",
            "--seed",
            "0",
            "--num_epochs",
            "1",
        ]
        policy = ACTPolicy(policy_config)
    finally:
        sys.argv = old_argv
    policy.to(device)
    return policy


def save_training_bundle(
    ckpt_dir: Path,
    dataset_path: str,
    policy_config: Dict,
    camera_names: Sequence[str],
    chunk_size: int,
    target_frequency,
    state_mode: str,
    train_episodes,
    val_episodes,
):
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(
            {
                "dataset_path": dataset_path,
                "policy_config": policy_config,
                "camera_names": list(camera_names),
                "chunk_size": int(chunk_size),
                "target_frequency": None if target_frequency is None else float(target_frequency),
                "state_mode": state_mode,
                "state_dim": 7,
                "action_dim": 7,
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
            },
            f,
            indent=2,
        )


def load_bundle(ckpt_dir: str, ckpt_name: str, device: torch.device):
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir / "config.json"
    stats_path = ckpt_dir / "dataset_stats.pkl"
    ckpt_path = ckpt_dir / ckpt_name
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing ACT config: {config_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing ACT dataset stats: {stats_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing ACT checkpoint: {ckpt_path}")

    with open(config_path, "r") as f:
        config = json.load(f)
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    policy = build_policy(config["policy_config"], device=device)
    state_dict = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return config, stats, policy, ckpt_path


def qpos_from_robot_state(robot_state: Dict[str, np.ndarray], state_mode: str = "eef") -> np.ndarray:
    gripper = np.asarray(robot_state["gripper_pos"], dtype=np.float32)
    if state_mode == "eef":
        pose = np.asarray(robot_state["ActualTCPPose"], dtype=np.float32)
        return np.concatenate([pose[:6], gripper[:1]], axis=0)
    if state_mode == "joint":
        joint = np.asarray(robot_state["ActualQ"], dtype=np.float32)
        return np.concatenate([joint[:6], gripper[:1]], axis=0)
    raise ValueError(f"Unsupported state_mode: {state_mode}")


def normalize_qpos(qpos: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (qpos - stats["qpos_mean"]) / stats["qpos_std"]


def unnormalize_action(action: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return action * stats["action_std"] + stats["action_mean"]


def make_image_tensor(camera_frames: Dict[str, np.ndarray], camera_names: Sequence[str], device: torch.device):
    import cv2

    images = []
    for name in camera_names:
        frame = camera_frames[name]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        images.append(frame)
    image_np = np.stack(images, axis=0)
    image = torch.from_numpy(image_np).float()
    image = torch.einsum("k h w c -> k c h w", image) / 255.0
    return image.unsqueeze(0).to(device)


def predict_action_chunk(policy, image, qpos, stats: Dict[str, np.ndarray]) -> np.ndarray:
    qpos_np = normalize_qpos(np.asarray(qpos, dtype=np.float32), stats)
    qpos_tensor = torch.from_numpy(qpos_np).float().unsqueeze(0).to(image.device)
    with torch.inference_mode():
        action = policy(qpos_tensor, image)
    action_np = action[0].detach().cpu().numpy()
    return unnormalize_action(action_np, stats)
