from pathlib import Path

import dill
import hydra
import torch

from diffusion_policy.workspace.base_workspace import BaseWorkspace


DEFAULT_CKPT = (
    "data/outputs/2026.07.08/"
    "18.15.27_train_diffusion_unet_arx5_hybrid_arx5_image/"
    "checkpoints/latest.ckpt"
)


def resolve_ckpt_path(ckpt_path) -> Path:
    path = Path(ckpt_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd().joinpath(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_policy_from_ckpt(ckpt_path, device: str = "auto", inference_steps: int = 16):
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    ckpt_path = resolve_ckpt_path(ckpt_path)
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if bool(cfg.training.get("use_ema", False)):
        policy = workspace.ema_model

    policy.eval().to(device)
    if hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = int(inference_steps)

    return cfg, policy, device, ckpt_path


def print_policy_summary(cfg, policy, device, ckpt_path):
    shape_meta = cfg.task.shape_meta

    print(f"ckpt: {ckpt_path}")
    print(f"device: {device}")
    print(f"workspace: {cfg._target_}")
    print(f"task: {cfg.task.name}")
    print(f"policy: {policy.__class__.__module__}.{policy.__class__.__name__}")
    print(f"horizon: {cfg.horizon}")
    print(f"n_obs_steps: {cfg.n_obs_steps}")
    print(f"n_action_steps: {cfg.n_action_steps}")
    print("obs:")
    for key, meta in shape_meta.obs.items():
        obs_type = meta.get("type", "low_dim")
        print(f"  {key}: type={obs_type}, shape={list(meta.shape)}")
    print(f"action: shape={list(shape_meta.action.shape)}")

