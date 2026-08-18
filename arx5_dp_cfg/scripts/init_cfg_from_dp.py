import argparse
import os
from pathlib import Path

for _key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_key, None)

import dill
import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from arx5_dp_cfg.train_diffusion_unet_hybrid_workspace_cfg import (
    TrainDiffusionUnetHybridWorkspace,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _resolve(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _load_payload(path):
    with Path(path).open("rb") as f:
        return torch.load(f, pickle_module=dill, map_location="cpu")


def _copy_expanded_cond_weight(dst, src):
    """Copy DP global-cond weights into CFG-expanded global-cond weights."""
    if dst.ndim != 2 or src.ndim != 2:
        return None
    if dst.shape[0] != src.shape[0] or dst.shape[1] < src.shape[1]:
        return None
    out = dst.clone()
    out[:, : src.shape[1]] = src
    # Keep the newly added prev_action columns close to neutral.
    out[:, src.shape[1] :] = 0
    return out


def merge_model_state(cfg_state, dp_state):
    merged = cfg_state.copy()
    copied_exact = 0
    copied_expanded = 0
    skipped = []
    for key, src in dp_state.items():
        if key not in merged:
            skipped.append((key, "missing_in_cfg"))
            continue
        dst = merged[key]
        if tuple(dst.shape) == tuple(src.shape):
            merged[key] = src.clone()
            copied_exact += 1
            continue
        expanded = _copy_expanded_cond_weight(dst, src)
        if expanded is not None:
            merged[key] = expanded
            copied_expanded += 1
            continue
        skipped.append((key, f"shape {tuple(src.shape)} -> {tuple(dst.shape)}"))
    return merged, copied_exact, copied_expanded, skipped


def build_cfg(args, dp_cfg):
    overrides = [
        f"task.dataset_path={args.dataset_path or dp_cfg.task.dataset_path}",
        f"prev_cond_steps={args.prev_cond_steps}",
        f"prev_chunk_dropout={args.prev_chunk_dropout}",
        f"prev_action_mode={args.prev_action_mode}",
        f"training.num_epochs={args.num_epochs}",
        f"logging.mode={args.logging_mode}",
        f"hydra.run.dir={args.output_dir}",
    ]
    if args.target_frequency is not None:
        overrides.append(f"task.dataset.target_frequency={args.target_frequency}")
    if args.batch_size is not None:
        overrides.append(f"dataloader.batch_size={args.batch_size}")
        overrides.append(f"val_dataloader.batch_size={args.batch_size}")
    if args.val_every is not None:
        overrides.append(f"training.val_every={args.val_every}")
    if args.max_val_steps is not None:
        overrides.append(f"training.max_val_steps={args.max_val_steps}")
    if args.checkpoint_every is not None:
        overrides.append(f"training.checkpoint_every={args.checkpoint_every}")

    config_dir = _resolve(args.config_dir)
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=args.config_name, overrides=overrides)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--config-dir", default="arx5_dp_cfg")
    parser.add_argument(
        "--config-name",
        default="train_diffusion_unet_arx5_hybrid_workspace_cfg",
    )
    parser.add_argument("--prev-cond-steps", type=int, default=4)
    parser.add_argument("--prev-chunk-dropout", type=float, default=0.3)
    parser.add_argument("--prev-action-mode", choices=["future", "past"], default="future")
    parser.add_argument("--target-frequency", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--logging-mode", default="offline")
    args = parser.parse_args()

    dp_ckpt = _resolve(args.dp_ckpt)
    output_dir = _resolve(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dp_payload = _load_payload(dp_ckpt)
    dp_cfg = dp_payload["cfg"]
    cfg = build_cfg(args, dp_cfg)

    dp_root = REPO_ROOT / "diffusion_policy-main"
    if dp_root.is_dir():
        os.chdir(dp_root)

    workspace = TrainDiffusionUnetHybridWorkspace(cfg, output_dir=str(output_dir))
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    workspace.model.set_normalizer(normalizer)
    if workspace.ema_model is not None:
        workspace.ema_model.set_normalizer(normalizer)

    cfg_state = workspace.model.state_dict()
    dp_state = dp_payload["state_dicts"]["model"]
    merged, copied_exact, copied_expanded, skipped = merge_model_state(cfg_state, dp_state)
    workspace.model.load_state_dict(merged, strict=True)

    if workspace.ema_model is not None:
        ema_src = dp_payload["state_dicts"].get("ema_model", dp_state)
        ema_state = workspace.ema_model.state_dict()
        ema_merged, ema_exact, ema_expanded, _ = merge_model_state(ema_state, ema_src)
        workspace.ema_model.load_state_dict(ema_merged, strict=True)
    else:
        ema_exact = 0
        ema_expanded = 0

    payload = {
        "cfg": cfg,
        "state_dicts": {
            "model": workspace.model.state_dict(),
            "optimizer": workspace.optimizer.state_dict(),
        },
        "pickles": {
            "global_step": dill.dumps(0),
            "epoch": dill.dumps(0),
            "_output_dir": dill.dumps(str(output_dir)),
        },
    }
    if workspace.ema_model is not None:
        payload["state_dicts"]["ema_model"] = workspace.ema_model.state_dict()

    latest_path = ckpt_dir / "latest.ckpt"
    torch.save(payload, latest_path.open("wb"), pickle_module=dill)

    print("CFG warm-start checkpoint created.")
    print(f"  dp_ckpt: {dp_ckpt}")
    print(f"  output_dir: {output_dir}")
    print(f"  latest: {latest_path}")
    print(f"  dataset_path: {cfg.task.dataset_path}")
    print(f"  prev_cond_steps: {cfg.prev_cond_steps}")
    print(f"  prev_chunk_dropout: {cfg.prev_chunk_dropout}")
    print(f"  prev_action_mode: {cfg.prev_action_mode}")
    print(f"  copied model exact: {copied_exact}")
    print(f"  copied model expanded condition weights: {copied_expanded}")
    print(f"  copied ema exact: {ema_exact}")
    print(f"  copied ema expanded condition weights: {ema_expanded}")
    print(f"  skipped dp keys: {len(skipped)}")
    for key, reason in skipped[:16]:
        print(f"    skip {key}: {reason}")


if __name__ == "__main__":
    main()
