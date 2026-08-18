import argparse
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DP_ROOT = REPO_ROOT / "diffusion_policy-main"
for path in (REPO_ROOT, DP_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from arx5_ckpt_loader.policy_loader import load_policy_from_ckpt  # noqa: E402


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _to_batch(item, device):
    obs = {
        key: value.unsqueeze(0).to(device)
        for key, value in item["obs"].items()
    }
    obs["prev_action"] = item["prev_action"].unsqueeze(0).to(device)
    obs["prev_action_mask"] = item["prev_action_mask"].unsqueeze(0).to(device)
    return obs


def _predict(policy, obs, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        result = policy.predict_action(obs)
    return result["action"][0].detach().cpu().numpy()


def _valid_condition(item):
    prev = item["prev_action"].detach().cpu().numpy()
    mask = item["prev_action_mask"].detach().cpu().numpy() > 0.5
    if not np.any(mask):
        return None
    valid_idx = np.nonzero(mask)[0]
    return prev[valid_idx], valid_idx


def build_dataset(args):
    config_dir = Path(args.config_dir).expanduser()
    if not config_dir.is_absolute():
        config_dir = REPO_ROOT / config_dir
    cfg_dir = str(config_dir.resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name=args.config_name,
            overrides=[
                f"task.dataset_path={args.dataset_path}",
                f"prev_cond_steps={args.prev_cond_steps}",
                f"task.dataset.target_frequency={args.target_frequency}",
                "task.dataset.use_cache=False" if args.no_cache else "task.dataset.use_cache=True",
                f"task.dataset.val_ratio={args.val_ratio}",
            ],
        )
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset


def choose_indices(dataset, args):
    if args.idx:
        return [int(x) for x in args.idx]
    rng = np.random.default_rng(args.seed)
    candidates = np.arange(len(dataset), dtype=np.int64)
    if len(candidates) == 0:
        return []
    n = min(args.num_samples, len(candidates))
    return sorted(rng.choice(candidates, size=n, replace=False).tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Compare CFG DP predictions with real prev_action versus empty prev_action."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--config-dir", default="arx5_dp_cfg")
    parser.add_argument("--config-name", default="train_diffusion_unet_arx5_hybrid_workspace_cfg")
    parser.add_argument("--prev-cond-steps", type=int, default=None, help="Defaults to ckpt policy.prev_cond_steps.")
    parser.add_argument("--target-frequency", type=float, default=20.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--cfg-guidance-weight", type=float, default=None)
    parser.add_argument("--idx", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--include-empty-prev", action="store_true")
    args = parser.parse_args()

    _, policy, device, ckpt_path = load_policy_from_ckpt(args.ckpt, device=args.device)
    if not hasattr(policy, "prev_cond_steps"):
        raise RuntimeError(f"Checkpoint is not a CFG prev_action policy: {ckpt_path}")
    ckpt_prev_steps = int(policy.prev_cond_steps)
    if args.prev_cond_steps is None:
        args.prev_cond_steps = ckpt_prev_steps
    elif int(args.prev_cond_steps) != ckpt_prev_steps:
        raise RuntimeError(
            f"--prev-cond-steps={args.prev_cond_steps} does not match checkpoint "
            f"prev_cond_steps={ckpt_prev_steps}. Retrain for a different value."
        )
    policy.eval()
    if args.cfg_guidance_weight is not None:
        policy.cfg_guidance_weight = float(args.cfg_guidance_weight)

    dataset = build_dataset(args)
    indices = choose_indices(dataset, args)
    print(f"ckpt={ckpt_path}")
    print(f"dataset={args.dataset_path} len={len(dataset)} sampled={len(indices)}")
    print(
        f"prev_cond_steps={args.prev_cond_steps}",
        f"target_frequency={args.target_frequency:g}Hz",
        f"device={device}",
        f"cfg_guidance_weight={policy.cfg_guidance_weight:g}",
    )

    rows = []
    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            print(f"skip idx={idx}: out of range")
            continue
        item = dataset[idx]
        condition = _valid_condition(item)
        if condition is None and not args.include_empty_prev:
            continue

        real_obs = _to_batch(item, device)
        empty_obs = {
            key: value.clone()
            for key, value in real_obs.items()
        }
        empty_obs["prev_action"].zero_()
        empty_obs["prev_action_mask"].zero_()

        real_action = _predict(policy, real_obs, seed=args.seed)
        empty_action = _predict(policy, empty_obs, seed=args.seed)
        gt_action = item["action"].detach().cpu().numpy()

        action_diff = float(np.linalg.norm(real_action - empty_action))
        real_gt = float(np.linalg.norm(real_action[0, :6] - gt_action[0, :6]))
        empty_gt = float(np.linalg.norm(empty_action[0, :6] - gt_action[0, :6]))
        if condition is None:
            real_consistency = np.nan
            empty_consistency = np.nan
        else:
            cond_action, valid_idx = condition
            n = min(len(valid_idx), len(real_action), len(empty_action))
            real_consistency = float(np.mean(np.linalg.norm(real_action[:n, :6] - cond_action[:n, :6], axis=-1)))
            empty_consistency = float(np.mean(np.linalg.norm(empty_action[:n, :6] - cond_action[:n, :6], axis=-1)))
        rows.append((idx, action_diff, real_consistency, empty_consistency, real_gt, empty_gt))

        print(
            f"idx={idx:6d}",
            f"diff={action_diff:.5f}",
            f"condition real/empty={real_consistency:.5f}/{empty_consistency:.5f}",
            f"gt real/empty={real_gt:.5f}/{empty_gt:.5f}",
        )

    if not rows:
        print("No valid samples inspected.")
        return

    arr = np.asarray(rows, dtype=np.float64)
    consistency_improve = arr[:, 3] - arr[:, 2]
    gt_improve = arr[:, 5] - arr[:, 4]
    print("=" * 88)
    print(f"valid_samples={len(rows)}")
    print(f"prediction_diff_mean={arr[:, 1].mean():.6f}")
    print(
        "condition_consistency_improvement(empty-real):",
        f"mean={np.nanmean(consistency_improve):.6f}",
        f"median={np.nanmedian(consistency_improve):.6f}",
        f"positive_ratio={np.nanmean(consistency_improve > 0):.2%}",
    )
    print(
        "gt_first_action_improvement(empty-real):",
        f"mean={np.nanmean(gt_improve):.6f}",
        f"median={np.nanmedian(gt_improve):.6f}",
        f"positive_ratio={np.nanmean(gt_improve > 0):.2%}",
    )


if __name__ == "__main__":
    main()
