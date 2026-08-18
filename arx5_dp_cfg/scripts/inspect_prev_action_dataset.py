import argparse
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fmt(arr, precision=4):
    return np.array2string(np.asarray(arr), precision=precision, suppress_small=True)


def inspect_item(dataset, normalizer, idx):
    item = dataset[idx]
    prev_action = _as_numpy(item["prev_action"])
    prev_mask = _as_numpy(item["prev_action_mask"])
    action = _as_numpy(item["action"])

    valid = prev_mask > 0.5
    n_valid = int(np.sum(valid))
    if n_valid > 0:
        first_prev = prev_action[np.nonzero(valid)[0][0]]
        last_prev = prev_action[np.nonzero(valid)[0][-1]]
        n_overlap = min(n_valid, len(action))
        cond_error = np.linalg.norm(action[:n_overlap, :6] - prev_action[valid][:n_overlap, :6], axis=-1)
        mean_cond_error = float(np.mean(cond_error))
        first_delta = action[0] - first_prev
    else:
        first_prev = None
        last_prev = None
        first_delta = None
        mean_cond_error = None

    nprev = normalizer["action"].normalize(torch.from_numpy(prev_action).float()).numpy()
    valid_nprev = nprev[valid] if n_valid > 0 else nprev[:0]

    buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = dataset.sampler.indices[idx]

    print("=" * 88)
    print(f"idx={idx}")
    print(
        "sampler:",
        f"buffer_start={buffer_start_idx}",
        f"buffer_end={buffer_end_idx}",
        f"sample_start={sample_start_idx}",
        f"sample_end={sample_end_idx}",
    )
    print(f"prev_action.shape={prev_action.shape} action.shape={action.shape}")
    print(f"prev_mask={_fmt(prev_mask, precision=1)} valid={n_valid}/{len(prev_mask)}")
    if n_valid > 0:
        print(f"prev_first_valid={_fmt(prev_action[np.nonzero(valid)[0][0]])}")
        print(f"prev_last_valid ={_fmt(last_prev)}")
        print(f"action_first     ={_fmt(action[0])}")
        print(
            "delta action_first-prev_first:",
            _fmt(first_delta),
            f"mean_condition_pose_error={mean_cond_error:.6f}",
        )
        print(
            "normalized valid prev:",
            f"min={float(valid_nprev.min()):.4f}",
            f"max={float(valid_nprev.max()):.4f}",
            f"mean={float(valid_nprev.mean()):.4f}",
            f"std={float(valid_nprev.std()):.4f}",
        )
    else:
        print("prev has no valid action; this is expected near episode start.")
    print(f"action_first_3={_fmt(action[:3])}")


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
                "task.dataset.use_cache=False" if args.no_cache else "task.dataset.use_cache=True",
            ],
        )
    if args.print_config:
        print(OmegaConf.to_yaml(cfg.task.dataset))
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    return dataset, normalizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--config-dir", default="arx5_dp_cfg")
    parser.add_argument(
        "--config-name",
        default="train_diffusion_unet_arx5_hybrid_workspace_cfg",
    )
    parser.add_argument("--prev-cond-steps", type=int, default=8)
    parser.add_argument("--idx", type=int, nargs="*", default=None)
    parser.add_argument("--num-auto", type=int, default=6)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset, normalizer = build_dataset(args)
    print(f"dataset_len={len(dataset)} prev_cond_steps={dataset.prev_cond_steps}")
    print(f"episodes={dataset.replay_buffer.n_episodes} steps={dataset.replay_buffer.n_steps}")

    if args.idx:
        indices = args.idx
    else:
        if len(dataset) == 0:
            raise RuntimeError("Dataset has no samples.")
        auto = np.linspace(0, len(dataset) - 1, num=min(args.num_auto, len(dataset)), dtype=int)
        indices = sorted(set(int(x) for x in auto))

    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            print(f"skip idx={idx}: out of range [0,{len(dataset)-1}]")
            continue
        inspect_item(dataset, normalizer, idx)


if __name__ == "__main__":
    main()
