import argparse

import torch

from arx5_ckpt_loader.obs_buffer import build_zero_obs
from arx5_ckpt_loader.policy_loader import (
    DEFAULT_CKPT,
    load_policy_from_ckpt,
    print_policy_summary,
)


def run_dummy_inference(cfg, policy, device: torch.device):
    obs_dict = build_zero_obs(cfg, device)
    with torch.no_grad():
        result = policy.predict_action(obs_dict)

    print("dummy predict_action output:")
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={list(value.shape)}, dtype={value.dtype}, device={value.device}")
        else:
            print(f"  {key}: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to a Diffusion Policy checkpoint.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one zero-observation policy.predict_action() call after loading.",
    )
    args = parser.parse_args()

    cfg, policy, device, ckpt_path = load_policy_from_ckpt(
        ckpt_path=args.ckpt,
        device=args.device,
    )
    print_policy_summary(cfg, policy, device, ckpt_path)

    if args.dry_run:
        run_dummy_inference(cfg, policy, device)


if __name__ == "__main__":
    main()
