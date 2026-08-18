import argparse

import numpy as np

from arx5_act.policy_utils import load_bundle, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", default="policy_best.ckpt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    config, stats, policy, ckpt_path = load_bundle(args.ckpt_dir, args.ckpt_name, device)
    print(f"ckpt: {ckpt_path}")
    print(f"device: {device}")
    print(f"camera_names: {config['camera_names']}")
    print(f"chunk_size: {config['chunk_size']}")
    print(f"state_dim: {config['state_dim']}")
    print(f"action_dim: {config['action_dim']}")
    print(f"policy_config: {config['policy_config']}")
    print("qpos_mean:", np.array2string(stats["qpos_mean"], precision=4))
    print("qpos_std:", np.array2string(stats["qpos_std"], precision=4))
    print("action_mean:", np.array2string(stats["action_mean"], precision=4))
    print("action_std:", np.array2string(stats["action_std"], precision=4))
    print(f"params: {sum(p.numel() for p in policy.parameters()):.6e}")


if __name__ == "__main__":
    main()
