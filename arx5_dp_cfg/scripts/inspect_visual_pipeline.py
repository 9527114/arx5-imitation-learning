import argparse
import os
from pathlib import Path

for _key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_key, None)

import hydra
import numpy as np
import torch

from arx5_ckpt_loader.policy_loader import load_policy_from_ckpt


def _to_device_obs(sample, device):
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in sample["obs"].items()
    }


def _predict(policy, obs, prev_action, prev_mask, device, seed, guidance_weight):
    obs_dict = {key: value.clone() for key, value in obs.items()}
    if prev_action is not None:
        obs_dict["prev_action"] = prev_action.unsqueeze(0).to(device)
        obs_dict["prev_action_mask"] = prev_mask.unsqueeze(0).to(device)
        obs_dict["cfg_guidance_weight"] = torch.as_tensor(
            [float(guidance_weight)],
            device=device,
            dtype=torch.float32,
        )
    torch.manual_seed(int(seed))
    with torch.no_grad():
        return policy.predict_action(obs_dict)["action_pred"][0].detach().cpu().numpy()


def _mean_pose_diff(a, b, n=8):
    return float(np.linalg.norm(a[:n, :6] - b[:n, :6], axis=-1).mean())


def _mean_gripper_diff(a, b, n=8):
    return float(np.abs(a[:n, 6] - b[:n, 6]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--guidance-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--indices", default="")
    args = parser.parse_args()

    cfg, policy, device, ckpt_path = load_policy_from_ckpt(
        args.ckpt,
        device=args.device,
        inference_steps=args.inference_steps,
    )
    for parent in [ckpt_path.parent, *ckpt_path.parents]:
        if parent.name == "diffusion_policy-main":
            os.chdir(parent)
            break
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    if args.indices.strip():
        indices = [int(x) for x in args.indices.split(",") if x.strip()]
    else:
        indices = [0, len(dataset) // 4, len(dataset) // 2, len(dataset) - 1]
    samples = [dataset[i] for i in indices]

    print(f"ckpt: {ckpt_path}")
    print(f"workspace: {cfg._target_}")
    print(f"policy: {cfg.policy._target_}")
    print(f"dataset_path: {cfg.task.dataset_path}")
    print(f"dataset_target: {cfg.task.dataset._target_}")
    print(f"prev_cond_steps: {cfg.get('prev_cond_steps', None)}")
    print(f"prev_chunk_dropout: {cfg.get('prev_chunk_dropout', None)}")
    print(f"prev_action_mode: {cfg.get('prev_action_mode', None)}")
    print(f"shape_meta_obs_order: {list(cfg.task.shape_meta.obs.keys())}")
    print(f"dataset_rgb_keys: {dataset.rgb_keys}")
    print(f"dataset_lowdim_keys: {dataset.lowdim_keys}")
    print(f"dataset_len: {len(dataset)}")
    print(f"policy_prev_cond_steps: {getattr(policy, 'prev_cond_steps', None)}")
    print(f"obs_encoder_trainable_params: {sum(p.numel() for p in policy.obs_encoder.parameters() if p.requires_grad)}")
    print(f"obs_encoder_total_params: {sum(p.numel() for p in policy.obs_encoder.parameters())}")

    has_prev_action = "prev_action" in samples[0]
    base_prev = samples[0]["prev_action"] if has_prev_action else None
    base_mask = samples[0]["prev_action_mask"] if has_prev_action else None

    title = "Fixed prev_action, varying obs" if has_prev_action else "Varying obs"
    print(f"\n{title}:")
    obs_outputs = []
    for idx, sample in zip(indices, samples):
        output = _predict(
            policy,
            _to_device_obs(sample, device),
            base_prev,
            base_mask,
            device,
            args.seed,
            args.guidance_weight,
        )
        obs_outputs.append(output)
        print(
            f"  idx={idx}",
            f"first={np.array2string(output[1, :7], precision=4)}",
            f"mean_xyz={np.array2string(output[:8, :3].mean(axis=0), precision=4)}",
        )
    for idx, output in zip(indices[1:], obs_outputs[1:]):
        print(
            f"  obs_diff idx0->{idx}:",
            f"pose_mean_l2={_mean_pose_diff(obs_outputs[0], output):.6f}",
            f"gripper_mean_abs={_mean_gripper_diff(obs_outputs[0], output):.6f}",
        )

    if not has_prev_action:
        print("\nNo prev_action fields in this dataset/policy; skipping CFG prev_action checks.")
        return

    print("\nFixed obs, varying prev_action:")
    base_obs = _to_device_obs(samples[0], device)
    prev_outputs = []
    for idx, sample in zip(indices, samples):
        output = _predict(
            policy,
            base_obs,
            sample["prev_action"],
            sample["prev_action_mask"],
            device,
            args.seed,
            args.guidance_weight,
        )
        prev_outputs.append(output)
        print(
            f"  prev_idx={idx}",
            f"first={np.array2string(output[1, :7], precision=4)}",
            f"mean_xyz={np.array2string(output[:8, :3].mean(axis=0), precision=4)}",
        )
    for idx, output in zip(indices[1:], prev_outputs[1:]):
        print(
            f"  prev_diff idx0->{idx}:",
            f"pose_mean_l2={_mean_pose_diff(prev_outputs[0], output):.6f}",
            f"gripper_mean_abs={_mean_gripper_diff(prev_outputs[0], output):.6f}",
        )

    zero_prev = torch.zeros_like(base_prev)
    zero_mask = torch.zeros_like(base_mask)
    zero_output = _predict(
        policy,
        base_obs,
        zero_prev,
        zero_mask,
        device,
        args.seed,
        args.guidance_weight,
    )
    print("\nFixed obs, zero prev vs real prev:")
    print(f"  real_first={np.array2string(prev_outputs[0][1, :7], precision=4)}")
    print(f"  zero_first={np.array2string(zero_output[1, :7], precision=4)}")
    print(
        "  zero_real_diff:",
        f"pose_mean_l2={_mean_pose_diff(prev_outputs[0], zero_output):.6f}",
        f"gripper_mean_abs={_mean_gripper_diff(prev_outputs[0], zero_output):.6f}",
    )


if __name__ == "__main__":
    main()
