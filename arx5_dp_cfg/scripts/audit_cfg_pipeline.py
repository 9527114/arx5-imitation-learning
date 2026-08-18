import argparse
import os
from pathlib import Path

for _key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_key, None)

import hydra
import numpy as np
import torch

from arx5_ckpt_loader.policy_loader import load_policy_from_ckpt


def _chdir_to_dp_root(ckpt_path: Path):
    for parent in [ckpt_path.parent, *ckpt_path.parents]:
        if parent.name == "diffusion_policy-main":
            os.chdir(parent)
            return parent
    return Path.cwd()


def _to_device_obs(sample, device):
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in sample["obs"].items()
    }


def _predict(policy, sample, device, seed, w=1.0, zero_prev=False):
    obs_dict = _to_device_obs(sample, device)
    if "prev_action" in sample:
        if zero_prev:
            prev_action = torch.zeros_like(sample["prev_action"])
            prev_mask = torch.zeros_like(sample["prev_action_mask"])
        else:
            prev_action = sample["prev_action"]
            prev_mask = sample["prev_action_mask"]
        obs_dict["prev_action"] = prev_action.unsqueeze(0).to(device)
        obs_dict["prev_action_mask"] = prev_mask.unsqueeze(0).to(device)
        obs_dict["cfg_guidance_weight"] = torch.as_tensor([float(w)], device=device)
    torch.manual_seed(int(seed))
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    return result["action_pred"][0].detach().cpu().numpy()


def _pose_l2(a, b, n=8):
    return float(np.linalg.norm(a[:n, :6] - b[:n, :6], axis=-1).mean())


def _gripper_l1(a, b, n=8):
    return float(np.abs(a[:n, 6] - b[:n, 6]).mean())


def _fmt(x, precision=5):
    return np.array2string(np.asarray(x), precision=precision, suppress_small=True)


def _normalizer_keys(policy):
    try:
        return sorted(policy.normalizer.params_dict.keys())
    except Exception:
        try:
            return sorted(policy.normalizer.state_dict().keys())
        except Exception:
            return []


def audit_dataset_windows(dataset, indices):
    prefix_pose_errors = []
    prefix_gripper_errors = []
    mask_counts = []
    starts = []
    for idx in indices:
        sample = dataset[idx]
        if "prev_action" not in sample:
            continue
        prev = sample["prev_action"].detach().cpu().numpy()
        mask = sample["prev_action_mask"].detach().cpu().numpy() > 0.5
        action = sample["action"].detach().cpu().numpy()
        n = min(int(mask.sum()), len(action))
        mask_counts.append(int(mask.sum()))
        starts.append(dataset.sampler.indices[idx].tolist())
        if n > 0:
            valid_prev = prev[mask][:n]
            prefix_pose_errors.append(
                float(np.linalg.norm(valid_prev[:, :6] - action[:n, :6], axis=-1).mean())
            )
            prefix_gripper_errors.append(float(np.abs(valid_prev[:, 6] - action[:n, 6]).mean()))
    if not mask_counts:
        print("prev_window: not present")
        return
    print("prev_window:")
    print(f"  samples_checked={len(mask_counts)}")
    print(
        "  valid_steps:",
        f"min={min(mask_counts)}",
        f"mean={float(np.mean(mask_counts)):.2f}",
        f"max={max(mask_counts)}",
    )
    if prefix_pose_errors:
        print(
            "  prev_vs_action_prefix:",
            f"pose_l2_mean={float(np.mean(prefix_pose_errors)):.8f}",
            f"gripper_l1_mean={float(np.mean(prefix_gripper_errors)):.8f}",
        )
    print(f"  first_sampler_indices={starts[0] if starts else None}")


def audit_sensitivity(policy, dataset, indices, device, seed):
    samples = [dataset[i] for i in indices]
    print("sensitivity:")
    if "prev_action" in samples[0]:
        base_prev = samples[0]["prev_action"]
        base_mask = samples[0]["prev_action_mask"]
        base_outputs = []
        for sample in samples:
            patched = dict(sample)
            patched["prev_action"] = base_prev
            patched["prev_action_mask"] = base_mask
            base_outputs.append(_predict(policy, patched, device, seed, w=1.0))
        for idx, output in zip(indices[1:], base_outputs[1:]):
            print(
                f"  fixed_prev obs0->{idx}:",
                f"pose_l2={_pose_l2(base_outputs[0], output):.6f}",
                f"gripper_l1={_gripper_l1(base_outputs[0], output):.6f}",
            )

        prev_outputs = [_predict(policy, sample, device, seed, w=1.0) for sample in samples]
        for idx, output in zip(indices[1:], prev_outputs[1:]):
            print(
                f"  obs_and_prev 0->{idx}:",
                f"pose_l2={_pose_l2(prev_outputs[0], output):.6f}",
                f"gripper_l1={_gripper_l1(prev_outputs[0], output):.6f}",
            )

        cond = _predict(policy, samples[0], device, seed, w=1.0)
        uncond = _predict(policy, samples[0], device, seed, w=0.0)
        zero_prev = _predict(policy, samples[0], device, seed, w=1.0, zero_prev=True)
        print(
            "  w1_vs_w0:",
            f"pose_l2={_pose_l2(cond, uncond):.6f}",
            f"gripper_l1={_gripper_l1(cond, uncond):.6f}",
        )
        print(
            "  real_prev_vs_zero_prev:",
            f"pose_l2={_pose_l2(cond, zero_prev):.6f}",
            f"gripper_l1={_gripper_l1(cond, zero_prev):.6f}",
        )
    else:
        outputs = [_predict(policy, sample, device, seed) for sample in samples]
        for idx, output in zip(indices[1:], outputs[1:]):
            print(
                f"  obs0->{idx}:",
                f"pose_l2={_pose_l2(outputs[0], output):.6f}",
                f"gripper_l1={_gripper_l1(outputs[0], output):.6f}",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--indices", default="")
    args = parser.parse_args()

    cfg, policy, device, ckpt_path = load_policy_from_ckpt(
        args.ckpt,
        device=args.device,
        inference_steps=args.inference_steps,
    )
    dp_root = _chdir_to_dp_root(ckpt_path)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if args.indices.strip():
        indices = [int(x) for x in args.indices.split(",") if x.strip()]
    else:
        indices = [0, len(dataset) // 4, len(dataset) // 2, len(dataset) - 1]

    print("cfg_audit:")
    print(f"  dp_root={dp_root}")
    print(f"  ckpt={ckpt_path}")
    print(f"  workspace={cfg._target_}")
    print(f"  policy={cfg.policy._target_}")
    print(f"  dataset_path={cfg.task.dataset_path}")
    print(f"  dataset_target={cfg.task.dataset._target_}")
    print(f"  horizon={cfg.horizon} n_obs_steps={cfg.n_obs_steps} n_action_steps={cfg.n_action_steps}")
    print(f"  prev_cond_steps={cfg.get('prev_cond_steps', None)}")
    print(f"  prev_chunk_dropout={cfg.get('prev_chunk_dropout', None)}")
    print(f"  prev_action_mode={cfg.get('prev_action_mode', None)}")
    print(f"  target_frequency={cfg.task.dataset.get('target_frequency', None)}")
    print(f"  shape_obs_order={list(cfg.task.shape_meta.obs.keys())}")
    print(f"  dataset_rgb_keys={dataset.rgb_keys}")
    print(f"  dataset_lowdim_keys={dataset.lowdim_keys}")
    print(f"  dataset_len={len(dataset)} episodes={dataset.replay_buffer.n_episodes} steps={dataset.replay_buffer.n_steps}")
    print(f"  normalizer_keys={_normalizer_keys(policy)}")
    print(
        "  obs_encoder_params:",
        f"trainable={sum(p.numel() for p in policy.obs_encoder.parameters() if p.requires_grad)}",
        f"total={sum(p.numel() for p in policy.obs_encoder.parameters())}",
    )
    print(
        "  prev_encoder_params:",
        "absent" if not hasattr(policy, "prev_chunk_encoder")
        else sum(p.numel() for p in policy.prev_chunk_encoder.parameters()),
    )
    audit_dataset_windows(dataset, indices)
    audit_sensitivity(policy, dataset, indices, device, args.seed)


if __name__ == "__main__":
    main()
