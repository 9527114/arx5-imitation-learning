import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from arx5_act.dataset import (
    Arx5ActDataset,
    compute_norm_stats,
    make_episode_split,
)
from arx5_act.policy_utils import (
    build_policy,
    make_policy_config,
    resolve_device,
    save_training_bundle,
)
from arx5_act.training import run_epoch, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--target-frequency", type=float, default=20.0)
    parser.add_argument("--state-mode", default="eef", choices=["eef", "joint"])
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs. Set <=0 to disable validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-names", nargs="+", default=["camera_0", "camera_1", "camera_2"])
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    stats = compute_norm_stats(
        args.dataset_path,
        target_frequency=args.target_frequency,
        camera_names=args.camera_names,
        state_mode=args.state_mode,
    )
    with open(ckpt_dir / "dataset_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    from diffusion_policy.common.replay_buffer import ReplayBuffer

    replay = ReplayBuffer.create_from_path(
        str(Path(args.dataset_path).joinpath("replay_buffer.zarr").absolute()),
        mode="r",
    )
    train_eps, val_eps = make_episode_split(replay.n_episodes, args.val_ratio, args.seed)
    image_size = (args.image_width, args.image_height)
    train_dataset = Arx5ActDataset(
        args.dataset_path,
        stats,
        camera_names=args.camera_names,
        chunk_size=args.chunk_size,
        image_size=image_size,
        episode_indices=train_eps,
        use_cache=not args.no_cache,
        target_frequency=args.target_frequency,
        state_mode=args.state_mode,
        cache_dir=args.cache_dir,
    )
    val_dataset = Arx5ActDataset(
        args.dataset_path,
        stats,
        camera_names=args.camera_names,
        chunk_size=args.chunk_size,
        image_size=image_size,
        episode_indices=val_eps,
        use_cache=not args.no_cache,
        target_frequency=args.target_frequency,
        state_mode=args.state_mode,
        cache_dir=args.cache_dir,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory and device.type == "cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = 1
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    print(
        "ACT dataloader:",
        f"batch_size={args.batch_size}",
        f"num_workers={args.num_workers}",
        f"pin_memory={loader_kwargs['pin_memory']}",
        f"cache={not args.no_cache}",
        f"target_frequency={args.target_frequency:g}Hz",
        f"state_mode={args.state_mode}",
        f"val_every={args.val_every}",
    )
    policy_config = make_policy_config(args)
    policy = build_policy(policy_config, device=device)
    optimizer = policy.configure_optimizers()
    save_training_bundle(
        ckpt_dir=ckpt_dir,
        dataset_path=args.dataset_path,
        policy_config=policy_config,
        camera_names=args.camera_names,
        chunk_size=args.chunk_size,
        target_frequency=args.target_frequency,
        state_mode=args.state_mode,
        train_episodes=train_eps,
        val_episodes=val_eps,
    )

    best_val = np.inf
    for epoch in range(args.num_epochs):
        train_metrics = run_epoch(policy, train_loader, device=device, optimizer=optimizer)
        train_loss = train_metrics["loss"]
        should_validate = args.val_every > 0 and (epoch % args.val_every) == 0
        if should_validate:
            with torch.inference_mode():
                val_metrics = run_epoch(policy, val_loader, device=device, optimizer=None)
            val_loss = val_metrics["loss"]
            best_metric = val_loss
        else:
            val_metrics = {}
            val_loss = float("nan")
            best_metric = train_loss
        train_extra = " ".join(
            f"train_{key}={value:.6f}" for key, value in sorted(train_metrics.items()) if key != "loss"
        )
        val_extra = " ".join(
            f"val_{key}={value:.6f}" for key, value in sorted(val_metrics.items()) if key != "loss"
        )
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} {train_extra} {val_extra}".strip())
        torch.save(policy.state_dict(), ckpt_dir / "policy_latest.ckpt")
        if best_metric < best_val:
            best_val = best_metric
            torch.save(policy.state_dict(), ckpt_dir / "policy_best.ckpt")
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            torch.save(policy.state_dict(), ckpt_dir / f"policy_epoch_{epoch + 1}.ckpt")

    print(f"Best val loss: {best_val:.6f}")
    print(f"Saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
