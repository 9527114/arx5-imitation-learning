import argparse

from arx5_act.dataset import Arx5ActDataset, compute_norm_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()

    stats = compute_norm_stats(args.dataset_path)
    dataset = Arx5ActDataset(args.dataset_path, stats, chunk_size=args.chunk_size)
    image, qpos, action, is_pad = dataset[0]
    print(f"dataset_path: {args.dataset_path}")
    print(f"samples: {len(dataset)}")
    print(f"image: shape={tuple(image.shape)}")
    print(f"qpos: shape={tuple(qpos.shape)}")
    print(f"action: shape={tuple(action.shape)}")
    print(f"is_pad: shape={tuple(is_pad.shape)}")
    print("qpos_mean:", stats["qpos_mean"])
    print("action_mean:", stats["action_mean"])


if __name__ == "__main__":
    main()

