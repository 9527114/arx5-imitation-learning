import numpy as np
import torch
from tqdm import tqdm


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(policy, dataloader, device, optimizer=None):
    is_train = optimizer is not None
    policy.train(is_train)
    metrics = {}
    for image, qpos, action, is_pad in tqdm(dataloader, leave=False):
        image = image.to(device, non_blocking=True)
        qpos = qpos.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        is_pad = is_pad.to(device, non_blocking=True)
        loss_dict = policy(qpos, image, action, is_pad)
        loss = loss_dict["loss"]
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for key, value in loss_dict.items():
            metrics.setdefault(key, []).append(float(value.detach().cpu()))
    if not metrics:
        return {"loss": np.inf}
    return {key: float(np.mean(values)) for key, values in metrics.items()}
