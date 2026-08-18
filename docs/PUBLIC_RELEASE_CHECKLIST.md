# Public Release Checklist

This checklist is based on static inspection only. No robot, CAN, camera, or training job was executed.

## SAFE

| Item | Reason | Action |
| --- | --- | --- |
| Core ARX5 Python source | Needed to reproduce collection, training, and deployment. | Keep. |
| Stable shell wrappers in `scripts/` | Public entry points for repeatable commands. | Keep after path review. |
| `README.md` and `docs/*.md` | Public documentation. | Keep. |
| Hydra configs for ARX5 DP/CFG | Needed for reproducible training. | Keep. |
| Environment YAML files | Useful reference for dependencies. | Keep, but do not promise they are universal. |

## NEEDS_REVIEW

| Item | Risk | Recommendation |
| --- | --- | --- |
| Absolute paths in Chinese notes and historical logs | They include local usernames, mount points, and lab paths such as `/media/...` and `/home/...`. | Keep locally; sanitize before public upload or move to private notes. |
| Default checkpoint paths in deployment wrappers | They point to local experiment names. | Acceptable for local use, but document that users should override `CKPT_PATH`. |
| `arx5-sdk-main/` | Contains hardware SDK code, binary extensions, and upstream licensing concerns. | Review redistribution permission before publishing. |
| `act-main/` and `diffusion_policy-main/` upstream snapshots | Vendored upstream code. | Keep license files and attribution; consider submodules later. |
| Camera config files under `scripts/camera/` or collector configs | May encode local device assumptions. | Review before publishing. |
| Chinese development logs | Useful project history but may expose local paths and experiments. | Keep private or publish a sanitized summary. |
| MDR / MoE / multi-task / adaptive expert notes | May relate to ongoing or unpublished research ideas. | Keep high-level only unless collaborators approve release. |
| Future point-cloud / DP3 references | No active implementation was confirmed in this checkout. | Do not present as completed work. |
| Demo media generated from raw datasets | May expose lab background, people, screens, or private object layouts. | Review frame content before adding to `assets/demos/`. |

## SHOULD_NOT_UPLOAD

| Item | Reason |
| --- | --- |
| `diffusion_policy-main/data_local/` | Contains datasets, videos, zarr buffers, and robot recordings. |
| `diffusion_policy-main/data/outputs/` | Contains checkpoints, W&B metadata, and run configs. |
| `act_outputs/` | Contains ACT checkpoints and stats. |
| `logs/` | Contains local run logs and paths. |
| `project_trash/` | Historical archive, backups, and potentially duplicated upstream files. |
| `*.ckpt`, `*.pt`, `*.pth`, `*.onnx` | Large trained models; publish only curated releases. |
| `*.zarr/`, `*.zarr.zip`, `videos/`, media files | Large data artifacts and possible privacy/lab-background exposure. |
| `wandb/`, `runs/`, `tensorboard/` | Experiment tracker metadata. |
| `__pycache__/`, build outputs, compiled objects | Generated artifacts. |
| Raw unreviewed demo videos | Privacy and size risk; use curated GIFs or release assets only. |

## Static Scan Findings

- Absolute local paths remain in historical notes, logs, and some wrapper defaults.
- The primary environment scripts were previously adjusted to use script-relative project roots.
- No obvious API key, password, SSH key, or bearer token was confirmed by the static search.
- Several defaults still reference local dataset/checkpoint names; that is acceptable for local scripts but should be explained in docs.
- This workspace root did not behave as a normal Git repository during previous checks, so tracked status could not be verified here. If `project_trash/` is already tracked, run:

```bash
git rm -r --cached project_trash
```

Do the same for any already-tracked local datasets, checkpoints, logs, or zarr caches.

## Minimal Fixes Already Applied

- Added broader ignores for `outputs/`, `dataset/`, `datasets/`, `checkpoints/`, `dist/`, and `*.egg-info/`.
- Kept active code untouched.
- Kept `project_trash/` ignored rather than deleting it.

## Before Publishing

1. Run `git status --ignored` in a real Git checkout.
2. Confirm no data, videos, checkpoints, W&B runs, or local logs are staged.
3. Check upstream license compatibility for Diffusion Policy, ACT/DETR, ARX5 SDK, and copied dependencies.
4. Replace local dataset/checkpoint examples with placeholders or public release artifacts.
5. Decide whether to keep large upstream snapshots or convert them to submodules.
