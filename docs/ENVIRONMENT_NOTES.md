# Environment Notes

This is not a fully frozen dependency lockfile. It records what the current project appears to need.

## Confirmed From Environment Files

Main file:

```text
conda_environment_arx5_real.yaml
```

Confirmed conda dependencies include:

- Python 3.9
- PyTorch 1.12.1
- torchvision 0.13.1
- cudatoolkit 11.6
- pytorch3d 0.7.0
- numpy 1.23.3
- scipy 1.9.1
- numba 0.56.4
- OpenCV 4.6
- zarr 2.12.0
- numcodecs 0.10.2
- h5py 3.7.0
- hydra-core 1.2.0
- diffusers 0.11.1
- datasets 2.6.1
- accelerate 0.13.2
- wandb 0.13.3
- tensorboard / tensorboardx
- av, imageio, imageio-ffmpeg
- scikit-image, scikit-video
- click, tqdm, dill, psutil

Confirmed pip dependencies include:

- `pyrealsense2`
- `spnav`
- `pynput`
- `ray[default,tune]`
- `ur-rtde`
- `atomics`

## Hardware-Specific Requirements

The real-robot stack also depends on:

- ARX5 SDK Python bindings and shared libraries.
- `LD_LIBRARY_PATH` pointing to the SDK library directory.
- SocketCAN support and a configured CAN interface such as `can1`.
- RealSense runtime support and udev permissions.
- V4L2 access for the USB wrist camera.
- `spacenavd` for the 3Dconnexion SpaceMouse.
- NVIDIA driver compatible with the installed CUDA/PyTorch stack.

## Likely or Branch-Specific Requirements

These are needed only for some branches or inherited upstream code:

- Robomimic for some Diffusion Policy workspaces.
- ACT / DETR dependencies from `act-main/`.
- Mujoco, PyBullet, gym, or simulation dependencies for upstream tasks not used in the ARX5 real-robot path.

## Version Unknown

The exact ARX5 SDK binary build, system CAN tooling versions, RealSense firmware, camera device naming, and robot firmware are machine-specific and were not inferred from a lockfile.

## Recommended Setup Notes

Use:

```bash
conda env create -f conda_environment_arx5_real.yaml
source ./activate_arx5_env.sh
```

`activate_arx5_env.sh` changes the working directory to `diffusion_policy-main` and adds the SDK plus `diffusion_policy-main` to `PYTHONPATH`. Root-level packages such as `arx5_act` and `arx5_dp_cfg` require the repository root on `PYTHONPATH`; stable shell wrappers should handle this, but manual commands may need:

```bash
export PYTHONPATH=/path/to/CY_arx5_dp:$PYTHONPATH
```

Then verify non-robot imports first:

```bash
python -m arx5_ckpt_loader.load_arx5_ckpt --help
export PYTHONPATH=/path/to/CY_arx5_dp:$PYTHONPATH
python -m arx5_act.train_act --help
python -m arx5_dp_cfg.run_arx5_cfg_policy --help
```

Robot, CAN, camera, and SpaceMouse checks should be run only on the target machine with hardware attached.
