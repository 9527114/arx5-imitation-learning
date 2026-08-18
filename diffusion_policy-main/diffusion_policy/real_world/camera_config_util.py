import os
from typing import Any, Dict, Optional

import cv2
import yaml


def load_realsense_config(config_path: str) -> Dict[str, Any]:
    return load_camera_config(config_path, key="realsense")


def load_usb_camera_config(config_path: str) -> Dict[str, Any]:
    return load_camera_config(config_path, key="usb_camera")


def load_camera_config(config_path: str, key: str) -> Dict[str, Any]:
    config_path = os.path.expanduser(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or key not in config:
        raise ValueError(f"Missing top-level '{key}' config in {config_path}")
    camera_config = config[key]
    if not isinstance(camera_config, dict):
        raise ValueError(f"'{key}' config must be a dict in {config_path}")
    return camera_config


def apply_realsense_config(realsense, config: Optional[Dict[str, Any]]) -> None:
    if config is None:
        config = {"mode": "auto"}

    mode = config.get("mode", "manual")
    if mode == "auto":
        realsense.set_exposure(exposure=None, gain=None)
        realsense.set_white_balance(white_balance=None)
        return
    if mode != "manual":
        raise ValueError(f"Unsupported RealSense mode: {mode}")

    exposure = config.get("exposure")
    gain = config.get("gain")
    white_balance = config.get("white_balance")
    if exposure is None or gain is None or white_balance is None:
        raise ValueError(
            "Manual RealSense config requires exposure, gain, and white_balance."
        )

    realsense.set_exposure(exposure=exposure, gain=gain)
    realsense.set_white_balance(white_balance=white_balance)


def load_and_apply_realsense_config(realsense, config_path: Optional[str]) -> None:
    if config_path is None:
        apply_realsense_config(realsense, None)
        return
    apply_realsense_config(realsense, load_realsense_config(config_path))


def apply_usb_camera_config(cap, config: Optional[Dict[str, Any]]) -> None:
    if config is None:
        return

    mode = config.get("mode", "manual")
    if mode == "auto":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
        return
    if mode != "manual":
        raise ValueError(f"Unsupported USB camera mode: {mode}")

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)

    prop_map = {
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "gain": cv2.CAP_PROP_GAIN,
        "white_balance": cv2.CAP_PROP_WB_TEMPERATURE,
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
    }
    for key, prop in prop_map.items():
        value = config.get(key)
        if value is not None:
            cap.set(prop, float(value))


def load_and_apply_usb_camera_config(cap, config_path: Optional[str]) -> None:
    if config_path is None:
        return
    apply_usb_camera_config(cap, load_usb_camera_config(config_path))
