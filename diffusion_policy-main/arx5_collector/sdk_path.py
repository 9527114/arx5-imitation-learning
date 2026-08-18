import os
import sys


def ensure_arx5_sdk_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_root = os.path.dirname(repo_root)
    sdk_python = os.path.join(workspace_root, "arx5-sdk-main", "python")
    if os.path.isdir(sdk_python) and sdk_python not in sys.path:
        sys.path.insert(0, sdk_python)
    return sdk_python
