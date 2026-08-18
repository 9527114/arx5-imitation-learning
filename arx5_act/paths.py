from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DP_ROOT = PROJECT_ROOT / "diffusion_policy-main"
ACT_ROOT = PROJECT_ROOT / "act-main"
ACT_DETR_ROOT = ACT_ROOT / "detr"


def ensure_project_paths():
    for path in (str(DP_ROOT), str(ACT_ROOT), str(ACT_DETR_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
