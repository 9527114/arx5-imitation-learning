import json
import time
from pathlib import Path

import numpy as np


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


class TrajectoryLogger:
    """Small JSONL logger for online policy debugging.

    This intentionally stays independent from wandb/training logs. Each line is
    one deployment event, so it can be inspected with normal shell tools.
    """

    def __init__(self, path=None, buffer_sample_interval: float = 0.1):
        self.path = Path(path).expanduser() if path else None
        self.buffer_sample_interval = float(buffer_sample_interval)
        self._file = None
        self._last_buffer_sample = 0.0

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def open(self):
        if not self.enabled or self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self.log("logger_open", log_path=str(self.path))

    def close(self):
        if self._file is None:
            return
        self.log("logger_close")
        self._file.close()
        self._file = None

    def log(self, event, **payload):
        if self._file is None:
            return
        now = time.time()
        if event == "buffer_sample":
            if now - self._last_buffer_sample < self.buffer_sample_interval:
                return
            self._last_buffer_sample = now
        record = {
            "event": str(event),
            "wall_time": now,
        }
        record.update({key: _json_safe(value) for key, value in payload.items()})
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
