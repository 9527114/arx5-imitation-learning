import numpy as np


class TemporalActionAggregator:
    """ACT-style temporal ensemble over overlapping action chunks."""

    def __init__(self, chunk_size: int, k: float = 0.01, order: str = "oldest"):
        if order not in ("oldest", "newest"):
            raise ValueError(f"Unsupported temporal aggregation order: {order}")
        self.chunk_size = int(chunk_size)
        self.k = float(k)
        self.order = order
        self._chunks = []

    def clear(self):
        self._chunks = []

    def add_chunk(self, step: int, chunk):
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError(f"Expected action chunk shape (T, D), got {chunk.shape}.")
        step = int(step)
        self._chunks.append((step, chunk.copy()))
        min_start = step - self.chunk_size + 1
        self._chunks = [
            (start, old_chunk)
            for start, old_chunk in self._chunks
            if start >= min_start
        ]

    def current_action(self, step: int, fallback_action):
        current_actions = []
        for start, chunk in self._chunks:
            offset = int(step) - start
            if 0 <= offset < len(chunk):
                current_actions.append(chunk[offset])
        if len(current_actions) == 0:
            return np.asarray(fallback_action, dtype=np.float64).copy(), 0

        current_actions = np.asarray(current_actions, dtype=np.float64)
        if self.order == "newest":
            current_actions = current_actions[::-1]
        weights = np.exp(-self.k * np.arange(len(current_actions), dtype=np.float64))
        weights = weights / np.sum(weights)
        return np.sum(current_actions * weights[:, None], axis=0), len(current_actions)
