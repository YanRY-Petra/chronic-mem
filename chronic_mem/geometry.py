"""Shared vector geometry for F2/F3 prototype updates."""
from __future__ import annotations

import numpy as np


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(x))
    if n < eps:
        return x
    return x / n


def cosine_unit_interval(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    sim = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def cosine_unit_interval_normalized(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine in [0, 1] when both vectors are already L2-normalized."""
    cos = float(np.dot(a, b))
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))
