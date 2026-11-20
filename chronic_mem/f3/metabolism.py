"""
F3 — Real-Time Metabolism Score f(E_p, t) and Tri-State Lifecycle Thresholds.

Profile-guided episode metabolism score:

    f(E_p, t) = Res(E_p, v_profile) · exp(-Δt / λ_Episode)

where:
- Res(E_p, v_profile) = (cos(v_{E_p}, v_profile) + 1) / 2 in [0, 1]
  (falls back to urgency φ salience in [0.6, 1.0] when v_profile is empty)
- Δt = t - reference_time (days since latest clinical activity or wake-up)
- λ_Episode = max_{e_j in E_p} λ_{e_j}

Tri-State Transition Thresholds (defaults):
- Active     (f > 0.7):         Full atomic & event memory retained
- Compressed (0.3 < f <= 0.7):  CSPC extracts Permanent Labels & physically prunes weak atoms
- Forgotten  (f <= 0.3):        All atoms & event vectors physically deleted; Permanent Labels retained
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from chronic_mem.f3.controller import (
    GRANULARITY_ATOMS,
    GRANULARITY_LABELS,
    GRANULARITY_SUMMARIES,
    retrieval_granularity,
)
from chronic_mem.geometry import cosine_unit_interval

ACTIVE = "active"
COMPRESSED = "compressed"
FORGOTTEN = "forgotten"

DEFAULT_ACTIVE_THRESHOLD = 0.7
DEFAULT_COMPRESS_THRESHOLD = 0.3
DEFAULT_PRUNE_THRESHOLD = 0.3


def phi_salience_fallback(max_phi: float) -> float:
    """Map urgency scalar φ in [0.8, 1.5] to salience in [0.6, 1.0] when v_profile is absent."""
    phi_clamped = max(0.8, min(1.5, float(max_phi)))
    return 0.6 + (phi_clamped - 0.8) * (0.4 / 0.7)


def profile_resonance(
    v_entity: np.ndarray,
    v_profile: Optional[np.ndarray],
    *,
    fallback_phi: float = 1.0,
) -> float:
    """Compute Res(·, v_profile) in [0, 1] using global patient profile vector v_profile."""
    if v_profile is None:
        return phi_salience_fallback(fallback_phi)
    return cosine_unit_interval(v_entity, v_profile)


def metabolism_score(resonance_val: float, delta_days: float, lambda_days: float) -> float:
    """Compute f(E_p, t) = Res · exp(-Δt / (λ + ε))."""
    lam = float(lambda_days) + 1e-8
    decay = math.exp(-float(delta_days) / lam)
    return float(resonance_val) * decay


def tri_state(
    score: float,
    *,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
    compress_threshold: float = DEFAULT_COMPRESS_THRESHOLD,
) -> str:
    """Map f(E_p, t) to Tri-State lifecycle (`active` | `compressed` | `forgotten`)."""
    if score > active_threshold:
        return ACTIVE
    if score > compress_threshold:
        return COMPRESSED
    return FORGOTTEN


__all__ = [
    "ACTIVE",
    "COMPRESSED",
    "FORGOTTEN",
    "DEFAULT_ACTIVE_THRESHOLD",
    "DEFAULT_COMPRESS_THRESHOLD",
    "DEFAULT_PRUNE_THRESHOLD",
    "GRANULARITY_ATOMS",
    "GRANULARITY_SUMMARIES",
    "GRANULARITY_LABELS",
    "phi_salience_fallback",
    "profile_resonance",
    "metabolism_score",
    "tri_state",
    "retrieval_granularity",
]
