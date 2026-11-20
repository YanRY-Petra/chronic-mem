"""
F2 — Associated Latching (Max-Pooling).

Protects an entire clinical event context whenever at least one high-value
Memory Atom (e.g., an allergy or surgical safety anchor) resides in the event:

    λ_Event = max(λ_{m_1}, ..., λ_{m_n})
"""
from __future__ import annotations

from typing import Sequence


def max_pool_lambda(lambda_values: Sequence[float], *, floor: float = 1.0) -> float:
    """Associated Latching: λ_Event = max(λ_{m_1}, ..., λ_{m_n}), or `floor` if empty."""
    vals = [float(x) for x in lambda_values]
    if not vals:
        return float(floor)
    return float(max(vals))
