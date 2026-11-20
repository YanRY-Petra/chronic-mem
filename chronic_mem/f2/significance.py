"""
F2 — Significance Gravity & Event Aggregation.

Aggregation score:

    T(Atom, Event_j) = (γ_med · Sim_med + γ_ctx · Sim_ctx) / (α + 1 / λ_Atom)

where Sim is cosine similarity mapped from [-1, 1] to [0, 1].
- Threshold criterion: T(Atom, Event_j) > ρ_event (default 0.85) attaches the
  atom to Event_j and updates its medical/context prototypes via EMA (0.9 / 0.1),
  skipping narrative-generic atoms (`summary`, `follow_up`, `people`, `T000`) to
  prevent prototype drift.
- Otherwise, a new Event_j is instantiated.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from chronic_mem.f2.latching import max_pool_lambda
from chronic_mem.geometry import cosine_unit_interval_normalized, l2_normalize

GAMMA_MED = 0.8
GAMMA_CTX = 0.2
EMA_OLD = 0.9
EMA_NEW = 0.1
DEFAULT_RHO_EVENT = 0.85
DEFAULT_ALPHA = 0.5


def choice_score(
    sim_terms: Sequence[Tuple[float, float]],
    *,
    alpha: float = DEFAULT_ALPHA,
    lambda_m: float,
    lambda_floor: float = 1e-6,
) -> float:
    """Compute T = (Σ_k γ_k · Sim_k) / (α + 1/λ_m). Higher λ_m yields stronger gravity."""
    lam = max(float(lambda_m), float(lambda_floor))
    numer = sum(float(g) * float(s) for g, s in sim_terms)
    denom = float(alpha) + 1.0 / lam
    return float(numer / denom) if denom > 1e-12 else 0.0


def significance_gravity(
    v_med: np.ndarray,
    v_ctx: np.ndarray,
    proto_med: np.ndarray,
    proto_ctx: np.ndarray,
    *,
    lambda_m: float,
    alpha: float = DEFAULT_ALPHA,
    gamma_med: float = GAMMA_MED,
    gamma_ctx: float = GAMMA_CTX,
) -> float:
    """Compute F2 Significance Gravity T(Atom, Event_j)."""
    sim_med = cosine_unit_interval_normalized(v_med, proto_med)
    sim_ctx = cosine_unit_interval_normalized(v_ctx, proto_ctx)
    return choice_score(
        [(gamma_med, sim_med), (gamma_ctx, sim_ctx)],
        alpha=alpha,
        lambda_m=lambda_m,
    )


def passes_event_threshold(score: float, *, rho_event: float = DEFAULT_RHO_EVENT) -> bool:
    """Check whether T(Atom, Event_j) > ρ_event."""
    return float(score) > float(rho_event)


def _skip_prototype_update(atom: Dict[str, Any]) -> bool:
    kind = str(atom.get("kind") or "")
    tui = str(atom.get("tui") or "")
    return kind in ("summary", "follow_up", "people") or tui == "T000"


def cluster_atoms_to_events(
    atoms: List[Dict[str, Any]],
    *,
    alpha: float = DEFAULT_ALPHA,
    rho_event: float = DEFAULT_RHO_EVENT,
    gamma_med: float = GAMMA_MED,
    gamma_ctx: float = GAMMA_CTX,
    ema_old: float = EMA_OLD,
    ema_new: float = EMA_NEW,
) -> List[Dict[str, Any]]:
    """
    Online Atom -> Event aggregation via Significance Gravity T and ρ_event.

    Each input atom dict contains L2-normalized `v_med`, `v_ctx`, `lambda_i`, and `created_at`.
    Returns event prototype dicts with `atom_indices`, `v_med_proto`, `v_ctx_proto`, `last_time`,
    and `lambda_event` (associated latching max-pool).
    """
    if not atoms:
        return []

    event_protos: List[Dict[str, Any]] = []
    for ai, atom in enumerate(atoms):
        v_med = atom["v_med"]
        v_ctx = atom["v_ctx"]
        lam = float(atom["lambda_i"])
        created_at = atom["created_at"]

        best_j = -1
        best_t = -1e30
        for j, proto in enumerate(event_protos):
            score = significance_gravity(
                v_med,
                v_ctx,
                proto["v_med_proto"],
                proto["v_ctx_proto"],
                lambda_m=lam,
                alpha=alpha,
                gamma_med=gamma_med,
                gamma_ctx=gamma_ctx,
            )
            if score > best_t:
                best_t = score
                best_j = j

        if best_j >= 0 and passes_event_threshold(best_t, rho_event=rho_event):
            proto = event_protos[best_j]
            proto["atom_indices"].append(ai)
            proto["lambda_event"] = max_pool_lambda([proto["lambda_event"], lam])
            if not _skip_prototype_update(atom):
                proto["v_med_proto"] = l2_normalize(ema_old * proto["v_med_proto"] + ema_new * v_med)
                proto["v_ctx_proto"] = l2_normalize(ema_old * proto["v_ctx_proto"] + ema_new * v_ctx)
            if created_at > proto["last_time"]:
                proto["last_time"] = created_at
        else:
            event_protos.append(
                {
                    "atom_indices": [ai],
                    "v_med_proto": np.asarray(v_med, dtype=np.float64).copy(),
                    "v_ctx_proto": np.asarray(v_ctx, dtype=np.float64).copy(),
                    "last_time": created_at,
                    "lambda_event": lam,
                }
            )
    return event_protos
