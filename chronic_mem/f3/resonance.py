"""
F3 — Cross-Time Episode Resonance, Spatio-Temporal Wake-up, and EMA Prototype Update.

Cross-time semantic resonance between incoming event e_j and episode E_p:

    R(e_j, E_p) = γ_med · Sim(v_med, w_med) + γ_ctx · Sim(v_ctx, w_ctx)

Time elapsed is deliberately excluded from R so clinically coherent trajectories
(e.g., an asthma relapse years later) can reconnect across arbitrary intervals.

When R(e_j, E_p) > ρ_episode (default 0.65):
1. Spatio-Temporal Wake-up: E_p's `reference_time` (`awake_time`) is reset to the
   current clock `t`, resetting Δt = t - reference_time -> 0 in f(E_p, t).
2. EMA Prototype Update: w_p <- normalize((1 - β) · w_p + β · v_{e_j}) with β = 0.1.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chronic_mem.geometry import cosine_unit_interval_normalized, l2_normalize
from chronic_mem.types import EpisodeNode

GAMMA_MED = 0.8
GAMMA_CTX = 0.2
EMA_OLD = 0.9
EMA_NEW = 0.1
DEFAULT_RHO_EPISODE = 0.65


def resonance(
    v_med: np.ndarray,
    v_ctx: np.ndarray,
    w_med: np.ndarray,
    w_ctx: np.ndarray,
    *,
    gamma_med: float = GAMMA_MED,
    gamma_ctx: float = GAMMA_CTX,
) -> float:
    """Compute cross-time semantic resonance R(e_j, E_p) in [0, 1]."""
    sim_med = cosine_unit_interval_normalized(v_med, w_med)
    sim_ctx = cosine_unit_interval_normalized(v_ctx, w_ctx)
    return float(gamma_med * sim_med + gamma_ctx * sim_ctx)


def ema_prototype(
    old: np.ndarray,
    new: np.ndarray,
    *,
    ema_old: float = EMA_OLD,
    ema_new: float = EMA_NEW,
) -> np.ndarray:
    """L2-normalized Exponential Moving Average (EMA) prototype update."""
    return l2_normalize(ema_old * np.asarray(old, dtype=np.float64) + ema_new * np.asarray(new, dtype=np.float64))


def awaken_episode(episode: EpisodeNode, awaken_time: datetime) -> float:
    """
    Spatio-Temporal Wake-up: reset `episode.reference_time` to `awaken_time`
    so elapsed decay time Δt = (awaken_time - reference_time) resets to 0.0 days.
    """
    episode.reference_time = awaken_time
    if episode.state == "compressed":
        episode.state = "active"
    return (awaken_time - episode.reference_time).total_seconds() / 86400.0


def cluster_events_to_episodes(
    event_protos: List[Dict[str, Any]],
    *,
    awaken_to: Optional[datetime] = None,
    rho_episode: float = DEFAULT_RHO_EPISODE,
    gamma_med: float = GAMMA_MED,
    gamma_ctx: float = GAMMA_CTX,
    ema_old: float = EMA_OLD,
    ema_new: float = EMA_NEW,
) -> Tuple[List[Dict[str, Any]], Dict[int, int], List[int]]:
    """
    Align Event prototypes into longitudinal Episodes via cross-time Resonance R,
    triggering spatio-temporal wake-up (`awake_time`) and EMA prototype updates.
    """
    if not event_protos:
        return [], {}, []

    event_order = sorted(range(len(event_protos)), key=lambda i: (event_protos[i]["last_time"], i))
    episode_protos: List[Dict[str, Any]] = []
    event_to_episode: Dict[int, int] = {}

    for ei in event_order:
        evp = event_protos[ei]
        v_med = evp["v_med_proto"]
        v_ctx = evp["v_ctx_proto"]
        current_awake = awaken_to or evp["last_time"]

        best_ep = -1
        best_r = -1.0
        for epi, ep in enumerate(episode_protos):
            score = resonance(
                v_med,
                v_ctx,
                ep["w_med_proto"],
                ep["w_ctx_proto"],
                gamma_med=gamma_med,
                gamma_ctx=gamma_ctx,
            )
            if score > best_r:
                best_r = score
                best_ep = epi

        if best_ep >= 0 and best_r > rho_episode:
            ep = episode_protos[best_ep]
            ep["event_indices"].append(ei)
            # Spatio-temporal wake-up: reset episode reference clock so Δt -> 0
            ep["awake_time"] = current_awake
            ep["w_med_proto"] = ema_prototype(ep["w_med_proto"], v_med, ema_old=ema_old, ema_new=ema_new)
            ep["w_ctx_proto"] = ema_prototype(ep["w_ctx_proto"], v_ctx, ema_old=ema_old, ema_new=ema_new)
            event_to_episode[ei] = best_ep
        else:
            episode_protos.append(
                {
                    "event_indices": [ei],
                    "w_med_proto": np.asarray(v_med, dtype=np.float64).copy(),
                    "w_ctx_proto": np.asarray(v_ctx, dtype=np.float64).copy(),
                    "awake_time": evp["last_time"],
                }
            )
            event_to_episode[ei] = len(episode_protos) - 1

    return episode_protos, event_to_episode, event_order
