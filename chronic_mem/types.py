"""Core data structures for Chronic-Mem (Atom, Event, Episode, PermanentLabel, EmbeddingStore)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class EmbeddingStore:
    """
    In-memory vector index with explicit physical deletion (`delete` / `delete_many`)
    to release vector space during F3 Compressed / Forgotten transitions.
    """

    vectors: Dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, key: str, vec: np.ndarray) -> None:
        self.vectors[key] = np.asarray(vec, dtype=np.float32)

    def get(self, key: str) -> Optional[np.ndarray]:
        return self.vectors.get(key)

    def delete(self, key: str) -> None:
        self.vectors.pop(key, None)

    def delete_many(self, keys: Iterable[str]) -> None:
        for k in keys:
            self.vectors.pop(k, None)

    def stats(self) -> Dict[str, float]:
        n = len(self.vectors)
        total = sum(int(v.nbytes) for v in self.vectors.values())
        return {
            "vector_count": float(n),
            "approx_bytes": float(total),
            "approx_mb": float(total) / (1024.0 * 1024.0),
        }


@dataclass
class AtomNode:
    """F1 atomic clinical memory unit m_i."""

    atom_id: str
    episode_id: int
    event_id: int
    kind: str  # diagnosis | symptom | treatment | medication | lifestyle | people | follow_up | summary | social | raw_dialogue
    text: str
    tui: str
    phi: float
    lambda_i: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0


@dataclass
class EventNode:
    """F2 aggregated clinical visit/event e_j with max-pooled lifetime lambda_event."""

    event_id: int
    episode_id: int
    event: Dict[str, Any]
    atom_ids: List[str]
    lambda_event: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0


@dataclass
class PermanentLabel:
    """F3 CSPC distilled semantic anchor retained across Compressed and Forgotten states."""

    label_id: str
    episode_id: int
    tui: str
    summary: str
    main_concepts: List[str]
    frequency: int
    created_at: datetime
    lambda_i: float = 99999.0


@dataclass
class EpisodeNode:
    """F3 longitudinal clinical episode E_p with tri-state lifecycle."""

    episode_id: int
    event_ids: List[int] = field(default_factory=list)
    label_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    # Latest clinical activity or F3 resonance wake-up clock (used for Δt = t - reference_time)
    reference_time: Optional[datetime] = None
    last_metabolized_at: datetime = field(default_factory=utc_now)
    state: str = "active"  # active | compressed | forgotten
    w_med_proto: Optional[np.ndarray] = None
    w_ctx_proto: Optional[np.ndarray] = None
