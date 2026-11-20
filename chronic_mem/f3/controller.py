"""
F3 — Adaptive Retrieval Granularity Controller.

Maps an Episode's current Tri-State lifecycle (`active`, `compressed`, `forgotten`)
to the permitted retrieval granularity and index access policy:

- `active`     -> `GRANULARITY_ATOMS`:     Full fine-grained Memory Atoms + Event headers
- `compressed` -> `GRANULARITY_SUMMARIES`: Distilled `PermanentLabel`s + Top-k Event summaries
- `forgotten`  -> `GRANULARITY_LABELS`:    `PermanentLabel` index ONLY (event/atom vectors remain unallocated)
"""
from __future__ import annotations

from dataclasses import dataclass

GRANULARITY_ATOMS = "atoms"
GRANULARITY_SUMMARIES = "summaries"
GRANULARITY_LABELS = "labels"


def retrieval_granularity(state: str) -> str:
    """Return adaptive retrieval granularity (`atoms` | `summaries` | `labels`) for `state`."""
    if state == "forgotten":
        return GRANULARITY_LABELS
    if state == "compressed":
        return GRANULARITY_SUMMARIES
    return GRANULARITY_ATOMS


@dataclass(frozen=True)
class RetrievalPolicy:
    """Resolved retrieval plan emitted by `AdaptiveRetrievalController`."""

    state: str
    granularity: str
    use_atom_vectors: bool
    use_event_vectors: bool
    use_label_vectors: bool
    k_events: int
    k_atoms_per_event: int


class AdaptiveRetrievalController:
    """Selects context representation and vector-store access permissions per Episode state."""

    def __init__(self, *, default_k_events: int = 10, default_k_atoms_per_event: int = 20):
        self.default_k_events = int(default_k_events)
        self.default_k_atoms_per_event = int(default_k_atoms_per_event)

    def resolve(
        self,
        state: str,
        *,
        k_events: int | None = None,
        k_atoms_per_event: int | None = None,
    ) -> RetrievalPolicy:
        gran = retrieval_granularity(state)
        ke = self.default_k_events if k_events is None else int(k_events)
        ka = self.default_k_atoms_per_event if k_atoms_per_event is None else int(k_atoms_per_event)

        if gran == GRANULARITY_LABELS:
            return RetrievalPolicy(
                state=state,
                granularity=gran,
                use_atom_vectors=False,
                use_event_vectors=False,
                use_label_vectors=True,
                k_events=0,
                k_atoms_per_event=0,
            )
        if gran == GRANULARITY_SUMMARIES:
            return RetrievalPolicy(
                state=state,
                granularity=gran,
                use_atom_vectors=False,
                use_event_vectors=True,
                use_label_vectors=True,
                k_events=ke,
                k_atoms_per_event=0,
            )
        return RetrievalPolicy(
            state=state,
            granularity=gran,
            use_atom_vectors=True,
            use_event_vectors=True,
            use_label_vectors=True,
            k_events=ke,
            k_atoms_per_event=ka,
        )
