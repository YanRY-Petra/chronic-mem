"""
F3 — Clinical Significance-Aware Progressive Compression (CSPC) & Physical Cleanup.

Implements the state transition side effects for the Tri-State Metabolism Pipeline:
1. `distill_episode_to_labels` (CSPC):
   - Consolidates recurrent clinical concepts (`diagnosis`, `symptom`, `summary`, `lifestyle`)
     reaching `min_frequency` (default 5) into `PermanentLabel` nodes (`λ = 99999d`).
   - Safety Anchor Rule: `T037` (allergy / hypersensitivity) atoms are distilled into
     Permanent Labels even at `frequency = 1`, preventing safety loss during compression/forgetting.
   - Optional `label_summarizer_fn` hook allows LLM-driven clinical distillation via
     `CSPC_DISTILLATION_SYSTEM_PROMPT`.
2. `prune_redundant_atoms_compressed`:
   - In `compressed` state, computes atom-level survival `f(m_i, t) = Res(m_i, v_profile) * exp(-Δt / λ_i)`
     and physically deletes atoms + cached vectors with `f(m_i, t) < τ_prune` (default 0.3).
3. `forget_episode_physically`:
   - In `forgotten` state, physically deletes all remaining `AtomNode`s, atom vectors,
     event vectors (`event:*`), and episode aggregate vectors (`episode_agg:*`),
     retaining ONLY the `PermanentLabel` index.
"""
from __future__ import annotations

from datetime import datetime
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from chronic_mem.f3.metabolism import metabolism_score, profile_resonance
from chronic_mem.types import (
    AtomNode,
    EmbeddingStore,
    EpisodeNode,
    EventNode,
    PermanentLabel,
    utc_now,
)

CSPC_DISTILLATION_SYSTEM_PROMPT = """You are the Clinical Significance-Aware Progressive Compression (CSPC) module in Chronic-Mem (Layer F3).
Given a cluster of recurring or safety-critical clinical Memory Atoms from Episode {episode_id} (UMLS TUI: {tui}),
distill them into a single concise, permanent clinical phenotype/safety label (<= 25 words) preserving drug allergies,
chronic diagnoses, and recurrence frequency."""

ELIGIBLE_DISTILL_KINDS = ("diagnosis", "symptom", "summary", "lifestyle")


def normalize_concept(text: str) -> str:
    s = str(text).lower().strip()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def distill_episode_to_labels(
    episode: EpisodeNode,
    events: Dict[int, EventNode],
    atoms: Dict[str, AtomNode],
    labels: Dict[str, PermanentLabel],
    embed: EmbeddingStore,
    encode_fn: Callable[[str], np.ndarray],
    *,
    min_frequency: int = 5,
    current_time: Optional[datetime] = None,
    label_summarizer_fn: Optional[Callable[[int, str, str, int], str]] = None,
) -> int:
    """
    CSPC Distillation: extract `PermanentLabel` nodes from an episode entering
    `compressed` or `forgotten` state, indexing label embeddings in `embed`.
    """
    now = current_time or utc_now()
    buckets: Dict[Tuple[str, str], List[str]] = {}

    for eid in episode.event_ids:
        ev = events.get(eid)
        if not ev:
            continue
        for aid in ev.atom_ids:
            atom = atoms.get(aid)
            if not atom or atom.kind not in ELIGIBLE_DISTILL_KINDS:
                continue
            concept = normalize_concept(atom.text)
            if not concept:
                continue
            buckets.setdefault((atom.tui, concept), []).append(aid)

    created = 0
    for (tui, concept), atom_ids in buckets.items():
        is_allergy_anchor = tui == "T037"
        if len(atom_ids) < min_frequency and not is_allergy_anchor:
            continue

        label_id = f"label:{episode.episode_id}:{tui}:{hash(concept) & 0xFFFFFFF}"
        if label_id in labels:
            continue

        freq = len(atom_ids)
        if label_summarizer_fn is not None:
            summary = label_summarizer_fn(episode.episode_id, tui, concept, freq)
        else:
            summary = f"Episode {episode.episode_id}: recurrent {concept} (freq={freq})"

        label = PermanentLabel(
            label_id=label_id,
            episode_id=episode.episode_id,
            tui=tui,
            summary=summary,
            main_concepts=[concept],
            frequency=freq,
            created_at=now,
            lambda_i=99999.0,
        )
        labels[label_id] = label
        episode.label_ids.append(label_id)
        embed.add(label_id, encode_fn(summary))
        created += 1

    return created


def compute_atom_profile_survival(
    atom: AtomNode,
    embed: EmbeddingStore,
    encode_fn: Callable[[str], np.ndarray],
    v_profile: Optional[np.ndarray],
    *,
    current_time: datetime,
    min_days: float = 1.0,
) -> float:
    """Compute atom survival against global patient profile: Res(m_i, v_profile) * exp(-Δt / λ_i)."""
    if v_profile is None:
        res = profile_resonance(np.zeros(1), None, fallback_phi=atom.phi)
    else:
        vec = embed.get(atom.atom_id)
        if vec is None:
            vec = encode_fn(atom.text)
            embed.add(atom.atom_id, vec)
        res = profile_resonance(vec, v_profile, fallback_phi=atom.phi)

    delta_days = (current_time - atom.created_at).total_seconds() / 86400.0
    lam = max(float(atom.lambda_i), float(min_days))
    return metabolism_score(res, delta_days, lam)


def prune_redundant_atoms_compressed(
    episode: EpisodeNode,
    events: Dict[int, EventNode],
    atoms: Dict[str, AtomNode],
    embed: EmbeddingStore,
    encode_fn: Callable[[str], np.ndarray],
    v_profile: Optional[np.ndarray],
    *,
    prune_threshold: float = 0.3,
    min_days: float = 1.0,
    current_time: Optional[datetime] = None,
) -> Dict[str, int]:
    """
    Compressed-state physical cleanup: remove redundant/low-survival Memory Atoms
    (`f(m_i, t) < prune_threshold`) from both node table and vector store.
    """
    now = current_time or utc_now()
    atom_ids_to_delete: List[str] = []

    for eid in episode.event_ids:
        ev = events.get(eid)
        if not ev:
            continue
        for aid in ev.atom_ids:
            atom = atoms.get(aid)
            if not atom:
                continue
            score = compute_atom_profile_survival(
                atom,
                embed,
                encode_fn,
                v_profile,
                current_time=now,
                min_days=min_days,
            )
            if score < prune_threshold:
                atom_ids_to_delete.append(aid)

    delete_set = set(atom_ids_to_delete)
    embed.delete_many(delete_set)
    for aid in delete_set:
        atoms.pop(aid, None)

    for eid in episode.event_ids:
        ev = events.get(eid)
        if ev:
            ev.atom_ids = [aid for aid in ev.atom_ids if aid not in delete_set]

    return {
        "atoms_deleted": len(delete_set),
        "vectors_deleted": len(delete_set),
    }


def forget_episode_physically(
    episode: EpisodeNode,
    events: Dict[int, EventNode],
    atoms: Dict[str, AtomNode],
    embed: EmbeddingStore,
) -> Dict[str, int]:
    """
    Forgotten-state physical cleanup: delete all remaining AtomNodes, atom vectors,
    event vectors, and episode aggregate vectors; retain ONLY PermanentLabel index.
    """
    atom_ids_to_delete: List[str] = []
    event_vec_keys_to_delete: List[str] = []

    for eid in episode.event_ids:
        ev = events.get(eid)
        if not ev:
            continue
        atom_ids_to_delete.extend([aid for aid in ev.atom_ids if aid in atoms])
        event_vec_keys_to_delete.append(f"event:{eid}")

    embed.delete_many(atom_ids_to_delete)
    for aid in atom_ids_to_delete:
        atoms.pop(aid, None)

    embed.delete_many(event_vec_keys_to_delete)
    embed.delete(f"episode_agg:{episode.episode_id}")

    for eid in episode.event_ids:
        ev = events.get(eid)
        if ev:
            ev.atom_ids = []

    vectors_deleted = len(atom_ids_to_delete) + len(event_vec_keys_to_delete)
    return {
        "atoms_deleted": len(atom_ids_to_delete),
        "events_deleted": 0,
        "vectors_deleted": vectors_deleted,
    }
