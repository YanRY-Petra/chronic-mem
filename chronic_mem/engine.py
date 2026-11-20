"""
Chronic-Mem Three-Layer Memory Metabolism Engine (Section 3: Atom -> Event -> Episode).

Orchestrates the three modular layers:
- Layer F1 (`chronic_mem.f1`): Semantic Triage & Non-Homogeneous λ_i Initialization
- Layer F2 (`chronic_mem.f2`): Significance Gravity T(Atom, Event_j) & Associated Latching (Max-Pooling)
- Layer F3 (`chronic_mem.f3`): Cross-Time Episode Resonance R(e_j, E_p), Spatio-Temporal Wake-up (Δt -> 0),
  Profile-Guided Metabolism f(E_p, t), Tri-State CSPC Compression/Cleanup, and Adaptive Retrieval Granularity
"""
from __future__ import annotations

from datetime import datetime
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from chronic_mem.encoding import bert_encode, parse_events_json
from chronic_mem.f1.atoms import (
    atomize_event,
    atomize_incremental_lines,
    extract_atoms_from_dialogue,
)
from chronic_mem.f1.context import ClinicalContext
from chronic_mem.f2.latching import max_pool_lambda
from chronic_mem.f2.significance import DEFAULT_ALPHA, DEFAULT_RHO_EVENT, cluster_atoms_to_events
from chronic_mem.f3.controller import (
    GRANULARITY_LABELS,
    GRANULARITY_SUMMARIES,
    AdaptiveRetrievalController,
)
from chronic_mem.f3.cspc import (
    compute_atom_profile_survival,
    distill_episode_to_labels,
    forget_episode_physically,
    prune_redundant_atoms_compressed,
)
from chronic_mem.f3.metabolism import (
    COMPRESSED,
    DEFAULT_ACTIVE_THRESHOLD,
    DEFAULT_COMPRESS_THRESHOLD,
    DEFAULT_PRUNE_THRESHOLD,
    FORGOTTEN,
    metabolism_score,
    phi_salience_fallback,
    profile_resonance,
    tri_state,
)
from chronic_mem.f3.resonance import (
    DEFAULT_RHO_EPISODE,
    awaken_episode,
    cluster_events_to_episodes,
    ema_prototype,
    resonance,
)
from chronic_mem.geometry import cosine_unit_interval, l2_normalize
from chronic_mem.types import (
    AtomNode,
    EmbeddingStore,
    EpisodeNode,
    EventNode,
    PermanentLabel,
    utc_now,
)


def _safe_parse_yyyy_mm_dd(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = str(date_str).strip()
    m = re.match(r"^(\d{4})[-/](\d{2})[-/](\d{2})", s)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


class ChronicMemEngine:
    """
    Three-Layer Clinical Memory Metabolism Engine (Atom -> Event -> Episode).
    Also exported as `CSAFHierarchicalMetabolizer` for backward compatibility.
    """

    def __init__(
        self,
        csaf: Optional[ClinicalContext] = None,
        *,
        patient_profile_text: str = "",
        active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
        compress_threshold: float = DEFAULT_COMPRESS_THRESHOLD,
        min_frequency: int = 5,
        survival_threshold_atom: float = 0.1,
        survival_prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
        metabolism_interval_rounds: int = 5,
        eager_atom_embedding: bool = False,
        eager_event_embedding: bool = False,
        choice_alpha: float = DEFAULT_ALPHA,
        choice_event_min_score: float = DEFAULT_RHO_EVENT,
        episode_resonance_vigilance_rho_raw: float = DEFAULT_RHO_EPISODE,
        f1_homogeneous_lambda_days: Optional[float] = None,
        f2_disabled_strict: bool = False,
        encode_fn: Callable[[str], np.ndarray] = bert_encode,
    ):
        self.csaf = csaf or ClinicalContext()
        self.encode_fn = encode_fn
        self.patient_profile_text = patient_profile_text
        self._profile_vector_cache: Optional[np.ndarray] = None

        self.active_threshold = float(active_threshold)
        self.compress_threshold = float(compress_threshold)
        self.min_frequency = int(min_frequency)
        self.survival_threshold_atom = float(survival_threshold_atom)
        self.survival_prune_threshold = float(survival_prune_threshold)
        self.metabolism_interval_rounds = max(1, int(metabolism_interval_rounds))
        self._dialog_round: int = 0

        self.eager_atom_embedding = bool(eager_atom_embedding)
        self.eager_event_embedding = bool(eager_event_embedding)

        self.choice_alpha = float(choice_alpha)
        self.choice_event_min_score = float(choice_event_min_score)
        self.episode_resonance_vigilance_rho_raw = float(episode_resonance_vigilance_rho_raw)

        self.f1_homogeneous_lambda_days = (
            float(f1_homogeneous_lambda_days)
            if f1_homogeneous_lambda_days is not None and float(f1_homogeneous_lambda_days) > 0
            else None
        )
        self.f2_disabled_strict = bool(f2_disabled_strict)

        self.retrieval_controller = AdaptiveRetrievalController()
        self.embed = EmbeddingStore()

        self.atoms: Dict[str, AtomNode] = {}
        self.events: Dict[int, EventNode] = {}
        self.labels: Dict[str, PermanentLabel] = {}
        self.episodes: Dict[int, EpisodeNode] = {}

    # ---------------------------
    # Global Patient Profile v_profile
    # ---------------------------

    def set_patient_profile(self, text: str) -> None:
        """Set global patient profile text used to build v_profile in F3 metabolism."""
        self.patient_profile_text = text or ""
        self._profile_vector_cache = None

    def _get_profile_vector(self) -> Optional[np.ndarray]:
        t = (self.patient_profile_text or "").strip()
        if not t:
            return None
        if self._profile_vector_cache is None:
            self._profile_vector_cache = self.encode_fn(t)
        return self._profile_vector_cache

    def _get_or_build_episode_aggregate_embedding(self, episode_id: int) -> np.ndarray:
        key = f"episode_agg:{episode_id}"
        cached = self.embed.get(key)
        if cached is not None:
            return cached
        ep = self.episodes[episode_id]
        parts: List[str] = []
        for eid in ep.event_ids:
            ev = self.events.get(eid)
            if ev:
                parts.append(self._event_text(ev.event))
        blob = "\n".join(parts) if parts else " "
        vec = self.encode_fn(blob)
        self.embed.add(key, vec)
        return vec

    def _episode_max_phi(self, episode_id: int) -> float:
        ep = self.episodes[episode_id]
        max_phi = 1.0
        for eid in ep.event_ids:
            ev = self.events.get(eid)
            if not ev:
                continue
            for aid in ev.atom_ids:
                atom = self.atoms.get(aid)
                if atom:
                    max_phi = max(max_phi, float(atom.phi))
        return max_phi

    def _episode_resonance_profile(self, episode_id: int) -> float:
        """Res(E_p, v_profile) = cos(v_{E_p}, v_profile) mapped to [0, 1]."""
        vp = self._get_profile_vector()
        if vp is None:
            return phi_salience_fallback(self._episode_max_phi(episode_id))
        ve = self._get_or_build_episode_aggregate_embedding(episode_id)
        return profile_resonance(ve, vp, fallback_phi=self._episode_max_phi(episode_id))

    # ---------------------------
    # Layer F1 & F2 Construction
    # ---------------------------

    def _compute_lambda_for_atom(self, *, tui: str, phi: float, context_text: str, kind: str) -> float:
        _, _, lam = self.csaf.triage_atom(
            context_text,
            kind,
            homogeneous_lambda_days=self.f1_homogeneous_lambda_days,
        )
        return lam

    def _f1_create_atoms(
        self,
        *,
        event_id: int,
        episode_id: int,
        event: Dict[str, Any],
        created_at: datetime,
    ) -> List[str]:
        """F1 Semantic Triage: decompose event into Memory Atoms and assign `(tui, phi, lambda_i)`."""
        now = utc_now()
        atom_ids: List[str] = []
        is_perm = bool(event.get("is_permanent", False))

        for idx, (kind, text) in enumerate(atomize_event(event)):
            atom_id = f"atom:{event_id}:{kind}:{idx}"
            tui, phi, lambda_i = self.csaf.triage_atom(
                text,
                kind,
                is_event_permanent=is_perm,
                homogeneous_lambda_days=self.f1_homogeneous_lambda_days,
            )
            node = AtomNode(
                atom_id=atom_id,
                episode_id=episode_id,
                event_id=event_id,
                kind=kind,
                text=text,
                tui=tui,
                phi=phi,
                lambda_i=float(lambda_i),
                created_at=created_at,
                last_accessed=now,
                access_count=0,
            )
            self.atoms[atom_id] = node
            atom_ids.append(atom_id)

            if self.eager_atom_embedding:
                self.embed.add(atom_id, self.encode_fn(text))

        return atom_ids

    def _f2_compute_lambda_event(self, atom_ids: Sequence[str]) -> float:
        """F2 Associated Latching: λ_Event = max(λ_{m_1}, ..., λ_{m_n})."""
        if self.f2_disabled_strict:
            return float(self.csaf.min_days)
        values = [self.atoms[aid].lambda_i for aid in atom_ids if aid in self.atoms]
        return max_pool_lambda(values, floor=float(self.csaf.min_days))

    def load_events(self, events_path: str) -> None:
        """Build Atom -> Event -> Episode hierarchy from `events.json` grouped by `record_index`."""
        flat_events = parse_events_json(events_path)
        next_event_id = 0
        for ev in flat_events:
            episode_id = int(ev.get("record_index", -1))
            if episode_id < 0:
                episode_id = 0

            if episode_id not in self.episodes:
                self.episodes[episode_id] = EpisodeNode(episode_id=episode_id)

            event_time = _safe_parse_yyyy_mm_dd(ev.get("date")) or utc_now()
            atom_ids = self._f1_create_atoms(
                event_id=next_event_id,
                episode_id=episode_id,
                event=ev,
                created_at=event_time,
            )

            if self.f2_disabled_strict:
                for atom_id in atom_ids:
                    atom = self.atoms.get(atom_id)
                    if atom is None:
                        continue
                    ev_dict = {"summary": atom.text, "raw_dialogue": [atom.text], "date": ev.get("date")}
                    self.events[next_event_id] = EventNode(
                        event_id=next_event_id,
                        episode_id=episode_id,
                        event=ev_dict,
                        atom_ids=[atom_id],
                        lambda_event=float(self.csaf.min_days),
                        created_at=atom.created_at,
                        last_accessed=utc_now(),
                    )
                    self.episodes[episode_id].event_ids.append(next_event_id)
                    if self.eager_event_embedding:
                        self.embed.add(f"event:{next_event_id}", self.encode_fn(self._event_text(ev_dict)))
                    next_event_id += 1
            else:
                lambda_event = self._f2_compute_lambda_event(atom_ids)
                self.events[next_event_id] = EventNode(
                    event_id=next_event_id,
                    episode_id=episode_id,
                    event=ev,
                    atom_ids=atom_ids,
                    lambda_event=lambda_event,
                    created_at=event_time,
                    last_accessed=utc_now(),
                )
                self.episodes[episode_id].event_ids.append(next_event_id)
                if self.eager_event_embedding:
                    self.embed.add(f"event:{next_event_id}", self.encode_fn(self._event_text(ev)))
                next_event_id += 1

        for ep in self.episodes.values():
            times = [self.events[eid].created_at for eid in ep.event_ids if eid in self.events]
            if times:
                ep.created_at = min(times)
                ep.reference_time = max(times)
            else:
                ep.reference_time = ep.created_at

    # ---------------------------
    # Full F1 -> F2 -> F3 Spatio-Semantic Hierarchy Construction
    # ---------------------------

    def _extract_atoms_from_flat_events(self, flat_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not flat_events:
            return []

        ordered: List[Tuple[datetime, int, Dict[str, Any]]] = []
        for i, ev in enumerate(flat_events):
            t = _safe_parse_yyyy_mm_dd(ev.get("date")) or utc_now()
            ordered.append((t, i, ev))
        ordered.sort(key=lambda x: (x[0], x[1]))

        out: List[Dict[str, Any]] = []
        for created_at, _i, ev in ordered:
            is_perm = bool(ev.get("is_permanent", False))
            for kind, text in atomize_event(ev):
                text_s = str(text).strip()
                if not text_s:
                    continue

                tui, phi, lambda_i = self.csaf.triage_atom(
                    text_s,
                    kind,
                    is_event_permanent=is_perm,
                    homogeneous_lambda_days=self.f1_homogeneous_lambda_days,
                )
                v_ctx = l2_normalize(np.asarray(self.encode_fn(text_s), dtype=np.float64))
                v_med_t = self.csaf.umls_mapper.get_embedding(tui)
                v_med_raw = v_med_t.detach().cpu().numpy() if hasattr(v_med_t, "detach") else np.asarray(v_med_t)
                v_med = l2_normalize(np.asarray(v_med_raw, dtype=np.float64))

                out.append(
                    {
                        "kind": kind,
                        "text": text_s,
                        "tui": tui,
                        "phi": float(phi),
                        "lambda_i": float(lambda_i),
                        "created_at": created_at,
                        "v_ctx": v_ctx,
                        "v_med": v_med,
                    }
                )
        return out

    def _build_event_dict_from_atom_indices(
        self,
        atoms: List[Dict[str, Any]],
        atom_indices: List[int],
    ) -> Dict[str, Any]:
        raw_dialogue: List[str] = []
        buckets: Dict[str, List[str]] = {
            "diagnosis": [],
            "symptoms": [],
            "treatment": [],
            "medications": [],
            "lifestyle": [],
            "people": [],
            "follow_up": [],
            "summary": [],
        }
        kind_to_bucket = {
            "diagnosis": "diagnosis",
            "symptom": "symptoms",
            "treatment": "treatment",
            "medication": "medications",
            "lifestyle": "lifestyle",
            "people": "people",
            "follow_up": "follow_up",
            "summary": "summary",
        }

        for ai in atom_indices:
            a = atoms[ai]
            kind = a["kind"]
            text = a["text"]
            raw_dialogue.append(text)
            if kind in kind_to_bucket:
                buckets[kind_to_bucket[kind]].append(text)

        def _dedup(xs: List[str]) -> List[str]:
            seen: Set[str] = set()
            res: List[str] = []
            for x in xs:
                sx = str(x).strip()
                if sx and sx not in seen:
                    seen.add(sx)
                    res.append(sx)
            return res

        ev: Dict[str, Any] = {"raw_dialogue": raw_dialogue}
        dx = _dedup(buckets["diagnosis"])
        if dx:
            ev["diagnosis"] = "\n".join(dx[:3])
        for key in ("symptoms", "treatment", "medications", "lifestyle", "people", "follow_up"):
            deduped = _dedup(buckets[key])
            if deduped:
                ev[key] = deduped
        summaries = _dedup(buckets["summary"])
        if summaries:
            ev["summary"] = "\n".join(summaries)[:800]
        return ev

    def load_events_clustered(
        self,
        events_path: str,
        *,
        awaken_to: Optional[datetime] = None,
    ) -> None:
        """
        Build hierarchy via dynamic F1 -> F2 -> F3 clustering:
        1. F1 Semantic Triage -> typed Memory Atoms + non-homogeneous λ_i
        2. F2 Significance Gravity T(Atom, Event_j) > ρ_event + Max-Pool λ_Event
        3. F3 Episode Resonance R(e_j, E_p) > ρ_episode + Spatio-Temporal Wake-up + EMA prototype
        """
        flat_events = parse_events_json(events_path)
        wake_clock = awaken_to or utc_now()

        atoms = self._extract_atoms_from_flat_events(flat_events)
        event_protos = cluster_atoms_to_events(
            atoms,
            alpha=self.choice_alpha,
            rho_event=self.choice_event_min_score,
        )
        episode_protos, event_to_episode, event_order = cluster_events_to_episodes(
            event_protos,
            awaken_to=wake_clock,
            rho_episode=self.episode_resonance_vigilance_rho_raw,
        )
        if not atoms or not event_protos or not episode_protos:
            return

        self.atoms.clear()
        self.events.clear()
        self.episodes.clear()
        self.labels.clear()
        self.embed.vectors.clear()

        now = utc_now()
        for ep_id, ep_proto in enumerate(episode_protos):
            event_indices: List[int] = ep_proto["event_indices"]
            times = [event_protos[ei]["last_time"] for ei in event_indices if 0 <= ei < len(event_protos)]
            created_at = min(times) if times else now
            reference_time = ep_proto.get("awake_time", created_at)
            self.episodes[ep_id] = EpisodeNode(
                episode_id=ep_id,
                created_at=created_at,
                reference_time=reference_time,
                last_metabolized_at=now,
                state="active",
                w_med_proto=ep_proto["w_med_proto"].copy(),
                w_ctx_proto=ep_proto["w_ctx_proto"].copy(),
            )

        next_event_id = 0
        for event_idx in event_order:
            episode_id = int(event_to_episode[event_idx])
            evp = event_protos[event_idx]
            event_time: datetime = evp["last_time"]
            atom_indices: List[int] = evp["atom_indices"]
            ev_dict = self._build_event_dict_from_atom_indices(atoms, atom_indices)
            ev_dict["date"] = event_time.strftime("%Y-%m-%d")

            atom_ids: List[str] = []
            for local_idx, atom_global_idx in enumerate(atom_indices):
                a = atoms[atom_global_idx]
                atom_id = f"atom:{next_event_id}:{a['kind']}:{local_idx}"
                self.atoms[atom_id] = AtomNode(
                    atom_id=atom_id,
                    episode_id=episode_id,
                    event_id=next_event_id,
                    kind=str(a["kind"]),
                    text=str(a["text"]),
                    tui=str(a["tui"]),
                    phi=float(a["phi"]),
                    lambda_i=float(a["lambda_i"]),
                    created_at=a["created_at"],
                    last_accessed=now,
                )
                atom_ids.append(atom_id)
                if self.eager_atom_embedding:
                    self.embed.add(atom_id, a["v_ctx"])

            lambda_event = self._f2_compute_lambda_event(atom_ids)
            self.events[next_event_id] = EventNode(
                event_id=next_event_id,
                episode_id=episode_id,
                event=ev_dict,
                atom_ids=atom_ids,
                lambda_event=lambda_event,
                created_at=event_time,
                last_accessed=now,
            )
            self.episodes[episode_id].event_ids.append(next_event_id)
            if self.eager_event_embedding:
                self.embed.add(f"event:{next_event_id}", self.encode_fn(self._event_text(ev_dict)))
            next_event_id += 1

    # ---------------------------
    # Online Ingestion & Spatio-Temporal Wake-up
    # ---------------------------

    def _next_event_id(self) -> int:
        return (max(self.events.keys()) + 1) if self.events else 0

    def _next_episode_id(self) -> int:
        return (max(self.episodes.keys()) + 1) if self.episodes else 0

    def _refresh_episode_times(self, episode_id: int, *, awaken_time: Optional[datetime] = None) -> None:
        ep = self.episodes.get(episode_id)
        if not ep:
            return
        times = [self.events[eid].created_at for eid in ep.event_ids if eid in self.events]
        if times:
            ep.created_at = min(times)
            ep.reference_time = awaken_time or max(times)
        else:
            ep.reference_time = awaken_time or ep.created_at
        self.embed.delete(f"episode_agg:{episode_id}")

    def _event_med_ctx_vectors(self, event: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        tui = self.csaf.umls_mapper.extract_tui_from_event(event)
        v_med_t = self.csaf.umls_mapper.get_embedding(tui)
        v_med_raw = v_med_t.detach().cpu().numpy() if hasattr(v_med_t, "detach") else np.asarray(v_med_t)
        v_med = l2_normalize(np.asarray(v_med_raw, dtype=np.float64))
        v_ctx = l2_normalize(np.asarray(self.encode_fn(self._event_text(event) or " "), dtype=np.float64))
        return v_med, v_ctx

    def append_event_to_episode(
        self,
        episode_id: int,
        event: Dict[str, Any],
        *,
        created_at: Optional[datetime] = None,
    ) -> int:
        """Append a clinical event to `episode_id` (F1 triage + F2 associated latching)."""
        if episode_id not in self.episodes:
            self.episodes[episode_id] = EpisodeNode(episode_id=episode_id)

        event_id = self._next_event_id()
        event_time = created_at or _safe_parse_yyyy_mm_dd(event.get("date")) or utc_now()

        atom_ids = self._f1_create_atoms(
            event_id=event_id,
            episode_id=episode_id,
            event=event,
            created_at=event_time,
        )
        lambda_event = self._f2_compute_lambda_event(atom_ids)

        self.events[event_id] = EventNode(
            event_id=event_id,
            episode_id=episode_id,
            event=event,
            atom_ids=atom_ids,
            lambda_event=lambda_event,
            created_at=event_time,
            last_accessed=utc_now(),
        )
        ep = self.episodes[episode_id]
        ep.event_ids.append(event_id)

        v_med, v_ctx = self._event_med_ctx_vectors(event)
        if ep.w_med_proto is None or ep.w_ctx_proto is None:
            ep.w_med_proto = v_med
            ep.w_ctx_proto = v_ctx
        else:
            ep.w_med_proto = ema_prototype(ep.w_med_proto, v_med)
            ep.w_ctx_proto = ema_prototype(ep.w_ctx_proto, v_ctx)

        if self.eager_event_embedding:
            self.embed.add(f"event:{event_id}", self.encode_fn(self._event_text(event)))
        else:
            self.embed.delete(f"event:{event_id}")

        self._refresh_episode_times(episode_id, awaken_time=event_time)
        return event_id

    def ingest_dialogue_resonant(
        self,
        dialogue_lines: Sequence[str],
        *,
        current_time: Optional[datetime] = None,
        llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end online ingestion of a raw dialogue turn across F1 -> F2 -> F3:
        1. F1: Extract Memory Atoms from dialogue & assign `(tui, phi, lambda_i)`
        2. F2: Aggregate into event & compute associated latching `λ_Event = max(λ_i)`
        3. F3: Compute cross-time resonance `R(e_j, E_p)` against existing episodes;
               if `R > ρ_episode`, trigger Spatio-Temporal Wake-up (`Δt -> 0`) and EMA prototype update.
        """
        now = current_time or utc_now()
        pairs = extract_atoms_from_dialogue(dialogue_lines, llm_fn=llm_fn)
        atom_dicts: List[Dict[str, Any]] = []
        for idx, (kind, text) in enumerate(pairs):
            atom_dicts.append({"kind": kind, "text": text})
        ev_dict = self._build_event_dict_from_atom_indices(atom_dicts, list(range(len(atom_dicts))))
        ev_dict["raw_dialogue"] = [str(x) for x in dialogue_lines if str(x).strip()]
        ev_dict["date"] = now.strftime("%Y-%m-%d")

        v_med, v_ctx = self._event_med_ctx_vectors(ev_dict)
        best_ep_id: Optional[int] = None
        best_r = -1.0
        for ep_id, ep in self.episodes.items():
            if ep.w_med_proto is None or ep.w_ctx_proto is None:
                # Build lazy episode prototype from its existing events if needed
                if ep.event_ids and ep.event_ids[-1] in self.events:
                    ep.w_med_proto, ep.w_ctx_proto = self._event_med_ctx_vectors(self.events[ep.event_ids[-1]].event)
                else:
                    continue
            r_score = resonance(v_med, v_ctx, ep.w_med_proto, ep.w_ctx_proto)
            if r_score > best_r:
                best_r = r_score
                best_ep_id = ep_id

        awakened = False
        if best_ep_id is not None and best_r > self.episode_resonance_vigilance_rho_raw:
            target_ep_id = best_ep_id
            event_id = self.append_event_to_episode(target_ep_id, ev_dict, created_at=now)
            awaken_episode(self.episodes[target_ep_id], now)
            awakened = True
        else:
            target_ep_id = self._next_episode_id()
            event_id = self.append_event_to_episode(target_ep_id, ev_dict, created_at=now)

        return {
            "episode_id": target_ep_id,
            "event_id": event_id,
            "resonance_score": float(max(best_r, 0.0)),
            "awakened": awakened,
            "lambda_event": self.events[event_id].lambda_event,
        }

    def merge_dialogue_into_last_event(
        self,
        episode_id: int,
        *,
        extra_raw_lines: List[str],
        created_at: Optional[datetime] = None,
    ) -> Optional[int]:
        """Append incremental dialogue turns into an ongoing clinical event and re-latch λ_Event."""
        ep = self.episodes.get(episode_id)
        if not ep or not ep.event_ids:
            return None
        event_id = ep.event_ids[-1]
        ev_node = self.events.get(event_id)
        if not ev_node:
            return None

        ev = ev_node.event
        raw = list(ev.get("raw_dialogue") or [])
        raw.extend([str(x) for x in extra_raw_lines if x])
        ev["raw_dialogue"] = raw

        event_time = created_at or ev_node.created_at
        new_atom_ids: List[str] = []
        base_idx = len(ev_node.atom_ids)
        for idx, (kind, text) in enumerate(atomize_incremental_lines(extra_raw_lines)):
            atom_id = f"atom:{event_id}:{kind}:{base_idx + idx}"
            tui, phi, lambda_i = self.csaf.triage_atom(
                text,
                kind,
                homogeneous_lambda_days=self.f1_homogeneous_lambda_days,
            )
            self.atoms[atom_id] = AtomNode(
                atom_id=atom_id,
                episode_id=episode_id,
                event_id=event_id,
                kind=kind,
                text=text,
                tui=tui,
                phi=phi,
                lambda_i=float(lambda_i),
                created_at=event_time,
                last_accessed=utc_now(),
            )
            new_atom_ids.append(atom_id)
            if self.eager_atom_embedding:
                self.embed.add(atom_id, self.encode_fn(text))

        ev_node.atom_ids.extend(new_atom_ids)
        ev_node.lambda_event = self._f2_compute_lambda_event(ev_node.atom_ids)
        ev_node.last_accessed = utc_now()

        self.embed.delete(f"event:{event_id}")
        self._refresh_episode_times(episode_id, awaken_time=event_time)
        return event_id

    # ---------------------------
    # Layer F3: Metabolism Scoring & Tri-State CSPC Transitions
    # ---------------------------

    def _episode_lambda(self, episode_id: int) -> float:
        ep = self.episodes[episode_id]
        vals = [self.events[eid].lambda_event for eid in ep.event_ids if eid in self.events]
        return max_pool_lambda(vals, floor=float(self.csaf.min_days))

    def compute_episode_score(
        self,
        episode_id: int,
        *,
        query: Optional[str] = None,
        current_time: Optional[datetime] = None,
        use_event_vectors: bool = True,
        use_label_vectors: bool = False,
    ) -> float:
        """
        Compute F3 Episode survival / metabolism score:
            f(E_p, t) = Res(E_p, ·) · exp(-Δt / λ_Episode)
        - When `query is None` (background metabolism): uses global patient profile `v_profile`.
        - When `query` is provided (retrieval ranking): uses query resonance over permitted indices.
        """
        now = current_time or utc_now()
        ep = self.episodes[episode_id]
        ref = ep.reference_time or ep.created_at
        delta_days = (now - ref).total_seconds() / 86400.0
        lam_ep = max(self._episode_lambda(episode_id), float(self.csaf.min_days))

        if query is None:
            res = self._episode_resonance_profile(episode_id)
            return metabolism_score(res, delta_days, lam_ep)

        qv = self.encode_fn(query)
        res = 0.0

        if use_label_vectors:
            for lid in ep.label_ids:
                lv = self.embed.get(lid)
                if lv is None:
                    label = self.labels.get(lid)
                    if label:
                        lv = self.encode_fn(label.summary)
                        self.embed.add(lid, lv)
                    else:
                        continue
                res = max(res, cosine_unit_interval(lv, qv))

        if use_event_vectors:
            for eid in ep.event_ids:
                ev_vec = self.embed.get(f"event:{eid}")
                if ev_vec is None:
                    ev = self.events.get(eid)
                    if ev:
                        ev_vec = self.encode_fn(self._event_text(ev.event))
                        self.embed.add(f"event:{eid}", ev_vec)
                    else:
                        continue
                res = max(res, cosine_unit_interval(ev_vec, qv))

        if res == 0.0:
            res = 0.5

        return metabolism_score(res, delta_days, lam_ep)

    def episode_state(self, episode_id: int, *, current_time: Optional[datetime] = None) -> str:
        """Evaluate current Tri-State (`active` | `compressed` | `forgotten`) from f(E_p, t)."""
        score = self.compute_episode_score(episode_id, query=None, current_time=current_time)
        return tri_state(
            score,
            active_threshold=self.active_threshold,
            compress_threshold=self.compress_threshold,
        )

    def metabolize_episode(
        self,
        episode_id: int,
        *,
        current_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Execute F3 Tri-State Metabolism & CSPC transitions on `episode_id`:
        - `active`:     retain all atoms and vectors
        - `compressed`: run CSPC label distillation + physically prune redundant atoms (f < τ_prune)
        - `forgotten`:  run CSPC label distillation + physically delete all atoms & event vectors
        """
        now = current_time or utc_now()
        ep = self.episodes[episode_id]
        prev_state = ep.state
        score = self.compute_episode_score(episode_id, query=None, current_time=now)
        new_state = tri_state(
            score,
            active_threshold=self.active_threshold,
            compress_threshold=self.compress_threshold,
        )

        report: Dict[str, Any] = {
            "episode_id": episode_id,
            "metabolism_score": score,
            "prev_state": prev_state,
            "new_state": new_state,
            "labels_created": 0,
            "atoms_deleted": 0,
            "events_deleted": 0,
            "vectors_deleted": 0,
        }

        if new_state == COMPRESSED:
            if prev_state == "active":
                report["labels_created"] = distill_episode_to_labels(
                    ep,
                    self.events,
                    self.atoms,
                    self.labels,
                    self.embed,
                    self.encode_fn,
                    min_frequency=self.min_frequency,
                    current_time=now,
                )
            purge_result = prune_redundant_atoms_compressed(
                ep,
                self.events,
                self.atoms,
                self.embed,
                self.encode_fn,
                self._get_profile_vector(),
                prune_threshold=self.survival_prune_threshold,
                min_days=float(self.csaf.min_days),
                current_time=now,
            )
            report["atoms_deleted"] = purge_result["atoms_deleted"]
            report["vectors_deleted"] = purge_result["vectors_deleted"]
        elif new_state == FORGOTTEN and prev_state != FORGOTTEN:
            report["labels_created"] = distill_episode_to_labels(
                ep,
                self.events,
                self.atoms,
                self.labels,
                self.embed,
                self.encode_fn,
                min_frequency=self.min_frequency,
                current_time=now,
            )
            deleted = forget_episode_physically(ep, self.events, self.atoms, self.embed)
            report.update(deleted)

        ep.state = new_state
        ep.last_metabolized_at = now
        return report

    def notify_dialog_turn(self, *, current_time: Optional[datetime] = None) -> Optional[List[Dict[str, Any]]]:
        """Trigger periodic F3 background metabolism every `metabolism_interval_rounds` dialog turns."""
        self._dialog_round += 1
        if self._dialog_round % self.metabolism_interval_rounds != 0:
            return None
        return self.run_background_metabolism(current_time=current_time)

    def run_background_metabolism(self, *, current_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        now = current_time or utc_now()
        return [self.metabolize_episode(ep_id, current_time=now) for ep_id in sorted(self.episodes.keys())]

    # ---------------------------
    # Adaptive Granularity Retrieval
    # ---------------------------

    def rank_episodes_for_query(
        self,
        question: str,
        *,
        top_k: int = 5,
        current_time: Optional[datetime] = None,
    ) -> List[Tuple[int, float]]:
        now = current_time or utc_now()
        scored: List[Tuple[int, float]] = []
        for episode_id, ep in self.episodes.items():
            policy = self.retrieval_controller.resolve(ep.state)
            score = self.compute_episode_score(
                episode_id,
                query=question,
                current_time=now,
                use_event_vectors=policy.use_event_vectors,
                use_label_vectors=policy.use_label_vectors,
            )
            scored.append((episode_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_across_episodes(
        self,
        question: str,
        *,
        top_k_episodes: int = 3,
        k_events: int = 10,
        k_atoms_per_event: int = 20,
        current_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        ranked = self.rank_episodes_for_query(question, top_k=top_k_episodes, current_time=current_time)
        out: List[Dict[str, Any]] = []
        for episode_id, ep_score in ranked:
            ctx = self.retrieve_episode_context(
                episode_id,
                question,
                k_events=k_events,
                k_atoms_per_event=k_atoms_per_event,
                current_time=current_time,
            )
            ctx["rank_score"] = ep_score
            out.append(ctx)
        return out

    def retrieve_episode_context(
        self,
        episode_id: int,
        question: str,
        *,
        k_events: int = 10,
        k_atoms_per_event: int = 20,
        current_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve episode context controlled by `AdaptiveRetrievalController`:
        - `active`     (`atoms`):     Top-k events + ranked Memory Atoms with f(m_i, q, t) >= τ_atom
        - `compressed` (`summaries`): Permanent Labels + Top-k Event summaries
        - `forgotten`  (`labels`):    Permanent Labels ONLY
        """
        now = current_time or utc_now()
        report = self.metabolize_episode(episode_id, current_time=now)
        state = self.episodes[episode_id].state
        policy = self.retrieval_controller.resolve(
            state,
            k_events=k_events,
            k_atoms_per_event=k_atoms_per_event,
        )

        episode_score = self.compute_episode_score(
            episode_id,
            query=question,
            current_time=now,
            use_event_vectors=policy.use_event_vectors,
            use_label_vectors=policy.use_label_vectors,
        )
        structure = self.build_clinical_episode_structure(episode_id)
        blocks: List[str] = []

        if policy.granularity == GRANULARITY_LABELS:
            for lid in self.episodes[episode_id].label_ids:
                label = self.labels.get(lid)
                if label:
                    blocks.append(f"[PermanentLabel] {label.summary}")
            return {
                "episode_id": episode_id,
                "state": state,
                "granularity": policy.granularity,
                "episode_score": episode_score,
                "blocks": blocks,
                "metabolism": report,
                "structure": structure,
            }

        if policy.granularity == GRANULARITY_SUMMARIES:
            for lid in self.episodes[episode_id].label_ids:
                label = self.labels.get(lid)
                if label:
                    blocks.append(f"[PermanentLabel] {label.summary}")
            for eid in self._rank_events_within_episode(episode_id, question, k=policy.k_events):
                ev = self.events.get(eid)
                if ev:
                    blocks.append(self._format_event_summary(ev.event))
            return {
                "episode_id": episode_id,
                "state": state,
                "granularity": policy.granularity,
                "episode_score": episode_score,
                "blocks": blocks,
                "metabolism": report,
                "structure": structure,
            }

        for eid in self._rank_events_within_episode(episode_id, question, k=policy.k_events):
            ev = self.events.get(eid)
            if not ev:
                continue
            blocks.append(self._format_event_header(ev.event))
            blocks.extend(
                self._rank_atoms_within_event(ev, question, k=policy.k_atoms_per_event, current_time=now)
            )

        return {
            "episode_id": episode_id,
            "state": state,
            "granularity": policy.granularity,
            "episode_score": episode_score,
            "blocks": blocks,
            "metabolism": report,
            "structure": structure,
        }

    def _rank_events_within_episode(self, episode_id: int, question: str, *, k: int = 10) -> List[int]:
        qv = self.encode_fn(question)
        scored: List[Tuple[int, float]] = []
        for eid in self.episodes[episode_id].event_ids:
            ev_vec = self.embed.get(f"event:{eid}")
            if ev_vec is None:
                ev = self.events.get(eid)
                if ev:
                    ev_vec = self.encode_fn(self._event_text(ev.event))
                    self.embed.add(f"event:{eid}", ev_vec)
                else:
                    continue
            scored.append((eid, cosine_unit_interval(ev_vec, qv)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in scored[:k]]

    def _rank_atoms_within_event(
        self,
        ev: EventNode,
        question: str,
        *,
        k: int = 20,
        current_time: Optional[datetime] = None,
    ) -> List[str]:
        now = current_time or utc_now()
        qv = self.encode_fn(question)
        scored: List[Tuple[str, float]] = []
        for aid in ev.atom_ids:
            atom = self.atoms.get(aid)
            if not atom:
                continue
            vec = self.embed.get(aid)
            if vec is None:
                vec = self.encode_fn(atom.text)
                self.embed.add(aid, vec)
            res = cosine_unit_interval(vec, qv)
            delta_days = (now - atom.created_at).total_seconds() / 86400.0
            decay = math.exp(-delta_days / (max(atom.lambda_i, float(self.csaf.min_days)) + 1e-8))
            survival = res * decay
            if survival >= self.survival_threshold_atom:
                scored.append((aid, survival))
        scored.sort(key=lambda x: x[1], reverse=True)
        lines: List[str] = []
        for aid, survival in scored[:k]:
            atom = self.atoms[aid]
            lines.append(
                f"- ({atom.kind}) {atom.text}  [tui={atom.tui}, φ={atom.phi:.2f}, λ={atom.lambda_i:.1f}d, f={survival:.3f}]"
            )
        return lines

    def build_clinical_episode_structure(self, episode_id: int) -> Dict[str, Any]:
        ep = self.episodes.get(episode_id)
        if not ep:
            return {}

        permanent_labels = [
            self.labels[lid].summary for lid in ep.label_ids if lid in self.labels
        ]
        key_timepoints: List[str] = []
        symptoms: List[str] = []
        directives: List[str] = []
        medications: List[str] = []
        diagnoses: List[str] = []
        seen: Set[str] = set()

        def _adduniq(bucket: List[str], text: object) -> None:
            t = str(text).strip()
            if t and t not in seen:
                seen.add(t)
                bucket.append(t)

        for eid in ep.event_ids:
            evn = self.events.get(eid)
            if not evn:
                continue
            d = evn.event
            date = d.get("date", "unknown")
            loc = d.get("location", "") or ""
            dx = d.get("diagnosis", "") or ""
            headline = dx or (d.get("summary") or "")[:160] or "(event)"
            key_timepoints.append(f"{date} | {loc} | {headline}".strip())

            for aid in evn.atom_ids:
                atom = self.atoms.get(aid)
                if not atom:
                    continue
                if atom.kind == "symptom":
                    _adduniq(symptoms, atom.text)
                elif atom.kind in ("treatment", "follow_up"):
                    _adduniq(directives, atom.text)
                elif atom.kind == "medication":
                    _adduniq(medications, atom.text)
                elif atom.kind == "diagnosis":
                    _adduniq(diagnoses, atom.text)

        ref = ep.reference_time or ep.created_at
        return {
            "episode_id": episode_id,
            "episode_last_activity_time": ref.isoformat() if ref else None,
            "permanent_labels": permanent_labels,
            "key_timepoints": key_timepoints[:30],
            "clinical_details": {
                "core_symptoms": symptoms[:30],
                "physician_directives": directives[:30],
                "medications_and_alternatives": medications[:30],
                "diagnoses": diagnoses[:20],
            },
        }

    def _event_text(self, ev: Dict[str, Any]) -> str:
        summary = ev.get("summary") or ""
        if summary:
            return str(summary)
        raw = ev.get("raw_dialogue") or []
        if raw:
            return "\n".join(str(x) for x in raw)
        return ""

    def _format_event_header(self, ev: Dict[str, Any]) -> str:
        return (
            f"[Event] date={ev.get('date', 'unknown')} "
            f"location={ev.get('location', 'unknown')} "
            f"diagnosis={ev.get('diagnosis', 'unknown')}"
        )

    def _format_event_summary(self, ev: Dict[str, Any]) -> str:
        return f"[EventSummary] {ev.get('summary', '')}"

    def storage_report(self) -> Dict[str, Any]:
        return {
            "episodes": len(self.episodes),
            "events": len(self.events),
            "atoms": len(self.atoms),
            "labels": len(self.labels),
            "embedding_store": self.embed.stats(),
        }


# Backward-compatible alias used in tests and paper appendix
CSAFHierarchicalMetabolizer = ChronicMemEngine
