"""
Chronic-Mem Core Package (Section 3): Three-Layer Clinical Memory Metabolism Engine.

Layers:
- `chronic_mem.f1`: Semantic Triage & Non-Homogeneous Lifecycle Initialization (λ_i)
- `chronic_mem.f2`: Significance Gravity Aggregation T(Atom, Event_j) & Associated Latching
- `chronic_mem.f3`: Episode Resonance R(e_j, E_p), Spatio-Temporal Wake-up, CSPC Tri-State Metabolism,
                    and Adaptive Retrieval Granularity Controller
"""

from chronic_mem.engine import CSAFHierarchicalMetabolizer, ChronicMemEngine
from chronic_mem.f1 import (
    ATOM_EXTRACTION_SYSTEM_PROMPT,
    ClinicalContext,
    LAMBDA_TIER_CONFIG,
    UMLSMapper,
    UrgencyDetector,
    atomize_event,
    extract_atoms_from_dialogue,
    lambda_days_hard_rule,
)
from chronic_mem.f2 import (
    DEFAULT_RHO_EVENT,
    cluster_atoms_to_events,
    max_pool_lambda,
    passes_event_threshold,
    significance_gravity,
)
from chronic_mem.f3 import (
    AdaptiveRetrievalController,
    DEFAULT_RHO_EPISODE,
    awaken_episode,
    cluster_events_to_episodes,
    distill_episode_to_labels,
    ema_prototype,
    forget_episode_physically,
    metabolism_score,
    profile_resonance,
    prune_redundant_atoms_compressed,
    resonance,
    retrieval_granularity,
    tri_state,
)
from chronic_mem.types import AtomNode, EmbeddingStore, EpisodeNode, EventNode, PermanentLabel

__all__ = [
    "ChronicMemEngine",
    "CSAFHierarchicalMetabolizer",
    "ClinicalContext",
    "ATOM_EXTRACTION_SYSTEM_PROMPT",
    "extract_atoms_from_dialogue",
    "atomize_event",
    "UMLSMapper",
    "UrgencyDetector",
    "LAMBDA_TIER_CONFIG",
    "lambda_days_hard_rule",
    "DEFAULT_RHO_EVENT",
    "significance_gravity",
    "passes_event_threshold",
    "cluster_atoms_to_events",
    "max_pool_lambda",
    "DEFAULT_RHO_EPISODE",
    "resonance",
    "awaken_episode",
    "ema_prototype",
    "cluster_events_to_episodes",
    "profile_resonance",
    "metabolism_score",
    "tri_state",
    "distill_episode_to_labels",
    "prune_redundant_atoms_compressed",
    "forget_episode_physically",
    "AdaptiveRetrievalController",
    "retrieval_granularity",
    "AtomNode",
    "EventNode",
    "EpisodeNode",
    "PermanentLabel",
    "EmbeddingStore",
]
