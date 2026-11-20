#!/usr/bin/env python3
"""
Standalone walkthrough of the Chronic-Mem Three-Layer Memory Metabolism Engine (Section 3).

Run from `chronic-mem/`:
    CHRONIC_MEM_ENCODER=hash python examples/demo_section3.py
"""
from __future__ import annotations

from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronic_mem import (
    AdaptiveRetrievalController,
    ChronicMemEngine,
    ClinicalContext,
    extract_atoms_from_dialogue,
    max_pool_lambda,
    significance_gravity,
)
from chronic_mem.geometry import l2_normalize
import numpy as np


def main() -> None:
    print("=" * 72)
    print("Chronic-Mem Core Engine Demo (Section 3: F1 -> F2 -> F3)")
    print("=" * 72)

    ctx = ClinicalContext()

    # 1. Layer F1: Dialogue -> Memory Atoms + UMLS TUI + Non-Homogeneous λ_i
    raw_dialogue = [
        "**Patient:** Good morning doctor, thank you for seeing me!",
        "**Patient:** I have severe wheezing since yesterday, and I have a penicillin allergy.",
        "**Doctor:** Avoid beta-lactams. Start albuterol inhaler and follow up in 2 weeks.",
    ]
    atoms = extract_atoms_from_dialogue(raw_dialogue)
    print("\n[Layer F1] Semantic Triage & Non-Homogeneous λ_i Initialization:")
    lambdas = []
    for kind, text in atoms:
        tui, phi, lam = ctx.triage_atom(text, kind)
        lambdas.append(lam)
        print(f"  - kind={kind:<10} tui={tui}  φ={phi:.1f}  λ_i={lam:>7.1f}d | {text}")

    # 2. Layer F2: Significance Gravity & Associated Latching (Max-Pooling)
    v = l2_normalize(np.array([1.0, 0.5, 0.2, 0.1]))
    t_critical = significance_gravity(v, v, v, v, lambda_m=99999.0)
    t_social = significance_gravity(v, v, v, v, lambda_m=3.0)
    lam_event = max_pool_lambda(lambdas)
    print("\n[Layer F2] Event Aggregation & Associated Latching:")
    print(f"  - Significance Gravity T(critical λ=99999d) = {t_critical:.4f} vs T(social λ=3d) = {t_social:.4f}")
    print(f"  - Associated Latching λ_Event = max(λ_i) = {lam_event:.1f}d")

    # 3. Layer F3: Episode Resonance, Wake-up, Profile Metabolism & CSPC Tri-State
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    events_path = os.path.join(root, "examples", "events_small.json")

    engine = ChronicMemEngine(
        ctx,
        patient_profile_text="Chronic asthma patient with seasonal allergies",
        min_frequency=1,
    )
    engine.load_events(events_path)
    t0 = datetime(2026, 1, 12)

    print("\n[Layer F3] Tri-State Metabolism & Adaptive Retrieval Controller:")
    active_res = engine.retrieve_episode_context(0, "Does the patient have asthma?", current_time=t0)
    print(f"  - t0 ({t0.date()}): state={active_res['state']} | granularity={active_res['granularity']} | blocks={len(active_res['blocks'])}")

    # Advance clock into Compressed window (0.3 < f <= 0.7) -> CSPC distills labels & prunes weak atoms
    t_mid = t0 + timedelta(days=95)
    rep_comp = engine.metabolize_episode(0, current_time=t_mid)
    comp_res = engine.retrieve_episode_context(0, "Does the patient have asthma?", current_time=t_mid)
    print(
        f"  - t_mid ({t_mid.date()}): state={comp_res['state']} | granularity={comp_res['granularity']} "
        f"| labels_created={rep_comp['labels_created']} | atoms_pruned={rep_comp['atoms_deleted']}"
    )

    # Cross-time Episode Resonance & Spatio-Temporal Wake-up (resets Δt -> 0)
    t_followup = t_mid + timedelta(days=30)
    wake_info = engine.ingest_dialogue_resonant(
        ["**Patient:** My asthma wheezing came back this week, using albuterol inhaler again."],
        current_time=t_followup,
    )
    delta_after_wake = (t_followup - engine.episodes[wake_info["episode_id"]].reference_time).total_seconds() / 86400.0
    print(
        f"  - Follow-up @ {t_followup.date()}: resonant_ep={wake_info['episode_id']} "
        f"R={wake_info['resonance_score']:.3f} awakened={wake_info['awakened']} Δt={delta_after_wake:.1f}d"
    )

    # Advance 10 years -> Forgotten state: physical vector cleanup, retain Permanent Labels only
    t_far = t_followup + timedelta(days=3650)
    rep_forg = engine.metabolize_episode(0, current_time=t_far)
    forg_res = engine.retrieve_episode_context(0, "Does the patient have asthma?", current_time=t_far)
    ctrl = AdaptiveRetrievalController()
    print(
        f"  - t_far ({t_far.date()}): state={forg_res['state']} | policy={ctrl.resolve(forg_res['state']).granularity} "
        f"| vectors_deleted={rep_forg['vectors_deleted']} | blocks={forg_res['blocks']}"
    )


if __name__ == "__main__":
    main()
