"""
Unit and integration tests for the Chronic-Mem Three-Layer Memory Metabolism Engine (Section 3).

Run from `chronic-mem/`:
  CHRONIC_MEM_ENCODER=hash python tests/test_csaf_hierarchical.py
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_f1_semantic_triage() -> None:
    from chronic_mem.f1 import (
        ATOM_EXTRACTION_SYSTEM_PROMPT,
        ClinicalContext,
        LAMBDA_ACUTE,
        LAMBDA_CRITICAL,
        LAMBDA_DECISION,
        LAMBDA_PHENOTYPE,
        LAMBDA_SOCIAL,
        build_atom_extraction_messages,
        extract_atoms_from_dialogue,
        parse_llm_atoms_response,
    )

    assert "Memory Atoms" in ATOM_EXTRACTION_SYSTEM_PROMPT
    msgs = build_atom_extraction_messages(["Patient: Hello!", "Patient: Severe penicillin allergy."])
    assert len(msgs) == 2 and msgs[0]["role"] == "system"

    mock_llm_out = json.dumps(
        {
            "atoms": [
                {"kind": "social", "text": "Good morning doctor, thank you!"},
                {"kind": "diagnosis", "text": "Penicillin allergy"},
                {"kind": "medication", "text": "Metformin 500mg daily"},
                {"kind": "symptom", "text": "Mild cough"},
            ]
        }
    )
    parsed = parse_llm_atoms_response(mock_llm_out)
    assert len(parsed) == 4

    heur = extract_atoms_from_dialogue(
        [
            "**Patient:** Good morning, thank you!",
            "**Patient:** I have a penicillin allergy and severe wheezing.",
        ]
    )
    kinds = {k for k, _ in heur}
    assert "social" in kinds and "diagnosis" in kinds

    ctx = ClinicalContext()
    tui_allergy, _, lam_allergy = ctx.triage_atom("Penicillin allergy", "diagnosis")
    assert tui_allergy == "T037" and lam_allergy == LAMBDA_CRITICAL

    tui_rx, _, lam_rx = ctx.triage_atom("Albuterol inhaler", "medication")
    assert tui_rx == "T121" and lam_rx == LAMBDA_DECISION

    tui_dx, _, lam_dx = ctx.triage_atom("Chronic hypertension", "diagnosis")
    assert tui_dx == "T047" and lam_dx == LAMBDA_PHENOTYPE

    tui_sym, phi_sym, lam_sym = ctx.triage_atom("Severe cough", "symptom")
    assert tui_sym == "T184" and phi_sym == 1.5 and lam_sym == LAMBDA_ACUTE

    _, _, lam_soc = ctx.triage_atom("Thank you doctor!", "social")
    assert lam_soc == LAMBDA_SOCIAL


def test_f2_significance_gravity_and_latching() -> None:
    from chronic_mem.f2 import (
        DEFAULT_RHO_EVENT,
        cluster_atoms_to_events,
        max_pool_lambda,
        passes_event_threshold,
        significance_gravity,
    )
    from chronic_mem.geometry import l2_normalize

    v1 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    v2 = l2_normalize(np.array([0.98, 0.1, 0.0, 0.0]))
    v_orth = l2_normalize(np.array([-1.0, 0.0, 0.0, 0.0]))

    t_high_lam = significance_gravity(v1, v1, v2, v2, lambda_m=99999.0)
    t_low_lam = significance_gravity(v1, v1, v2, v2, lambda_m=3.0)
    assert t_high_lam > t_low_lam
    assert passes_event_threshold(t_high_lam, rho_event=DEFAULT_RHO_EVENT)

    assert max_pool_lambda([3.0, 14.0, 99999.0]) == 99999.0

    now = datetime(2026, 1, 1)
    atoms = [
        {"kind": "diagnosis", "tui": "T047", "v_med": v1, "v_ctx": v1, "lambda_i": 30.0, "created_at": now},
        {"kind": "Diagnosis_2", "tui": "T037", "v_med": v2, "v_ctx": v2, "lambda_i": 99999.0, "created_at": now},
        {"kind": "symptom", "tui": "T184", "v_med": v_orth, "v_ctx": v_orth, "lambda_i": 3.0, "created_at": now},
    ]
    events = cluster_atoms_to_events(atoms)
    assert len(events) == 2
    assert events[0]["lambda_event"] == 99999.0


def test_f3_resonance_wakeup_cspc_and_controller() -> None:
    from chronic_mem.engine import CSAFHierarchicalMetabolizer
    from chronic_mem.f1 import ClinicalContext, UMLSMapper, UrgencyDetector
    from chronic_mem.f3 import (
        AdaptiveRetrievalController,
        awaken_episode,
        cluster_events_to_episodes,
        ema_prototype,
        metabolism_score,
        resonance,
        retrieval_granularity,
        tri_state,
    )
    from chronic_mem.geometry import l2_normalize

    # Pure F3 resonance + wake-up + EMA check
    e_vec = l2_normalize(np.array([1.0, 0.2, 0.0, 0.0]))
    e_vec2 = l2_normalize(np.array([0.95, 0.25, 0.0, 0.0]))
    assert resonance(e_vec, e_vec, e_vec2, e_vec2) > 0.65

    updated_proto = ema_prototype(e_vec, e_vec2)
    assert abs(np.linalg.norm(updated_proto) - 1.0) < 1e-6

    t_old = datetime(2025, 1, 1)
    t_wake = datetime(2026, 6, 1)
    ev_protos = [
        {"v_med_proto": e_vec, "v_ctx_proto": e_vec, "last_time": t_old},
        {"v_med_proto": e_vec2, "v_ctx_proto": e_vec2, "last_time": datetime(2026, 5, 30)},
    ]
    ep_protos, ev2ep, _ = cluster_events_to_episodes(ev_protos, awaken_to=t_wake)
    assert len(ep_protos) == 1 and ev2ep[0] == ev2ep[1] == 0
    assert ep_protos[0]["awake_time"] == t_wake  # Δt resets to 0 relative to awaken_to

    assert tri_state(0.85) == "active"
    assert tri_state(0.50) == "compressed"
    assert tri_state(0.20) == "forgotten"
    assert retrieval_granularity("active") == "atoms"
    assert retrieval_granularity("compressed") == "summaries"
    assert retrieval_granularity("forgotten") == "labels"

    ctrl = AdaptiveRetrievalController()
    assert not ctrl.resolve("forgotten").use_event_vectors
    assert ctrl.resolve("compressed").use_event_vectors and not ctrl.resolve("compressed").use_atom_vectors
    assert metabolism_score(0.8, 0.0, 30.0) == 0.8

    # End-to-end hierarchical engine check on examples/events_small.json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    events_path = os.path.join(root, "examples", "events_small.json")

    csaf = ClinicalContext(
        cpp_model=None,
        umls_mapper=UMLSMapper(),
        urgency_detector=UrgencyDetector(),
    )
    metabolizer = CSAFHierarchicalMetabolizer(
        csaf,
        patient_profile_text="Chronic asthma patient with seasonal pollen allergy",
        min_frequency=1,
    )
    metabolizer.load_events(events_path)

    sr = metabolizer.storage_report()
    assert sr["episodes"] == 2, f"expected 2 episodes, got {sr['episodes']}"
    assert sr["events"] == 3, f"expected 3 events, got {sr['events']}"
    assert sr["atoms"] > 0, "expected atoms to be created"

    as_of = datetime(2026, 1, 15)
    ctx = metabolizer.retrieve_episode_context(
        0, "Does the patient have asthma?", k_events=2, k_atoms_per_event=10, current_time=as_of
    )
    assert ctx["state"] == "active" and ctx["granularity"] == "atoms"
    assert "asthma" in "\n".join(ctx["blocks"]).lower()

    # Spatio-temporal wake-up check: awaken_episode resets Δt -> 0
    ep0 = metabolizer.episodes[0]
    assert awaken_episode(ep0, datetime(2026, 5, 1)) == 0.0

    # Compressed state transition check (0.3 < f <= 0.7)
    ep0.reference_time = datetime(2026, 1, 10)
    t_compressed = datetime(2026, 4, 15)  # ~95d with λ_Episode=120d
    rep_comp = metabolizer.metabolize_episode(0, current_time=t_compressed)
    assert rep_comp["new_state"] == "compressed", f"expected compressed, got {rep_comp}"
    assert rep_comp["labels_created"] >= 1
    ctx_comp = metabolizer.retrieve_episode_context(
        0, "Does the patient have asthma?", k_events=2, k_atoms_per_event=10, current_time=t_compressed
    )
    assert ctx_comp["granularity"] == "summaries"

    # Forgotten state transition + physical deletion check
    delta = timedelta(days=3650)
    ep0.created_at = ep0.created_at - delta
    ep0.reference_time = ep0.reference_time - delta
    rep = metabolizer.metabolize_episode(0, current_time=as_of)
    assert metabolizer.episodes[0].state == "forgotten"
    assert rep["vectors_deleted"] >= 2

    ctx2 = metabolizer.retrieve_episode_context(
        0, "Does the patient have asthma?", k_events=2, k_atoms_per_event=10, current_time=as_of
    )
    blocks2 = ctx2["blocks"]
    assert blocks2 and all(b.startswith("[PermanentLabel]") for b in blocks2)
    assert metabolizer.embed.get("event:0") is None
    assert metabolizer.embed.get("event:1") is None

    # Clustered F1 -> F2 -> F3 ingestion check
    clustered_engine = CSAFHierarchicalMetabolizer(
        csaf,
        patient_profile_text="Chronic asthma patient with seasonal pollen allergy",
        min_frequency=1,
    )
    clustered_engine.load_events_clustered(events_path, awaken_to=as_of)
    sr_clustered = clustered_engine.storage_report()
    assert sr_clustered["episodes"] >= 1 and sr_clustered["events"] >= 1 and sr_clustered["atoms"] > 0


def main() -> int:
    test_f1_semantic_triage()
    test_f2_significance_gravity_and_latching()
    test_f3_resonance_wakeup_cspc_and_controller()
    print("✅ All Chronic-Mem Section 3 core tests (F1 / F2 / F3) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
