"""
F1 — UMLS TUI (Semantic Type Identifier) Mapping & Embedding Layer.

Maps clinical entities in Memory Atoms to UMLS Semantic Types (TUIs) and projects
each TUI to a 16-dimensional semantic type vector (`v_med` / `v_umls`).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# Embedded clinical UMLS TUI vocabulary (21 clinical types + default T000)
UMLS_TUI_MAPPING: Dict[str, str] = {
    # Disease / Syndrome
    "T047": "Disease or Syndrome",
    "T048": "Mental or Behavioral Dysfunction",
    "T049": "Cell or Molecular Dysfunction",
    "T050": "Experimental Model of Disease",
    # Sign / Symptom / Finding
    "T184": "Sign or Symptom",
    "T033": "Finding",
    # Therapeutic / Diagnostic Procedures
    "T061": "Therapeutic or Preventive Procedure",
    "T060": "Diagnostic Procedure",
    "T059": "Laboratory Procedure",
    # Pharmacologic / Chemical
    "T109": "Organic Chemical",
    "T121": "Pharmacologic Substance",
    "T125": "Hormone",
    # Allergy / Injury / Abnormality
    "T037": "Injury or Poisoning",  # Core allergy & hypersensitivity lane
    "T020": "Acquired Abnormality",
    # Anatomy
    "T023": "Body Part, Organ, or Organ Component",
    "T031": "Body Substance",
    # Demographics
    "T100": "Age Group",
    "T101": "Patient or Disabled Group",
}

DEFAULT_TUI = "T000"


class UMLSMapper:
    """Maps clinical text / event slices to UMLS TUIs and 16-d type embeddings."""

    def __init__(self, embedding_dim: int = 16, *, seed: int = 42, use_torch: bool = False):
        self.embedding_dim = int(embedding_dim)
        self.tui_to_id = {tui: idx for idx, tui in enumerate(UMLS_TUI_MAPPING.keys())}
        self.tui_to_id[DEFAULT_TUI] = len(self.tui_to_id)
        self.vocab_size = len(self.tui_to_id)
        self.use_torch = bool(use_torch)

        rng = np.random.RandomState(seed)
        self._table = rng.randn(self.vocab_size, self.embedding_dim).astype(np.float32)
        self._torch_embedding: Any = None
        if self.use_torch:
            import torch
            import torch.nn as nn

            emb = nn.Embedding(self.vocab_size, self.embedding_dim)
            with torch.no_grad():
                emb.weight.copy_(torch.from_numpy(self._table))
            self._torch_embedding = emb

    def detect_tui_from_text(self, text: str, event_type: Optional[str] = None) -> str:
        """
        Rule precedence (Section 3.1):
          1. Allergy / hypersensitivity -> T037
          2. Treatment / surgery        -> T061
          3. Diagnosis / disease        -> T047
          4. Symptom / sign             -> T184
          5. Medication / drug          -> T121
          6. Otherwise                  -> T000
        """
        if not text:
            return DEFAULT_TUI

        text_lower = str(text).lower()

        if any(
            word in text_lower
            for word in ["allergy", "allergic", "allergen", "hypersensitivity", "过敏"]
        ):
            return "T037"

        if event_type == "treatment" or any(
            word in text_lower
            for word in ["surgery", "operation", "procedure", "therapy", "treatment", "手术", "治疗"]
        ):
            return "T061"

        if event_type == "diagnosis" or any(
            word in text_lower
            for word in ["diagnosis", "disease", "syndrome", "disorder", "诊断", "疾病"]
        ):
            return "T047"

        if event_type == "symptom" or any(
            word in text_lower
            for word in ["symptom", "pain", "ache", "fever", "cough", "wheez", "症状", "疼痛", "发烧", "咳", "喘"]
        ):
            return "T184"

        if event_type == "medication" or any(
            word in text_lower
            for word in ["medication", "drug", "pill", "prescription", "inhaler", "药物", "药"]
        ):
            return "T121"

        return DEFAULT_TUI

    def get_embedding(self, tui: str) -> Any:
        """Return 16-d embedding vector for `tui` (torch.Tensor if `use_torch=True`, else np.ndarray)."""
        tui_id = self.tui_to_id.get(tui, self.tui_to_id[DEFAULT_TUI])
        if self._torch_embedding is not None:
            import torch

            return self._torch_embedding(torch.tensor(tui_id))
        return self._table[tui_id].copy()

    def extract_tui_from_event(self, event: Dict[str, Any]) -> str:
        """Event-level TUI priority: diagnosis > treatment > symptoms > medications > T000."""
        if event.get("diagnosis"):
            return self.detect_tui_from_text(str(event["diagnosis"]), "diagnosis")
        if event.get("treatment") and len(event.get("treatment", [])) > 0:
            return self.detect_tui_from_text(str(event["treatment"][0]), "treatment")
        if event.get("symptoms") and len(event.get("symptoms", [])) > 0:
            return self.detect_tui_from_text(str(event["symptoms"][0]), "symptom")
        if event.get("medications") and len(event.get("medications", [])) > 0:
            return self.detect_tui_from_text(str(event["medications"][0]), "medication")
        return DEFAULT_TUI
