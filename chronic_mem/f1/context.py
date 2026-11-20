"""F1 Semantic Triage runtime context: UMLS mapper, urgency scalar φ, rule table, optional CPP."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

from chronic_mem.f1.cpp_model import ClinicalPersistencePredictor
from chronic_mem.f1.lambda_rules import (
    LAMBDA_CRITICAL,
    is_critical_lambda,
    is_social_lambda,
    lambda_days_hard_rule,
)
from chronic_mem.f1.umls_mapper import UMLSMapper
from chronic_mem.f1.urgency import UrgencyDetector


class ClinicalContext:
    """Bundles F1 Semantic Triage components and computes `(tui, phi, lambda_i)` per Memory Atom."""

    def __init__(
        self,
        umls_mapper: Optional[UMLSMapper] = None,
        urgency_detector: Optional[UrgencyDetector] = None,
        cpp_model: Optional[ClinicalPersistencePredictor] = None,
        context_encoder: Optional[Callable] = None,
        min_days: int = 1,
        max_days: int = 10000,
    ):
        self.umls_mapper = umls_mapper or UMLSMapper()
        self.urgency_detector = urgency_detector or UrgencyDetector()
        self.cpp_model = cpp_model
        self.context_encoder = context_encoder
        self.min_days = min_days
        self.max_days = max_days

    def triage_atom(
        self,
        text: str,
        kind: Optional[str] = None,
        *,
        is_event_permanent: bool = False,
        homogeneous_lambda_days: Optional[float] = None,
    ) -> Tuple[str, float, float]:
        """Run F1 triage on one Memory Atom -> `(tui, phi, lambda_i)`."""
        tui = self.umls_mapper.detect_tui_from_text(text, kind)
        phi = float(self.urgency_detector.detect_urgency(text))

        if homogeneous_lambda_days is not None and homogeneous_lambda_days > 0:
            return tui, phi, float(homogeneous_lambda_days)

        hr = lambda_days_hard_rule(
            text=text,
            kind=kind,
            tui=tui,
            is_event_permanent=is_event_permanent,
        )
        if is_critical_lambda(hr):
            return tui, phi, LAMBDA_CRITICAL
        if is_social_lambda(hr) or kind == "social":
            return tui, phi, float(hr)

        if self.cpp_model is not None and self.context_encoder is not None:
            try:
                v_umls = self.umls_mapper.get_embedding(tui)
                v_context = self.context_encoder(text)
                pred = self.cpp_model.predict_lambda(v_umls, v_context, phi)
                return tui, phi, float(pred)
            except Exception:
                pass

        return tui, phi, float(hr)
