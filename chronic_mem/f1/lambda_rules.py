"""
F1 — Non-Homogeneous Decay Rate (λ_i) Rule Configuration Table.

Initializes atomic memory retention horizons λ_i (in days) according to clinical
decision value:
- Critical (∞ / 99999d): Core safety anchors (drug/food allergies T037, major surgeries, permanent flags)
- Decision (120d):       Key clinical interventions (prescriptions T121, procedures T061, disease recurrence)
- Phenotype (30d):       Chronic disease phenotype (T047), lifestyle/monitoring logs, visit summaries
- Acute (14d):           Transient acute signs & symptoms (T184)
- Social (3d):           Low-entropy conversational chatter / greetings
"""
from __future__ import annotations

from typing import Any, Dict, Optional

LAMBDA_CRITICAL = 99999.0  # operational ∞ (core safety anchor: allergy / major surgery)
LAMBDA_DECISION = 120.0    # key clinical decision (medication, procedure, recurrence)
LAMBDA_PHENOTYPE = 30.0    # chronic phenotype & baseline context
LAMBDA_ACUTE = 14.0        # transient symptom
LAMBDA_SOCIAL = 3.0        # low-entropy social chatter

# Explicit rule configuration table (Section 3.1 / Table of Non-Homogeneous λ_i Tiers)
LAMBDA_TIER_CONFIG: Dict[str, Dict[str, Any]] = {
    "critical": {
        "lambda_days": LAMBDA_CRITICAL,
        "tui_triggers": ("T037",),
        "description": "Core safety anchors: allergies, hypersensitivity, major surgeries, permanent flags",
    },
    "decision": {
        "lambda_days": LAMBDA_DECISION,
        "tui_triggers": ("T121", "T061"),
        "description": "Key clinical decisions: medications, therapeutic procedures, disease recurrence",
    },
    "phenotype": {
        "lambda_days": LAMBDA_PHENOTYPE,
        "tui_triggers": ("T047",),
        "description": "Chronic disease phenotype, lifestyle/CGM monitoring logs, clinical summaries",
    },
    "acute": {
        "lambda_days": LAMBDA_ACUTE,
        "tui_triggers": ("T184",),
        "description": "Acute transient symptoms and complaints",
    },
    "social": {
        "lambda_days": LAMBDA_SOCIAL,
        "tui_triggers": (),
        "description": "Low-entropy social greetings and small talk",
    },
}

ALLERGY_MARKERS = (
    "allergy", "allergic", "allergen", "hypersensitivity", "anaphylaxis", "contraindicated",
    "过敏", "过敏原", "青霉素过敏", "药物过敏", "禁忌",
)

MAJOR_SURGERY_MARKERS = (
    "open heart", "cardiac surgery", "coronary bypass", "cabg", "bypass surgery",
    "heart transplant", "lung transplant", "liver transplant", "kidney transplant",
    "organ transplant", "transplant surgery", "craniotomy", "whipple",
    "colectomy", "colostomy", "nephrectomy", "esophagectomy",
    "hysterectomy", "major surgery", "radical resection",
    "knee replacement", "hip replacement", "joint replacement", "arthroplasty",
    "搭桥", "开胸", "器官移植", "重大手术", "置换术", "全麻大手术",
)

RECURRENCE_MARKERS = (
    "recurrence", "recurrent", "relapse", "relapsed", "再次出现",
    "复发", "再发", "反复发作",
)

PHENOTYPE_MARKERS = (
    "cgm", "continuous glucose", "glucose monitor", "blood sugar log",
    "fasting glucose", "hba1c trend", "血糖监测", "连续血糖", "糖化血红蛋白",
    "exercise frequency", "weekly exercise", "steps per day", "minutes of activity",
    "运动频率", "每日步数", "有氧", "冥想瑜伽", "yoga 2x",
)

SOCIAL_GREETING_FRAGMENTS = (
    "thank you", "thanks", "you're welcome", "my pleasure",
    "good morning", "good afternoon", "good evening", "have a seat",
    "nice to meet", "see you", "take care",
    "谢谢", "感谢", "您好", "早上好", "下午好", "不客气", "再见", "请坐",
)


def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _strip_role_prefix(tl: str) -> str:
    for p in (
        "**patient:**", "**doctor:**", "**patient**:", "**doctor**:",
        "**nurse:**", "**nurse**:", "patient:", "doctor:", "患者:", "医生:", "患者：", "医生：",
    ):
        if tl.startswith(p):
            return tl[len(p):].strip()
    return tl


def is_social_dialogue_line(text: str) -> bool:
    """Detect polite greetings / short low-entropy conversational noise -> Social tier (λ=3d)."""
    raw = (text or "").strip()
    if not raw or len(raw) > 320:
        return False
    tl = _strip_role_prefix(_norm(raw))
    if not tl:
        return False
    return any(g in tl for g in SOCIAL_GREETING_FRAGMENTS)


def is_major_surgery_text(text: str, kind: Optional[str], tui: str) -> bool:
    if kind == "medication":
        return False
    tl = _norm(text)
    if not tl:
        return False
    return any(m in tl for m in MAJOR_SURGERY_MARKERS)


def has_recurrence(text: str) -> bool:
    tl = _norm(text)
    return any(m in tl for m in RECURRENCE_MARKERS)


def has_phenotype_marker(text: str, kind: Optional[str]) -> bool:
    if kind == "lifestyle":
        return True
    tl = _norm(text)
    return any(m in tl for m in PHENOTYPE_MARKERS)


def has_allergy_marker(text: str, tui: str) -> bool:
    if tui == "T037":
        return True
    tl = _norm(text)
    return any(m in tl for m in ALLERGY_MARKERS)


def lambda_days_hard_rule(
    *,
    text: str,
    kind: Optional[str],
    tui: str,
    is_event_permanent: bool = False,
) -> float:
    """
    Assign non-homogeneous decay horizon λ_i (days) to a single Memory Atom.
    Precedence:
      Critical (99999d) > Social (3d) > Decision (120d) > Phenotype (30d) > Acute (14d) > Default (30d).
    """
    if is_event_permanent:
        return LAMBDA_CRITICAL

    if has_allergy_marker(text, tui) or is_major_surgery_text(text, kind, tui):
        return LAMBDA_CRITICAL

    if kind == "social" or (kind == "raw_dialogue" and is_social_dialogue_line(text)):
        return LAMBDA_SOCIAL

    if tui == "T121" or kind == "medication":
        return LAMBDA_DECISION

    if tui == "T061" or kind == "treatment":
        return LAMBDA_DECISION

    if tui == "T047" and has_recurrence(text):
        return LAMBDA_DECISION

    if has_phenotype_marker(text, kind):
        return LAMBDA_PHENOTYPE

    if tui == "T047" or kind == "diagnosis":
        return LAMBDA_PHENOTYPE

    if tui == "T184" or kind == "symptom":
        return LAMBDA_ACUTE

    if kind in ("people", "follow_up", "summary"):
        return LAMBDA_PHENOTYPE

    if kind == "raw_dialogue":
        return LAMBDA_PHENOTYPE

    return LAMBDA_PHENOTYPE


def rule_lambda_max_over_event(event: Dict[str, Any], umls_mapper: Any) -> float:
    """Conservative event-level rule fallback: max λ_i across structured event fields."""
    if event.get("is_permanent"):
        return LAMBDA_CRITICAL

    scores: list[float] = []

    if event.get("diagnosis"):
        t = str(event["diagnosis"])
        tu = umls_mapper.detect_tui_from_text(t, "diagnosis")
        scores.append(lambda_days_hard_rule(text=t, kind="diagnosis", tui=tu))

    for s in event.get("symptoms") or []:
        if s:
            t = str(s)
            tu = umls_mapper.detect_tui_from_text(t, "symptom")
            scores.append(lambda_days_hard_rule(text=t, kind="symptom", tui=tu))

    for tr in event.get("treatment") or []:
        if tr:
            t = str(tr)
            tu = umls_mapper.detect_tui_from_text(t, "treatment")
            scores.append(lambda_days_hard_rule(text=t, kind="treatment", tui=tu))

    for m in event.get("medications") or []:
        if m:
            t = str(m)
            tu = umls_mapper.detect_tui_from_text(t, "medication")
            scores.append(lambda_days_hard_rule(text=t, kind="medication", tui=tu))

    for x in event.get("lifestyle") or []:
        if x:
            t = str(x)
            tu = umls_mapper.detect_tui_from_text(t, "lifestyle")
            scores.append(lambda_days_hard_rule(text=t, kind="lifestyle", tui=tu))

    if event.get("summary"):
        t = str(event["summary"])
        tu = umls_mapper.detect_tui_from_text(t, None)
        scores.append(lambda_days_hard_rule(text=t, kind="summary", tui=tu))

    for line in event.get("raw_dialogue") or []:
        if not line:
            continue
        t = str(line).strip()
        if is_social_dialogue_line(t):
            scores.append(lambda_days_hard_rule(text=t, kind="raw_dialogue", tui="T000"))

    if not scores:
        return LAMBDA_PHENOTYPE
    return max(scores)


def is_critical_lambda(value: float) -> bool:
    return value >= 90000.0


def is_social_lambda(value: float) -> bool:
    return value <= LAMBDA_SOCIAL + 0.6
