"""
F1 — Dialogue-to-Memory-Atoms Extraction Script & Prompt.

Decomposes raw clinical dialogue turns (or structured clinical visit dicts) into
typed atomic memory units `m_i = (kind, text)` before UMLS TUI mapping and
non-homogeneous decay rate λ_i initialization:

- Clinical safety & decision atoms: `diagnosis`, `symptom`, `treatment`, `medication`
- Longitudinal phenotype & context atoms: `lifestyle`, `people`, `follow_up`, `summary`
- Low-entropy conversational chatter: `social` (routed to fast decay λ = 3d)

Supports:
1. LLM-based extraction via `ATOM_EXTRACTION_SYSTEM_PROMPT` + `extract_atoms_from_dialogue(..., llm_fn=...)`
2. Offline rule-based heuristic extraction via `extract_atoms_heuristic(...)` (zero API dependency)
3. Structured event decomposition via `atomize_event(...)`
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from chronic_mem.f1.lambda_rules import ALLERGY_MARKERS, MAJOR_SURGERY_MARKERS, is_social_dialogue_line

VALID_ATOM_KINDS = (
    "diagnosis",
    "symptom",
    "treatment",
    "medication",
    "lifestyle",
    "people",
    "follow_up",
    "summary",
    "social",
)

ATOM_EXTRACTION_SYSTEM_PROMPT = """You are a clinical memory extraction module (Layer F1: Semantic Triage) in Chronic-Mem.
Given a doctor-patient dialogue segment, decompose it into atomic, self-contained Memory Atoms.

Classify each Memory Atom into ONE `kind`:
- "diagnosis": Confirmed/suspected diseases, syndromes, or drug/food allergies & contraindications (critical safety anchors).
- "treatment": Surgeries, clinical procedures, therapeutic directives, dose adjustments, or discontinuation orders.
- "medication": Prescribed drugs, inhalers, supplements, or active pharmacological regimens.
- "symptom": Acute or chronic patient-reported symptoms, signs, or adverse reactions.
- "lifestyle": Diet, exercise, sleep, home monitoring (e.g., CGM / blood pressure logs), smoking/alcohol habits.
- "follow_up": Scheduled re-examinations, lab check intervals, or emergency red-flag instructions.
- "people": Relevant caregivers or family medical history mentions.
- "summary": A concise 1-sentence clinical summary of the turn (omit if purely social).
- "social": Greetings, thanks, pleasantries, or low-entropy small talk (isolated so F3 can rapidly prune noise).

Rules:
1. Preserve exact drug names, dosages, negations, and allergy/contraindication anchors verbatim.
2. Separate low-entropy social chatter into `"social"` atoms rather than mixing it into clinical facts.
3. Return STRICT JSON only (no Markdown fences, no commentary) matching:
{
  "atoms": [
    {"kind": "diagnosis|symptom|treatment|medication|lifestyle|people|follow_up|summary|social", "text": "..."}
  ]
}"""

ATOM_EXTRACTION_USER_TEMPLATE = """Decompose the following clinical dialogue into Memory Atoms (JSON):

{dialogue_block}"""


def build_atom_extraction_messages(dialogue_lines: Sequence[str]) -> List[Dict[str, str]]:
    """Build `[system, user]` chat messages for LLM-driven Memory Atom extraction."""
    cleaned = [str(line).strip() for line in dialogue_lines if str(line).strip()]
    block = "\n".join(cleaned)
    return [
        {"role": "system", "content": ATOM_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": ATOM_EXTRACTION_USER_TEMPLATE.format(dialogue_block=block)},
    ]


def parse_llm_atoms_response(raw_response: str) -> List[Tuple[str, str]]:
    """Parse JSON output from `ATOM_EXTRACTION_SYSTEM_PROMPT` into `(kind, text)` pairs."""
    s = (raw_response or "").strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        s = m.group(0)
    payload = json.loads(s)

    out: List[Tuple[str, str]] = []
    if isinstance(payload, dict) and isinstance(payload.get("atoms"), list):
        for item in payload["atoms"]:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip().lower()
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            if kind not in VALID_ATOM_KINDS:
                kind = "summary"
            out.append((kind, text))
        return out

    # Also accept MemoryART event-schema JSON if the caller's LLM returns event keys directly
    if isinstance(payload, dict):
        return atomize_event(payload)
    return out


_MEDICATION_HINTS = (
    "mg", "ml", "tablet", "pill", "capsule", "inhaler", "insulin", "metformin",
    "lisinopril", "amlodipine", "albuterol", "atorvastatin", "aspirin", "penicillin",
    "loratadine", "prescription", "medication", "drug",
    "片", "毫克", "吸入", "口服", "胰岛素", "二甲双胍", "普利", "沙坦", "地平", "他汀", "药",
)
_TREATMENT_HINTS = (
    "surgery", "operation", "procedure", "therapy", "nebulizer", "discontinue", "stop taking",
    "transplant", "bypass", "resection",
    "手术", "治疗", "雾化", "停用", "停药", "减量", "加量", "换用", "方案",
)
_DIAGNOSIS_HINTS = (
    "diagnosed", "diagnosis", "hypertension", "diabetes", "asthma", "rhinitis", "syndrome",
    "disease", "disorder",
    "确诊", "诊断", "高血压", "糖尿病", "哮喘", "鼻炎", "综合征", "病史",
)
_SYMPTOM_HINTS = (
    "pain", "ache", "cough", "wheez", "shortness of breath", "dyspnea", "fever", "dizzy",
    "rash", "itching", "sneez", "nausea", "fatigue", "swelling",
    "痛", "疼", "咳", "喘", "胸闷", "气促", "发烧", "发热", "头晕", "皮疹", "瘙痒", "恶心", "乏力",
)
_FOLLOW_UP_HINTS = (
    "follow up", "follow-up", "recheck", "next visit", "in 2 weeks", "in 3 months", "return if",
    "随访", "复查", "复诊", "周后", "月后", "急诊",
)
_LIFESTYLE_HINTS = (
    "diet", "exercise", "walk", "jog", "sleep", "smoking", "alcohol", "glucose log", "cgm",
    "饮食", "运动", "慢跑", "步数", "睡眠", "作息", "戒烟", "饮酒", "血糖监测",
)


def _classify_clause_heuristic(clause: str) -> str:
    if is_social_dialogue_line(clause):
        return "social"
    cl = clause.lower()
    if any(k in cl for k in ALLERGY_MARKERS):
        return "diagnosis"
    if any(k in cl for k in MAJOR_SURGERY_MARKERS):
        return "treatment"
    if any(k in cl for k in _FOLLOW_UP_HINTS):
        return "follow_up"
    if any(k in cl for k in _MEDICATION_HINTS):
        return "medication"
    if any(k in cl for k in _TREATMENT_HINTS):
        return "treatment"
    if any(k in cl for k in _DIAGNOSIS_HINTS):
        return "diagnosis"
    if any(k in cl for k in _SYMPTOM_HINTS):
        return "symptom"
    if any(k in cl for k in _LIFESTYLE_HINTS):
        return "lifestyle"
    return "raw_dialogue"


def extract_atoms_heuristic(dialogue_lines: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Offline rule-based dialogue-to-atoms decomposition.
    Splits multi-sentence turns into atomic clauses and assigns F1 structural kinds.
    """
    atoms: List[Tuple[str, str]] = []
    for raw_line in dialogue_lines:
        line = str(raw_line or "").strip()
        if not line:
            continue

        # Strip optional speaker prefix (e.g., "**Patient:** ", "Doctor: ", "患者：")
        body = re.sub(
            r"^\*{0,2}(patient|doctor|nurse|assistant|user|患者|医生)\*{0,2}\s*[:：]\s*\*{0,2}\s*",
            "",
            line,
            flags=re.I,
        ).strip()
        if not body:
            continue

        if is_social_dialogue_line(line) and len(body) <= 60:
            atoms.append(("social", line))
            continue

        clauses = [c.strip() for c in re.split(r"[。！？!?；;\n]+|(?<=[.!?])\s+", body) if c and c.strip()]
        for clause in clauses:
            kind = _classify_clause_heuristic(clause)
            atoms.append((kind, clause))

    return atoms


def extract_atoms_from_dialogue(
    dialogue_lines: Sequence[str],
    *,
    llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> List[Tuple[str, str]]:
    """
    Decompose raw dialogue text into `(kind, text)` Memory Atoms.

    Args:
        dialogue_lines: Sequence of speaker turns or raw dialogue utterances.
        llm_fn: Optional callable accepting OpenAI-style `messages` and returning JSON text.
                Falls back to `extract_atoms_heuristic` when `llm_fn` is None or fails.
    """
    if not dialogue_lines:
        return []
    if llm_fn is not None:
        try:
            messages = build_atom_extraction_messages(dialogue_lines)
            raw_json = llm_fn(messages)
            parsed = parse_llm_atoms_response(raw_json)
            if parsed:
                return parsed
        except Exception:
            pass
    return extract_atoms_heuristic(dialogue_lines)


def atomize_event(ev: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Extract `(kind, text)` Memory Atoms from a structured clinical visit dict (`events.json`),
    plus low-entropy social chatter lines from `raw_dialogue`.
    If no structured clinical fields are populated, falls back to heuristic dialogue extraction.
    """
    atoms: List[Tuple[str, str]] = []

    if ev.get("diagnosis"):
        atoms.append(("diagnosis", str(ev["diagnosis"])))
    for s in ev.get("symptoms", []) or []:
        if s:
            atoms.append(("symptom", str(s)))
    for t in ev.get("treatment", []) or []:
        if t:
            atoms.append(("treatment", str(t)))
    for m in ev.get("medications", []) or []:
        if m:
            atoms.append(("medication", str(m)))
    for x in ev.get("lifestyle", []) or []:
        if x:
            atoms.append(("lifestyle", str(x)))
    for p in ev.get("people", []) or []:
        if p:
            atoms.append(("people", str(p)))
    for f in ev.get("follow_up", []) or []:
        if f:
            atoms.append(("follow_up", str(f)))

    has_structured_clinical = bool(atoms)

    if ev.get("summary"):
        atoms.append(("summary", str(ev["summary"])))

    raw_lines = ev.get("raw_dialogue") or []
    if not has_structured_clinical and raw_lines:
        for kind, text in extract_atoms_heuristic(raw_lines):
            atoms.append((kind, text))
    else:
        for line in raw_lines:
            if not line:
                continue
            text = str(line).strip()
            if text and is_social_dialogue_line(text):
                atoms.append(("social", text))

    return atoms


def atomize_incremental_lines(extra_raw_lines: Sequence[str]) -> List[Tuple[str, str]]:
    """Atomize appended raw dialogue lines for an ongoing visit (`social` vs `raw_dialogue`)."""
    out: List[Tuple[str, str]] = []
    for line in extra_raw_lines:
        if not line:
            continue
        t = str(line).strip()
        if not t:
            continue
        if is_social_dialogue_line(t):
            out.append(("social", t))
        else:
            out.append(("raw_dialogue", t))
    return out
