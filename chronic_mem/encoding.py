"""
Text encoding and clinical event JSON loader for Chronic-Mem.

Uses `bert-base-uncased` CLS embeddings (768-d) loaded lazily on first call:
1. Local snapshot in `<repo>/bert/` or `CHRONIC_MEM_BERT_PATH` if present.
2. Hugging Face `bert-base-uncased` unless `HF_HUB_OFFLINE=1` or `CHRONIC_MEM_ENCODER=hash`.
3. Deterministic 768-d hash + lexical feature fallback so offline execution never blocks.

Custom encoders can also be injected via `set_encoder(fn)`.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)

_tokenizer: Any = None
_bert_model: Any = None
_encoder_initialized: bool = False
_use_simple_encoding: bool = False
_custom_encoder: Optional[Callable[[str], np.ndarray]] = None


def set_encoder(fn: Optional[Callable[[str], np.ndarray]]) -> None:
    """Override the default text encoder (must return a 1-D numpy float array)."""
    global _custom_encoder
    _custom_encoder = fn


def simple_text_encode(text: str, dim: int = 768) -> np.ndarray:
    """Deterministic L2-normalized fallback encoder when BERT weights are unavailable."""
    if not text or not text.strip():
        return np.zeros(dim, dtype=np.float32)

    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)
    char_count = len(text)

    features: List[float] = []
    for i in range(dim - 4):
        seed = f"{text}_{i}".encode("utf-8")
        hash_val = int(hashlib.md5(seed).hexdigest(), 16)
        features.append((hash_val % 10000) / 10000.0)

    features.append(min(word_count / 100.0, 1.0))
    features.append(min(char_count / 1000.0, 1.0))
    features.append(len(set(words)) / max(word_count, 1))
    features.append(sum(len(w) for w in words) / max(word_count, 1) / 10.0)

    arr = np.asarray(features, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr


def _init_bert_lazy() -> None:
    global _tokenizer, _bert_model, _encoder_initialized, _use_simple_encoding
    if _encoder_initialized:
        return
    _encoder_initialized = True

    mode = (os.environ.get("CHRONIC_MEM_ENCODER") or "").strip().lower()
    if mode in ("hash", "simple", "fallback"):
        _use_simple_encoding = True
        return

    candidate_dirs = [
        os.environ.get("CHRONIC_MEM_BERT_PATH") or "",
        os.path.join(_REPO_ROOT, "bert"),
        os.path.join(_PACKAGE_DIR, "bert"),
    ]
    local_dir: Optional[str] = None
    for d in candidate_dirs:
        if not d:
            continue
        cfg = os.path.join(d, "config.json")
        w_bin = os.path.join(d, "pytorch_model.bin")
        w_safe = os.path.join(d, "model.safetensors")
        if os.path.isdir(d) and os.path.isfile(cfg) and (os.path.isfile(w_bin) or os.path.isfile(w_safe)):
            local_dir = d
            break

    offline = (os.environ.get("HF_HUB_OFFLINE") == "1") or (os.environ.get("TRANSFORMERS_OFFLINE") == "1")
    if local_dir is None and offline:
        _use_simple_encoding = True
        return

    try:
        from transformers import BertModel, BertTokenizer

        source = local_dir or "bert-base-uncased"
        _tokenizer = BertTokenizer.from_pretrained(source, local_files_only=(local_dir is not None))
        _bert_model = BertModel.from_pretrained(source, local_files_only=(local_dir is not None))
        _bert_model.eval()
    except Exception:
        _use_simple_encoding = True


def bert_encode(text: str) -> np.ndarray:
    """Encode clinical text into a 768-d semantic vector."""
    if _custom_encoder is not None:
        return np.asarray(_custom_encoder(text), dtype=np.float32)

    _init_bert_lazy()
    if _use_simple_encoding or _bert_model is None or _tokenizer is None:
        return simple_text_encode(text, dim=768)

    try:
        import torch

        with torch.no_grad():
            inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = _bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            return cls_embedding.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        return simple_text_encode(text, dim=768)


def safe_extract_date(date_str: str) -> str:
    try:
        parts = str(date_str).split()
        if len(parts) >= 2 and parts[0].lower().startswith("date"):
            return parts[1]
        if len(parts) >= 1:
            return parts[0]
    except Exception:
        pass
    return ""


def safe_extract_location(loc_str: str) -> str:
    try:
        return str(loc_str).split(":")[-1].strip()
    except Exception:
        return ""


def parse_events_json(events_path: str) -> List[Dict[str, Any]]:
    """Load structured clinical records (`events.json`) into a flat event list."""
    with open(events_path, encoding="utf-8") as f:
        raw = json.load(f)

    all_events: List[Dict[str, Any]] = []
    for record in raw:
        rec_idx_default = record.get("record_index", -1)
        for ev in record.get("events", []):
            meta = ev.get("meta", {}) or {}
            parsed = {
                "date": safe_extract_date(meta.get("date", "") or ev.get("date", "")),
                "location": safe_extract_location(meta.get("location", "") or ev.get("location", "")),
                "diagnosis": ev.get("diagnosis", None),
                "symptoms": ev.get("symptoms", []),
                "treatment": ev.get("treatment", []),
                "medications": ev.get("medications", []),
                "lifestyle": ev.get("lifestyle", []),
                "people": ev.get("people", []),
                "follow_up": ev.get("follow_up", []),
                "summary": ev.get("summary", ""),
                "raw_dialogue": meta.get("raw_dialogue", ev.get("raw_dialogue", [])),
                "record_index": meta.get("record_index", rec_idx_default),
                "dialogue_index": meta.get("dialogue_index", -1),
                "is_permanent": bool(ev.get("is_permanent", False)),
                "meta": meta,
            }
            all_events.append(parsed)
    return all_events
