"""
predictor.py — FIXED
─────────────────────
Fix: model BiLSTM_Attention_TFIDF butuh 2 input:
  1. input_text  : string (skills + title + jd)
  2. input_tfidf : float32 array (10000-dim TF-IDF vector)

Sebelumnya cuma dikasih 1 input → error "expects 2 input(s), received 1".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT      = Path(__file__).resolve().parents[3]
MODEL_PATH     = REPO_ROOT / "models" / "job_role_model.keras"
ENCODER_PATH   = REPO_ROOT / "models" / "label_encoder.pkl"
TFIDF_PATH     = REPO_ROOT / "models" / "tfidf_vectorizer.pkl"
SKILL_MAP_PATH = REPO_ROOT / "models" / "skill_per_role.json"


# ── Dataclasses (tidak berubah) ───────────────────────────────────────────────

@dataclass(frozen=True)
class JobRolePrediction:
    label: str
    confidence: float | None


@dataclass(frozen=True)
class JobRoleRanking:
    predictions: list[JobRolePrediction]

    @property
    def best(self) -> JobRolePrediction:
        return self.predictions[0]

    def top(self, n: int = 3) -> list[JobRolePrediction]:
        return self.predictions[:n]


# ── Loaders (lazy, cached) ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required.") from exc
    return tf.keras.models.load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_encoder():
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found at: {ENCODER_PATH}")
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("joblib is required.") from exc
    return joblib.load(ENCODER_PATH)


@lru_cache(maxsize=1)
def _load_tfidf():
    """Load TF-IDF vectorizer — WAJIB ada, dipakai untuk input ke-2 model."""
    if not TFIDF_PATH.exists():
        raise FileNotFoundError(
            f"tfidf_vectorizer.pkl tidak ditemukan: {TFIDF_PATH}\n"
            "Pastikan file hasil training sudah di-copy ke folder models/."
        )
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("joblib is required.") from exc
    return joblib.load(TFIDF_PATH)


@lru_cache(maxsize=1)
def _load_skill_map() -> dict[str, dict[str, int]]:
    if not SKILL_MAP_PATH.exists():
        raise FileNotFoundError(f"Skill map not found at: {SKILL_MAP_PATH}")
    with open(SKILL_MAP_PATH) as f:
        return json.load(f)


# ── Text helpers (konsisten dengan training notebook) ─────────────────────────

def _normalize_skillset(skillset: Iterable[str]) -> list[str]:
    normalized, seen = [], set()
    for raw in skillset:
        if raw is None:
            continue
        item = str(raw).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


def _skills_to_text(skillset: Iterable[str]) -> str:
    skills = _normalize_skillset(skillset)
    return " ".join(skills).lower().strip() or "(no skills)"


# ── Core inference (FIXED) ────────────────────────────────────────────────────

def _get_scores(skillset: Iterable[str]) -> tuple[list[str], list[float]]:
    """
    Return (labels, scores) sorted by score descending.

    FIX: sekarang feed 2 input ke model:
      - inp_seq   : np.array of string shape (1,)
      - inp_tfidf : TF-IDF dense vector shape (1, 10000)
    """
    import tensorflow as tf

    model   = _load_model()
    encoder = _load_encoder()
    tfidf   = _load_tfidf()

    x_text = _skills_to_text(skillset)

    # Input 1: string tensor
    inp_seq = np.array([x_text], dtype=object)

    # Input 2: TF-IDF vector (harus sama persis seperti waktu training)
    inp_tfidf = tfidf.transform([x_text]).toarray().astype(np.float32)

    # Feed keduanya sebagai list, urutan sesuai model.inputs
    y_pred = model.predict([inp_seq, inp_tfidf], verbose=0)

    if isinstance(y_pred, (list, tuple)):
        y_pred = y_pred[0]

    scores = y_pred[0]
    try:
        scores_list = [float(v) for v in scores]
    except TypeError:
        scores_list = [float(v) for v in scores.numpy().tolist()]

    labels = encoder.classes_.tolist()
    paired = sorted(zip(labels, scores_list), key=lambda it: it[1], reverse=True)
    labels_sorted, scores_sorted = zip(*paired)
    return list(labels_sorted), list(scores_sorted)


# ── Public API (tidak berubah) ────────────────────────────────────────────────

def predict_job_role(skillset: Iterable[str]) -> JobRolePrediction:
    labels, scores = _get_scores(skillset)
    return JobRolePrediction(label=labels[0], confidence=scores[0])


def rank_job_roles(skillset: Iterable[str]) -> JobRoleRanking:
    labels, scores = _get_scores(skillset)
    return JobRoleRanking(predictions=[
        JobRolePrediction(label=label, confidence=score)
        for label, score in zip(labels, scores)
    ])


def predict_job_field(skillset: Iterable[str]) -> JobRolePrediction:
    return predict_job_role(skillset)


# ── Skill gap & scoring (tidak berubah) ──────────────────────────────────────

def get_skill_gap(role: str, user_skills: Iterable[str]) -> list[tuple[str, float]]:
    skill_map  = _load_skill_map()
    role_skills = skill_map.get(role, {})
    if not role_skills:
        return []
    user_set  = {s.strip().lower() for s in user_skills if s}
    max_count = max(role_skills.values(), default=1)
    gaps = [
        (skill, round(count / max_count, 4))
        for skill, count in role_skills.items()
        if skill not in user_set
    ]
    return sorted(gaps, key=lambda x: x[1], reverse=True)


def get_user_skill_scores(role: str, user_skills: Iterable[str]) -> list[tuple[str, float]]:
    skill_map   = _load_skill_map()
    role_skills = skill_map.get(role, {})
    if not role_skills:
        return []
    max_count = max(role_skills.values(), default=1)
    user_set  = {s.strip().lower() for s in user_skills if s}
    matched = [
        (skill, round(count / max_count, 4))
        for skill, count in role_skills.items()
        if skill in user_set
    ]
    return sorted(matched, key=lambda x: x[1], reverse=True)