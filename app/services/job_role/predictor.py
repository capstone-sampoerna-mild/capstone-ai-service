"""
predictor.py
─────────────
Model BiLSTM_TFIDF butuh 2 input:
    1. input_text  : np.array of string (skills text)
    2. input_tfidf : float32 array (TF-IDF vector)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT      = Path(__file__).resolve().parents[3]
MODEL_PATH     = REPO_ROOT / "models" / "job_role_model.keras"
ENCODER_PATH   = REPO_ROOT / "models" / "label_encoder.pkl"
TFIDF_PATH     = REPO_ROOT / "models" / "tfidf_vectorizer.pkl"
SKILL_MAP_PATH = REPO_ROOT / "models" / "skills_freq_per_role.json"


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


def _parse_skills(skillset: Iterable[str]) -> str:
    """
    Normalize skills list → space-separated token string.
    Format konsisten dengan parse_skills() di training notebook:
    lowercase, spasi → underscore, slash → underscore.
    """
    tokens = []
    seen   = set()
    for raw in skillset:
        if not raw:
            continue
        token = str(raw).strip().lower().replace(' ', '_').replace('/', '_')
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)
    return ' '.join(tokens)


def _get_scores(skillset: Iterable[str]) -> tuple[list[str], list[float]]:
    """Return (labels, confidences) sorted by confidence descending."""
    model   = _load_model()
    encoder = _load_encoder()
    tfidf   = _load_tfidf()

    skills_text = _parse_skills(skillset)
    if not skills_text:
        skills_text = ''

    # Input 1: sequence (string array)
    inp_seq   = np.array([skills_text], dtype=object)

    # Input 2: TF-IDF dense vector
    inp_tfidf = tfidf.transform([skills_text]).toarray().astype(np.float32)

    # Model expects [inp_seq, inp_tfidf]
    y_pred = model.predict([inp_seq, inp_tfidf], verbose=0)

    if isinstance(y_pred, (list, tuple)):
        y_pred = y_pred[0]

    try:
        scores_list = [round(float(v), 4) for v in y_pred[0]]
    except TypeError:
        scores_list = [round(float(v), 4) for v in y_pred[0].numpy().tolist()]

    labels = encoder.classes_.tolist()
    paired = sorted(zip(labels, scores_list), key=lambda it: it[1], reverse=True)
    labels_sorted, scores_sorted = zip(*paired)

    return list(labels_sorted), list(scores_sorted)


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


def get_skill_gap(role: str, user_skills: Iterable[str]) -> list[tuple[str, float]]:
    skill_map   = _load_skill_map()
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