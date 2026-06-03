from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
import json

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "models" / "job_role_model.keras"
ENCODER_PATH = REPO_ROOT / "models" / "label_encoder.pkl"
SKILL_MAP_PATH = REPO_ROOT / "models" / "skill_per_role.json"

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


def _normalize_skillset(skillset: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
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


@lru_cache(maxsize=1)
def _load_skill_map() -> dict[str, dict[str, int]]:
    if not SKILL_MAP_PATH.exists():
        raise FileNotFoundError(f"Skill map not found at: {SKILL_MAP_PATH}")
    with open(SKILL_MAP_PATH) as f:
        return json.load(f)

def get_skill_gap(role: str, user_skills: Iterable[str]) -> list[tuple[str, float]]:
    """Return list of (skill, confidence) yang belum dimiliki user untuk role tersebut."""
    skill_map = _load_skill_map()
    role_skills = skill_map.get(role, {})
    if not role_skills:
        return []

    user_set = {s.strip().lower() for s in user_skills if s}
    max_count = max(role_skills.values(), default=1)

    gaps = [
        (skill, round(count / max_count, 4))
        for skill, count in role_skills.items()
        if skill not in user_set
    ]
    return sorted(gaps, key=lambda x: x[1], reverse=True)

def get_user_skill_scores(role: str, user_skills: Iterable[str]) -> list[tuple[str, float]]:
    """Return list of (skill, confidence) dari skill user yang relevan untuk role."""
    skill_map = _load_skill_map()
    role_skills = skill_map.get(role, {})
    if not role_skills:
        return []

    max_count = max(role_skills.values(), default=1)
    user_set = {s.strip().lower() for s in user_skills if s}

    matched = [
        (skill, round(count / max_count, 4))
        for skill, count in role_skills.items()
        if skill in user_set
    ]
    return sorted(matched, key=lambda x: x[1], reverse=True)


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required to load models/job_role_model.keras. "
            "Install it in a compatible Python environment (recommended: Python 3.11/3.12)."
        ) from exc

    return tf.keras.models.load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_encoder():
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder file not found at: {ENCODER_PATH}")

    try:
        import joblib
    except Exception as exc:
        raise RuntimeError(
            "joblib is required to load models/label_encoder.pkl. "
            "Install it (pip install joblib) in your environment."
        ) from exc

    return joblib.load(ENCODER_PATH)


def _get_scores(skillset: Iterable[str]) -> tuple[list[str], list[float]]:
    """Internal: return (labels, scores) sorted by score descending."""
    model = _load_model()
    encoder = _load_encoder()

    try:
        import tensorflow as tf
    except Exception:
        tf = None

    x_text = _skills_to_text(skillset)
    x = tf.constant([x_text], dtype=tf.string)

    y_pred = model.predict(x, verbose=0)
    if isinstance(y_pred, (list, tuple)):
        y_pred = y_pred[0]

    scores = y_pred[0]
    try:
        scores_list = [float(v) for v in scores]
    except TypeError:
        scores_list = [float(v) for v in scores.numpy().tolist()]

    try:
        labels = encoder.classes_.tolist()
    except Exception as exc:
        raise RuntimeError(
            "Failed to decode class labels via label_encoder.pkl. "
            "Ensure it is a compatible scikit-learn LabelEncoder."
        ) from exc

    paired = sorted(zip(labels, scores_list), key=lambda it: it[1], reverse=True)
    labels_sorted, scores_sorted = zip(*paired)
    return list(labels_sorted), list(scores_sorted)


def predict_job_role(skillset: Iterable[str]) -> JobRolePrediction:
    """Return only the top-1 prediction (backward compatible)."""
    labels, scores = _get_scores(skillset)
    return JobRolePrediction(label=labels[0], confidence=scores[0])


def rank_job_roles(skillset: Iterable[str]) -> JobRoleRanking:
    """Return all roles ranked by confidence, highest first."""
    labels, scores = _get_scores(skillset)
    predictions = [
        JobRolePrediction(label=label, confidence=score)
        for label, score in zip(labels, scores)
    ]
    return JobRoleRanking(predictions=predictions)


def predict_job_field(skillset: Iterable[str]) -> JobRolePrediction:
    return predict_job_role(skillset)