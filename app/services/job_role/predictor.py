from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "models" / "job_role_model.keras"
ENCODER_PATH = REPO_ROOT / "models" / "label_encoder.pkl"


@dataclass(frozen=True)
class JobRolePrediction:
    label: str
    confidence: float | None


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


def predict_job_role(skillset: Iterable[str]) -> JobRolePrediction:
    model = _load_model()
    encoder = _load_encoder()

    try:
        import tensorflow as tf  
    except Exception:  
        tf = None  # type: ignore

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

    best_index, best_score = max(enumerate(scores_list), key=lambda it: it[1])
    try:
        label = encoder.inverse_transform([int(best_index)])[0]
    except Exception as exc:
        raise RuntimeError(
            "Failed to decode predicted class index via label_encoder.pkl. "
            "Ensure it is a compatible scikit-learn LabelEncoder."
        ) from exc

    return JobRolePrediction(label=str(label), confidence=float(best_score))


def predict_job_field(skillset: Iterable[str]) -> JobRolePrediction:
    return predict_job_role(skillset)
