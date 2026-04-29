from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "models" / "job_role_model.keras"

DEFAULT_ROLE_LABELS: list[str] = [
    "Data & AI",
    "Software Engineering",
    "Product & Business",
]


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
    return ", ".join(skills).lower().strip() or "(no skills)"


def _load_role_labels() -> list[str]:
    import os

    raw = (os.getenv("JOB_ROLE_LABELS") or "").strip()
    if not raw:
        return DEFAULT_ROLE_LABELS

    labels = [part.strip() for part in raw.split(",") if part.strip()]
    return labels or DEFAULT_ROLE_LABELS


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


def predict_job_field(skillset: Iterable[str]) -> JobRolePrediction:
    labels = _load_role_labels()
    model = _load_model()

    try:
        import tensorflow as tf  
    except Exception:  
        tf = None  # type: ignore

    x_text = _skills_to_text(skillset)

    x = tf.constant([x_text])

    y = model.predict(x, verbose=0)
    if isinstance(y, (list, tuple)):
        y = y[0]

    scores = y[0]
    try:
        scores_list = [float(v) for v in scores]
    except TypeError:
        scores_list = [float(v) for v in scores.numpy().tolist()]

    best_index, best_score = max(enumerate(scores_list), key=lambda it: it[1])
    label = labels[best_index] if best_index < len(labels) else f"Class {best_index}"

    return JobRolePrediction(label=label, confidence=float(best_score))
