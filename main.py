from __future__ import annotations

from pathlib import Path

import joblib
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parent
TF_MODEL_PATH = REPO_ROOT / "models" / "job_role_model.keras"
ENCODER_PATH = REPO_ROOT / "models" / "label_encoder.pkl"


# === ML assets (loaded at global scope) ===
tf_model = tf.keras.models.load_model(TF_MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


class JobRoleRecommendRequest(BaseModel):
    nama: str
    skillset: list[str]


app = FastAPI()


@app.post("/job-role/recommend")
def recommend_job_role(request: JobRoleRecommendRequest):
    teks_gabungan = " ".join(request.skillset).lower().strip()

    x = tf.constant([teks_gabungan], dtype=tf.string)
    y_pred = tf_model.predict(x, verbose=0)

    index = int(tf.argmax(y_pred, axis=1).numpy()[0])
    predicted_role = encoder.inverse_transform([index])[0]

    return {
        "reply": f"hai {request.nama}, pekerjaan yang cocok untukmu adalah {predicted_role}",
    }


__all__ = ["app", "tf_model", "encoder"]
