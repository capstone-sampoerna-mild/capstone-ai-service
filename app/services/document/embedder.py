from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from supabase import create_client, Client

from app.core.config import settings

logger = logging.getLogger(__name__)

TFIDF_PATH = Path(settings.MODEL_DIR) / "tfidf_vectorizer.pkl"


@lru_cache(maxsize=1)
def _get_tfidf():
    """Load TF-IDF vectorizer sekali, reuse selamanya."""
    if not TFIDF_PATH.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer tidak ditemukan: {TFIDF_PATH}")
    return joblib.load(TFIDF_PATH)


@lru_cache(maxsize=1)
def _get_supabase() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


def embed_and_store(
    chunks: list[str],
    table: str = "documents",
) -> Optional[object]:
    """
    Vektorisasi chunks pakai TF-IDF lalu insert batch ke Supabase.

    Return: Supabase response, atau None kalau chunks kosong.
    """
    if not chunks:
        logger.warning("embed_and_store dipanggil dengan chunks kosong.")
        return None

    tfidf = _get_tfidf()

    try:
        vectors: np.ndarray = tfidf.transform(chunks).toarray().astype(np.float32)
    except Exception as e:
        logger.error("TF-IDF transform gagal: %s", e)
        raise

    records = [
        {
            "content":   chunk,
            "embedding": vector.tolist(),
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    supabase = _get_supabase()
    response = supabase.table(table).insert(records).execute()
    logger.info("Berhasil insert %d records ke tabel '%s'.", len(records), table)
    return response