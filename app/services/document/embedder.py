from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import google.generativeai as genai
from supabase import create_client, Client

from app.core.config import settings

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)

@lru_cache(maxsize=1)
def _get_supabase() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

def embed_and_store(
    chunks: list[str],
    table: str = "documents",
) -> Optional[object]:
    """
    Vektorisasi chunks pakai Google GenAI lalu insert batch ke Supabase.
    """
    if not chunks:
        logger.warning("embed_and_store dipanggil dengan chunks kosong.")
        return None

    try:
        response = genai.embed_content(
            model="models/text-embedding-004", 
            content=chunks,
            task_type="retrieval_document"
        )
        vectors = response['embedding']
    except Exception as e:
        logger.error("GenAI embedding gagal: %s", e)
        raise

    records = [
        {
            "content":   chunk,
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    supabase = _get_supabase()
    response = supabase.table(table).insert(records).execute()
    logger.info("Berhasil insert %d records ke tabel '%s'.", len(records), table)
    return response