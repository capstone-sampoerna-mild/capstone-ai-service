from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.services.document.extractor import extract_text_from_pdf

logger = logging.getLogger(__name__)

SKILLS_CATALOG_PATH = Path(settings.MODEL_DIR) / "skills_catalog.json"

@lru_cache(maxsize=1)
def _get_skills_catalog() -> set[str]:
    """Load skills catalog sekali, reuse selamanya."""
    if not SKILLS_CATALOG_PATH.exists():
        raise FileNotFoundError(f"skills_catalog.json tidak ditemukan: {SKILLS_CATALOG_PATH}")
    with open(SKILLS_CATALOG_PATH, "r") as f:
        return set(json.load(f))

def extract_skills_from_pdf(file_bytes: bytes) -> str:
    raw_text = extract_text_from_pdf(file_bytes)
    if not raw_text.strip():
        logger.warning("PDF tidak mengandung teks yang bisa diekstrak.")
        return ""

    catalog = _get_skills_catalog()

    normalized = re.sub(r"[^a-zA-Z0-9\+#\s]", " ", raw_text.lower())
    
    found_skills = {
        skill for skill in catalog
        if skill.replace("_", " ") in normalized or skill in normalized.split()
    }

    if not found_skills:
        logger.warning("Tidak ada skill dari catalog yang ditemukan di CV ini.")

    return " ".join(sorted(found_skills))