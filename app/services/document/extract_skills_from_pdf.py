"""
extract_skills_from_pdf.py
──────────────────────────
Ekstrak skill teknis dari CV PDF menggunakan keyword matching
terhadap skills_catalog.json — tanpa API eksternal.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from app.core.config import settings

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
    """
    Ekstrak skill teknis dari CV PDF.

    Alur:
        1. Baca teks dari PDF (via extractor.py)
        2. Normalisasi teks
        3. Match kata/frasa terhadap skills_catalog.json

    Return: string skill dinormalisasi, dipisah spasi.
            Contoh: "python docker aws react typescript"
    """
    from app.services.document.extractor import extract_text_from_pdf

    raw_text = extract_text_from_pdf(file_bytes)
    if not raw_text.strip():
        logger.warning("PDF tidak mengandung teks yang bisa diekstrak.")
        return ""

    catalog = _get_skills_catalog()

    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", raw_text.lower())
    tokens = set(normalized.split())

    found_skills: list[str] = []
    for skill in catalog:
        skill_surface = skill.replace("_", " ")
        if skill_surface in normalized or skill in tokens:
            found_skills.append(skill)

    if not found_skills:
        logger.warning("Tidak ada skill dari catalog yang ditemukan di CV ini.")

    return " ".join(sorted(set(found_skills)))