"""
extractor.py
────────────
Baca raw teks dari PDF bytes menggunakan PyMuPDF (fitz).
"""

from __future__ import annotations

import logging

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Ekstrak seluruh teks dari PDF bytes.

    Return: string teks gabungan semua halaman,
            atau empty string kalau PDF kosong / gagal dibaca.
    """
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error("Gagal membaca PDF: %s", e)
        return ""