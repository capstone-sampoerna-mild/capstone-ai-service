import numpy as np
import fitz
from google import genai
from app.core.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

def extract_skills_from_pdf(file_bytes: bytes) -> str:
    """Service untuk membaca PDF dan mengekstrak skill pakai Gemini"""
    
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pdf_raw_text = "\n".join([page.get_text() for page in doc])
    
    prompt = f"Ekstrak HANYA daftar skill teknis (tools, bahasa pemrograman, framework) dari teks CV berikut. Gabungkan dalam satu baris dengan spasi:\n\n{pdf_raw_text}"
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text.strip()