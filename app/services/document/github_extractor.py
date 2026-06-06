from __future__ import annotations

import base64
import logging
from functools import lru_cache
from pathlib import Path
import json
import httpx

from google import genai
from app.core.config import settings

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.GEMINI_API_KEY)

SKILLS_CATALOG_PATH = Path(settings.MODEL_DIR) / "skills_catalog.json"

@lru_cache(maxsize=1)
def _get_skills_catalog() -> set[str]:
    if not SKILLS_CATALOG_PATH.exists():
        logger.warning("skills_catalog.json tidak ditemukan untuk GitHub extraction.")
        return set()
    with open(SKILLS_CATALOG_PATH, "r") as f:
        return set(json.load(f))

async def fetch_github_repo_data(owner: str, repo: str, token: str = "") -> str:
    """
    Tarik data Languages dan isi README.md dari GitHub API.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
        
    async with httpx.AsyncClient() as http_client:
        lang_url = f"https://api.github.com/repos/{owner}/{repo}/languages"
        lang_res = await http_client.get(lang_url, headers=headers)
        languages = list(lang_res.json().keys()) if lang_res.status_code == 200 else []

        readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        readme_res = await http_client.get(readme_url, headers=headers)
        
        readme_text = ""
        if readme_res.status_code == 200:
            data = readme_res.json()
            if "content" in data:
                readme_text = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")

    combined_info = f"Languages used: {', '.join(languages)}\n\nREADME Content:\n{readme_text[:3000]}" 
    return combined_info

async def extract_skills_from_github(owner: str, repo: str, github_token: str = "") -> str:
    """
    Ekstrak skill IT dari repo GitHub menggunakan Gemini LLM.
    """
    repo_content = await fetch_github_repo_data(owner, repo, github_token)
    
    if not repo_content.strip() or "README Content:\n" in repo_content and len(repo_content) < 50:
        return ""

    prompt = f"""
    Kamu adalah sistem ekstraksi skill IT yang sangat teliti. 
    Baca data repository GitHub berikut ini yang berisi daftar bahasa pemrograman dan teks dari README.
    
    Tugas: Ekstrak HANYA bahasa pemrograman, framework, database, dan tools IT yang SECARA LANGSUNG DIGUNAKAN untuk membangun kode di repository ini.
    
    ATURAN SANGAT PENTING:
    1. Fokus pada teknologi utama penyusun repo ini (lihat bagian 'Languages used' sebagai petunjuk utama).
    2. Berdasarkan Languages used, cari document yang biasa digunakan untuk mencantumkan nama library seperti requirements.txt, package.json 
       dan lain sebagainya yang dimana berbeda di tiap bahasa
    2. Hanya kembalikan nama skill, pisahkan dengan koma.
    3. Jangan beri penjelasan, jangan beri salam.
    4. Huruf kecil semua.
    
    Repository Data:
    {repo_content}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        llm_output = response.text.lower()
    except Exception as e:
        logger.error(f"Gagal memanggil Gemini untuk GitHub Repo {owner}/{repo}: {e}")
        return ""

    catalog = _get_skills_catalog()
    
    extracted_raw = [s.strip().replace(" ", "_") for s in llm_output.split(",")]
    
    valid_skills = {skill for skill in extracted_raw if skill in catalog}

    return " ".join(sorted(valid_skills))