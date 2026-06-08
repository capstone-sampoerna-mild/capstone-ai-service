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
token = settings.GITHUB_PAT_TOKEN


@lru_cache(maxsize=1)
def _get_skills_catalog() -> set[str]:
    if not SKILLS_CATALOG_PATH.exists():
        return set()
    with open(SKILLS_CATALOG_PATH, "r") as f:
        return set(json.load(f))


async def fetch_github_repo_data(owner: str, repo: str, token: str = "") -> dict:
    headers = {"Accept": "application/vnd.github.mercy-preview+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    DEP_FILES = {
        "requirements.txt",
        "pyproject.toml",
        "Pipfile",
        "package.json",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "go.mod",
        "Cargo.toml",
        "composer.json",
        "Gemfile",
        "pubspec.yaml",
        "*.csproj",
        "*.sln",
        "CMakeLists.txt",
        "conanfile.txt", 
    }

    async with httpx.AsyncClient() as http_client:
        lang_url = f"https://api.github.com/repos/{owner}/{repo}/languages"
        lang_res = await http_client.get(lang_url, headers=headers)
        languages = list(lang_res.json().keys()) if lang_res.status_code == 200 else []

        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        repo_res = await http_client.get(repo_url, headers=headers)
        topics = (
            repo_res.json().get("topics", []) if repo_res.status_code == 200 else []
        )

        tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD"
        tree_res = await http_client.get(tree_url, headers=headers)
        found_dep_files = []
        if tree_res.status_code == 200:
            all_files = [item["path"] for item in tree_res.json().get("tree", [])]
            found_dep_files = [f for f in all_files if f in DEP_FILES]

        dep_content = ""
        for filename in found_dep_files:
            file_url = (
                f"https://api.github.com/repos/{owner}/{repo}/contents/{filename}"
            )
            file_res = await http_client.get(file_url, headers=headers)
            if file_res.status_code == 200:
                data = file_res.json()
                if "content" in data:
                    decoded = base64.b64decode(data["content"]).decode(
                        "utf-8", errors="ignore"
                    )[:1500]
                    dep_content += f"\n--- {filename} ---\n{decoded}"

    return {
        "languages": languages,
        "topics": topics,
        "dependencies": dep_content,
    }


async def extract_skills_from_github(
    owner: str, repo: str, github_token: str = ""
) -> str:
    repo_content = await fetch_github_repo_data(owner, repo, github_token)

    languages = repo_content.get("languages", [])
    topics = repo_content.get("topics", [])
    dependencies = repo_content.get("dependencies", "")

    if not languages and not dependencies:
        return ""

    repo_text = (
        f"Languages used: {', '.join(languages)}\n"
        f"Topics: {', '.join(topics)}\n"
        f"Dependency Files:{dependencies if dependencies else ' (tidak ditemukan)'}"
    )

    prompt = f"""
    Kamu adalah sistem ekstraksi skill IT yang sangat teliti.

    TUGAS: Ekstrak bahasa pemrograman, framework, library, database, dan tools IT dari data repository berikut.

    ATURAN:
    1. Gunakan 'Languages used' sebagai skill dasar.
    2. Gunakan isi dependency files sebagai sumber utama library dan framework.
    3. Kembalikan HANYA nama skill dipisah koma, huruf kecil, tanpa penjelasan.

    Contoh output: python, fastapi, postgresql, docker, redis

    Repository Data:
    {repo_text}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        llm_output = response.text.strip().lower()
        logger.info(f"Gemini raw output [{owner}/{repo}]: {llm_output}")
    except Exception as e:
        logger.error(f"Gagal memanggil Gemini untuk {owner}/{repo}: {e}")
        return ""

    catalog = _get_skills_catalog()
    extracted_raw = [s.strip().replace(" ", "_") for s in llm_output.split(",")]
    logger.info(f"Extracted raw: {extracted_raw}")

    valid_skills = {skill for skill in extracted_raw if skill in catalog}
    logger.info(f"Valid skills after catalog filter: {valid_skills}")

    return " ".join(sorted(valid_skills))
