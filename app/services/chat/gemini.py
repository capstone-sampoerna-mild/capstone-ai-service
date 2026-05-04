import json
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
    return _client


async def gemini_generate_text(prompt: str) -> str:
    client = _get_client()
    resp = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    text = getattr(resp, "text", None)
    return (text or "").strip()

async def gemini_stream_chat(prompt: str):
    client = _get_client()
    async for chunk in client.aio.models.generate_content_stream(
        model=MODEL_ID,
        contents=prompt,
    ):
        text = getattr(chunk, "text", None)
        if text:
            json_data = json.dumps({"text": text})
            yield f"data: {json_data}\n\n"