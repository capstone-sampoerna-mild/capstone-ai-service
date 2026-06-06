# from __future__ import annotations

# import logging
# from functools import lru_cache
# from typing import Optional

# from google import genai
# from google.genai import types
# from supabase import create_client, Client

# from app.core.config import settings

# logger = logging.getLogger(__name__)

# client = genai.Client(api_key=settings.GEMINI_API_KEY)

# @lru_cache(maxsize=1)
# def _get_supabase() -> Client:
#     return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

# def embed_and_store(chunks: list[str]):
#     records = []
    
#     for chunk in chunks:
#         result = client.models.embed_content(
#             model="gemini-embedding-2",
#             contents=chunk
#         )
        
#         vector_data = result.embeddings[0].values
        
#         records.append({
#             "content": chunk,
#             "embedding": vector_data
#         })
        
#     if records:
#         response = supabase.table('documents').insert(records).execute()
#         return response
    
#     return None