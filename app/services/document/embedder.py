import os
from google import genai
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

def embed_and_store(chunks: list[str]):
    records = []
    
    for chunk in chunks:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=chunk
        )
        
        vector_data = result.embeddings[0].values
        
        records.append({
            "contents": chunk,
            "embedding": vector_data
        })
        
    if records:
        response = supabase.table('documents').insert(records).execute()
        return response
    
    return None