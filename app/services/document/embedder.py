from google import genai
from supabase import create_client, Client
from app.core.config import settings

supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
client = genai.Client(api_key=settings.GEMINI_API_KEY)

def embed_and_store(chunks: list[str]):
    records = []
    
    for chunk in chunks:
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=chunk
        )
        
        vector_data = result.embeddings[0].values
        
        records.append({
            "content": chunk,
            "embedding": vector_data
        })
        
    if records:
        response = supabase.table('documents').insert(records).execute()
        return response
    
    return None