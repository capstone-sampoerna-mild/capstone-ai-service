import json
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite")

async def gemini_stream_chat(prompt: str):
    response = await model.generate_content_async(prompt, stream=True)
    
    async for chunk in response:
        if chunk.text:
            json_data = json.dumps({"text": chunk.text})
            
            yield f"data: {json_data}\n\n"