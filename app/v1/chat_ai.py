from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.chat.gemini import gemini_stream_chat
from app.schemas.chat import ChatRequest

router = APIRouter()

@router.post("/chat-ai/stream")
async def chat_stream(request : ChatRequest) :
    return StreamingResponse(gemini_stream_chat(request.prompt), media_type='text/event-stream')