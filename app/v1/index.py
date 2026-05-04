from fastapi import APIRouter

router = APIRouter(tags=["Default"])

@router.get("/")
async def hello() :
    return { "message" : "hello world"}
