from fastapi import APIRouter
import os

router = APIRouter()

@router.get("/")
async def hello() :
    return { "message" : "hello world"}
