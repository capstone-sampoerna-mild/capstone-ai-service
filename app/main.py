from fastapi import FastAPI
from dotenv import load_dotenv
from app.v1.router import router as v1_router

load_dotenv()
app = FastAPI()
app.include_router(v1_router)
