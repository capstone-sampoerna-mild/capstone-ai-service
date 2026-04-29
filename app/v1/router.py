from fastapi import APIRouter
from .index import router as index_router
from .chat_ai import router as chat_ai_router
from .job_role import router as job_role_router

router = APIRouter()

router.include_router(index_router)
router.include_router(chat_ai_router)
router.include_router(job_role_router)