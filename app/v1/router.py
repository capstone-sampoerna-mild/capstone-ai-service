from fastapi import APIRouter
from .index import router as index_router
from .document import router as document_router
from .job_role import router as job_role_router
from .github import router as github_router

router = APIRouter()

router.include_router(index_router)
router.include_router(document_router)
router.include_router(job_role_router)
router.include_router(github_router)