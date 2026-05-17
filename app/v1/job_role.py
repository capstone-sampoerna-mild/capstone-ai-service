import json
import logging
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.schemas.job_role import JobRoleRecommendRequest, JobRoleRecommendResponse
from app.services.job_role.predictor import predict_job_role

router = APIRouter(prefix="/job-role", tags=["Job Role Recomendation"])
logger = logging.getLogger(__name__)

@router.post("/job-role/recommend", response_model=JobRoleRecommendResponse)
async def recommend_job_role(request: JobRoleRecommendRequest):
    try:
        prediction = predict_job_role(request.skillset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    name = request.name.strip() or request.name
    skills_str = ", ".join([s.strip() for s in request.skillset if str(s).strip()])

    greeting = f"Halo {name}!"
    if skills_str:
        recommendation = (
            f"Berdasarkan skillset kamu ({skills_str}), pekerjaan yang cocok untukmu adalah "
            f"{prediction.label}."
        )
    else:
        recommendation = (
            f"Berdasarkan skillset kamu, pekerjaan yang cocok untukmu adalah {prediction.label}."
        )

    return JobRoleRecommendResponse(
        greeting=greeting,
        recommendation=recommendation,
        predicted_role=prediction.label,
        confidence=prediction.confidence,
    )
