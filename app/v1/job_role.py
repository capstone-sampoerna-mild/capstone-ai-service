import logging
from fastapi import APIRouter, HTTPException
from app.schemas.job_role import (
    JobRoleRecommendRequest,
    JobRoleRecommendResponse,
    RankedRole,
    SkillGap,
)
from app.services.job_role.predictor import rank_job_roles, get_skill_gap

router = APIRouter(prefix="/job-role", tags=["Job Role Recomendation"])
logger = logging.getLogger(__name__)

@router.post("/recommend", response_model=JobRoleRecommendResponse)
async def recommend_job_role(request: JobRoleRecommendRequest):
    try:
        ranking = rank_job_roles(request.skillset)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    top_roles = []
    for pred in ranking.top(4):
        gaps = get_skill_gap(pred.label, request.skillset)
        top_roles.append(
            RankedRole(
                role=pred.label,
                confidence=pred.confidence,
                skill_gap=[
                    SkillGap(skill=skill, confidence=conf)
                    for skill, conf in gaps[:10] 
                ],
            )
        )

    return JobRoleRecommendResponse(top_roles=top_roles)