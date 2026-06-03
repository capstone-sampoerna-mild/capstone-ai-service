import logging
from fastapi import APIRouter, HTTPException
from app.schemas.job_role import (
    JobRoleRecommendRequest,
    JobRoleRecommendResponse,
    RankedRole,
    SkillItem,
)
from app.services.job_role.predictor import rank_job_roles, get_skill_gap, get_user_skill_scores

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
        user_scores = get_user_skill_scores(pred.label, request.skillset)
        gap_scores = get_skill_gap(pred.label, request.skillset)

        top_roles.append(
            RankedRole(
                role=pred.label,
                confidence=pred.confidence,
                user_skill=[
                    SkillItem(skill=skill, confidence=conf)
                    for skill, conf in user_scores[:10]
                ],
                recommended_skill_to_learn=[
                    SkillItem(skill=skill, confidence=conf)
                    for skill, conf in gap_scores[:10]
                ],
            )
        )

    return JobRoleRecommendResponse(
        top_roles=top_roles,
    )
