import re
import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.document.github_extractor import extract_skills_from_github
from app.services.job_role.predictor import (
    rank_job_roles,
    get_skill_gap,
    get_user_skill_scores,
)
from app.schemas.job_role import (
    JobRoleRecommendResponse,
    RankedRole,
    SkillItem,
)
from app.core.config import settings

router = APIRouter(prefix="/document", tags=["Github Repository Extraction"])

class GithubPredictRequest(BaseModel):
    github_url: str

@router.post("/predict-github", response_model=JobRoleRecommendResponse)
async def predict_from_github(payload: GithubPredictRequest):
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+)/?", payload.github_url)
    if not match:
        raise HTTPException(
            status_code=400,
            detail="Format URL tidak valid. Harus berupa https://github.com/owner/repo",
        )
    
    owner, repo = match.groups()
    repo = repo.replace(".git", "") 

    try:
        github_token = getattr(settings, "GITHUB_TOKEN", "")
        skills_text = await extract_skills_from_github(owner, repo, github_token)
        
        skillset = [s.strip() for s in skills_text.split() if s.strip()]
        if not skillset:
            raise HTTPException(
                status_code=422,
                detail="Tidak ada skill IT yang bisa diekstrak dari repository ini.",
            )

        ranking = rank_job_roles(skillset)
        top_roles = []

        for pred in ranking.top(4):
            user_scores = get_user_skill_scores(pred.label, skillset)
            gap_scores = get_skill_gap(pred.label, skillset)

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
            extracted_skills=skillset,
            top_roles=top_roles,
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")