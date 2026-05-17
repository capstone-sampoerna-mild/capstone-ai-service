from pydantic import BaseModel

class SkillGap(BaseModel):
    skill: str
    confidence: float 

class RankedRole(BaseModel):
    role: str
    confidence: float
    skill_gap: list[SkillGap]  

class JobRoleRecommendRequest(BaseModel):
    skillset: list[str]

class JobRoleRecommendResponse(BaseModel):
    top_roles: list[RankedRole]