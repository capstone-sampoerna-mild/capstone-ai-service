from pydantic import BaseModel

class SkillItem(BaseModel):
    skill: str
    confidence: float

class RankedRole(BaseModel):
    role: str
    confidence: float
    user_skill: list[SkillItem]
    recommended_skill_to_learn: list[SkillItem]

class JobRoleRecommendRequest(BaseModel):
    skillset: list[str]

class JobRoleRecommendResponse(BaseModel):
    top_roles: list[RankedRole]