from __future__ import annotations

from pydantic import BaseModel, Field


class JobRoleRecommendRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Nama user")
    skillset: list[str] = Field(
        default_factory=list,
        description="Daftar skill (contoh: ['python', 'sql', 'ml'])",
    )


class JobRoleRecommendResponse(BaseModel):
    greeting: str
    recommendation: str
    predicted_role: str
    confidence: float | None = None
