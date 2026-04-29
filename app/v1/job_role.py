from fastapi import APIRouter, HTTPException

from app.schemas.job_role import JobRoleRecommendRequest, JobRoleRecommendResponse
from app.services.job_role.predictor import predict_job_field

router = APIRouter()


_FIELD_TO_JOBS: dict[str, list[str]] = {
    "Data & AI": ["Data Analyst", "Data Scientist", "Machine Learning Engineer"],
    "Software Engineering": ["Backend Engineer", "Frontend Engineer", "Mobile Developer"],
    "Product & Business": ["Business Analyst", "Product Manager", "Project Manager"],
}


@router.post("/job-role/recommend", response_model=JobRoleRecommendResponse)
async def recommend_job_role(request: JobRoleRecommendRequest):
    try:
        prediction = predict_job_field(request.skillset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    name = request.name.strip() or request.name
    skills_str = ", ".join([s.strip() for s in request.skillset if str(s).strip()])

    greeting = f"Halo {name}!"
    recommended_jobs = _FIELD_TO_JOBS.get(prediction.label, [])
    jobs_str = ", ".join(recommended_jobs)
    if skills_str:
        recommendation = (
            f"Berdasarkan skillset kamu ({skills_str}), rekomendasi bidang pekerjaan: "
            f"{prediction.label}."
        )
    else:
        recommendation = (
            f"Berdasarkan skillset kamu, rekomendasi bidang pekerjaan: {prediction.label}."
        )

    if jobs_str:
        recommendation = f"{recommendation} Contoh role: {jobs_str}."

    return JobRoleRecommendResponse(
        greeting=greeting,
        recommendation=recommendation,
        predicted_field=prediction.label,
        recommended_jobs=recommended_jobs,
        confidence=prediction.confidence,
    )
