from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document.extractor import extract_text_from_pdf
from app.services.document.chunker import chunk_text
from app.services.document.embedder import embed_and_store
from app.services.document.extract_skills_from_pdf import extract_skills_from_pdf
from app.services.job_role.predictor import rank_job_roles, get_skill_gap
from app.schemas.job_role import JobRoleRecommendResponse, RankedRole, SkillGap

router = APIRouter(prefix="/document", tags=["Document Processing"])

@router.post("/predict-pdf", response_model=JobRoleRecommendResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PDF files are accepted.")

    try:
        file_bytes = await file.read()

        # Extract & embed seperti sebelumnya
        raw_text = extract_text_from_pdf(file_bytes)
        if not raw_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text extraction failed. Ensure the PDF contains selectable text and is not a scanned image."
            )
        chunks = chunk_text(raw_text)
        embed_and_store(chunks)

        # Extract skills dari PDF pakai Gemini, lalu split jadi list
        skills_text = extract_skills_from_pdf(file_bytes)
        skillset = [s.strip() for s in skills_text.split() if s.strip()]

        if not skillset:
            raise HTTPException(status_code=422, detail="No skills could be extracted from the PDF.")

        # Predict + skill gap (sama persis kayak endpoint /recommend)
        ranking = rank_job_roles(skillset)

        top_roles = []
        for pred in ranking.top(4):
            gaps = get_skill_gap(pred.label, skillset)
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")