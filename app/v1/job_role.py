import json
import logging
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.schemas.job_role import JobRoleRecommendRequest, JobRoleRecommendResponse
from app.services.chat.gemini import gemini_generate_text
from app.services.job_role.predictor import predict_job_role

router = APIRouter(prefix="/job-role", tags=["Job Role Recomendation"])
logger = logging.getLogger(__name__)


def _extract_first_json_object(text: str) -> dict:
    if not text:
        raise ValueError("Empty Gemini response")

    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Gemini response")

    return json.loads(match.group(0))


def _build_skills_str(skillset: list[str]) -> str:
    return ", ".join([str(s).strip() for s in skillset if str(s).strip()])


def _build_greeting(name: str) -> str:
    clean_name = (name or "").strip() or name
    return f"Halo {clean_name}!"


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


@router.post("/job-role/recommend/gemini", response_model=JobRoleRecommendResponse)
async def recommend_job_role_gemini(request: JobRoleRecommendRequest):
    name = (request.name or "").strip() or request.name
    skills_str = _build_skills_str(request.skillset)
    greeting = _build_greeting(name)

    prompt = (
        "Kamu adalah career coach. Berdasarkan data user berikut, berikan 1 rekomendasi job role "
        "yang paling cocok dan alasan singkat dalam Bahasa Indonesia.\n\n"
        f"Nama: {name}\n"
        f"Skillset: {skills_str or '(kosong)'}\n\n"
        "Balas sebagai JSON murni (tanpa markdown) dengan format:\n"
        "{\"predicted_role\": string, \"recommendation\": string}\n\n"
        "Aturan: recommendation harus 1 kalimat, natural, dan menyebut skillset jika ada."
    )

    try:
        text = await gemini_generate_text(prompt)
        data = _extract_first_json_object(text)
        predicted_role = str(data.get("predicted_role") or "").strip()
        recommendation = str(data.get("recommendation") or "").strip()

        if not predicted_role:
            raise ValueError("Missing predicted_role")
        if not recommendation:
            if skills_str:
                recommendation = (
                    f"Berdasarkan skillset kamu ({skills_str}), pekerjaan yang cocok untukmu adalah "
                    f"{predicted_role}."
                )
            else:
                recommendation = (
                    f"Berdasarkan skillset kamu, pekerjaan yang cocok untukmu adalah {predicted_role}."
                )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JobRoleRecommendResponse(
        greeting=greeting,
        recommendation=recommendation,
        predicted_role=predicted_role,
        confidence=None,
    )


@router.post("/job-role/recommend/stream")
async def recommend_job_role_stream(request: JobRoleRecommendRequest):
    name = request.name.strip() or request.name
    skills_str = ", ".join([s.strip() for s in request.skillset if str(s).strip()])

    async def event_stream():
        yield f"data: {json.dumps({'status': 'pending', 'message': ''})}\n\n"

        try:
            prediction = await run_in_threadpool(predict_job_role, request.skillset)
        except Exception as exc:
            yield f"data: {json.dumps({'status': 'error', 'message': str(exc)})}\n\n"
            return

        logger.info(
            "job_role_prediction predicted_role=%s confidence=%s",
            prediction.label,
            prediction.confidence,
        )

        if skills_str:
            recommendation = (
                f"Berdasarkan skillset kamu ({skills_str}), pekerjaan yang cocok untukmu adalah "
                f"{prediction.label}."
            )
        else:
            recommendation = (
                f"Berdasarkan skillset kamu, pekerjaan yang cocok untukmu adalah {prediction.label}."
            )

        for word in recommendation.split(" "):
            word = word.strip()
            if not word:
                continue
            yield f"data: {json.dumps({'status': 'stream', 'message': word})}\n\n"

        yield f"data: {json.dumps({'status': 'done', 'message': ''})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/job-role/recommend/gemini/stream")
async def recommend_job_role_gemini_stream(request: JobRoleRecommendRequest):
    name = (request.name or "").strip() or request.name
    skills_str = _build_skills_str(request.skillset)

    prompt = (
        "Kamu adalah career coach. Berdasarkan data user berikut, berikan 1 rekomendasi job role "
        "yang paling cocok dan alasan singkat dalam Bahasa Indonesia.\n\n"
        f"Nama: {name}\n"
        f"Skillset: {skills_str or '(kosong)'}\n\n"
        "Balas sebagai JSON murni (tanpa markdown) dengan format:\n"
        "{\"predicted_role\": string, \"recommendation\": string}\n\n"
        "Aturan: recommendation harus 1 kalimat, natural, dan menyebut skillset jika ada."
    )

    async def event_stream():
        yield f"data: {json.dumps({'status': 'pending', 'message': ''})}\n\n"

        try:
            text = await gemini_generate_text(prompt)
            data = _extract_first_json_object(text)
            predicted_role = str(data.get("predicted_role") or "").strip()
            recommendation = str(data.get("recommendation") or "").strip()

            if not predicted_role:
                raise ValueError("Missing predicted_role")
            if not recommendation:
                if skills_str:
                    recommendation = (
                        f"Berdasarkan skillset kamu ({skills_str}), pekerjaan yang cocok untukmu adalah "
                        f"{predicted_role}."
                    )
                else:
                    recommendation = (
                        f"Berdasarkan skillset kamu, pekerjaan yang cocok untukmu adalah {predicted_role}."
                    )
        except Exception as exc:
            yield f"data: {json.dumps({'status': 'error', 'message': str(exc)})}\n\n"
            return

        logger.info("job_role_gemini predicted_role=%s", predicted_role)

        for word in recommendation.split(" "):
            word = word.strip()
            if not word:
                continue
            yield f"data: {json.dumps({'status': 'stream', 'message': word})}\n\n"

        yield f"data: {json.dumps({'status': 'done', 'message': ''})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
