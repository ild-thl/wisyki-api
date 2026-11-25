import asyncio
import json
import logging
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

import psycopg2
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..models.elm_loq_metadata import (
    ElmLoqMetadataRequest,
    ElmLoqMetadataResponse,
    MetadataModel,
)
from ..models.ElmLoqMetadataRecommender import (
    ElmLoqMetadataRecommender,
    RecommendationError,
)
from ..models.metadata_job_store import (
    JobQueueFull,
    MetadataJobStore,
    valid_job_id,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class MetadataJobAccepted(MetadataModel):
    job_id: str
    job_token: str
    status_url: str
    events_url: str


class MetadataJobStatus(MetadataModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str
    progress: int
    created_at: str
    started_at: Optional[str] = None
    queue_wait_ms: Optional[int] = None
    updated_at: str
    completed_at: Optional[str] = None
    result: Optional[ElmLoqMetadataResponse] = None
    error: Optional[dict[str, Any]] = None


def get_metadata_store(request: Request) -> Optional[MetadataJobStore]:
    return getattr(request.app.state, "METADATA_STORE", None)


def get_recommender(request: Request) -> ElmLoqMetadataRecommender:
    try:
        return ElmLoqMetadataRecommender(
            embedding=request.app.state.EMBEDDING_FUNCTION,
            collection=request.app.state.SKILL_COLLECTION,
            metadata_store=get_metadata_store(request),
        )
    except ValueError as error:
        logger.error("Invalid metadata LLM configuration.")
        raise HTTPException(
            status_code=503, detail="Invalid metadata LLM configuration."
        ) from error


@router.post(
    "/recommend/elm-loq-metadata",
    response_model=ElmLoqMetadataResponse | MetadataJobAccepted,
    summary="Recommend learning outcomes and their related framework skills",
    description=(
        "Recommends framework skills for supplied learning outcomes or outcomes "
        "extracted from course information, and can generate summaries, translations "
        "and course-level classifications. Results are proposals for human review, "
        "not a complete QualityLink record or JSON-LD.\n\n"
        "Request parameters:\n"
        "- `sourceLanguage` (required): BCP 47 language tag for the source text.\n"
        "- `course`: course title and context; `description` supplies extraction "
        "evidence.\n"
        "- `learningOutcome`: authored outcomes; when omitted, outcomes are extracted "
        "unless extraction is disabled.\n"
        "- `learningOutcomeSummary`: existing source-language summary to preserve.\n"
        "- `targetLanguages`: languages for generated text; an empty list uses "
        "`sourceLanguage` without a translation call.\n"
        "- `purpose`: wording for a learning opportunity or credential; neither "
        "asserts learner attainment.\n"
        "- `options.frameworks`: skill frameworks to search. `candidateLimit` bounds "
        "retrieval per outcome and framework; accepted results are not capped.\n"
        "- `options.extraction`, `validation`, `evidenceMode` and `summary`: control "
        "extraction, skill validation, evidence strictness and summary generation. "
        "Semantic validation does not use an LLM.\n"
        "- `options.minSimilarity`: cosine-similarity floor before validation "
        "(default `0.4`); not a confidence guarantee.\n"
        "- `options.classifications`: optional course-level `EQF` and/or `ISCED_F` "
        "recommendations. EQF is not an official qualification fact.\n"
        "- `options.stats`: `none` (default), `summary` or `full` usage statistics.\n\n"
        "`relatedESCOSkill` and `relatedSkill` map to ELM skill predicates; "
        "`learningOutcomeSummary` maps to `elm:Note`. Outcome objects contain no "
        "identifiers, origin or evidence quotes. LLM validation uses full course "
        "context and returns reasons only for accepted links. Send `Prefer: "
        "respond-async` to receive a job handle instead of waiting for results."
    ),
    responses={
        200: {
            "description": "Recommendations completed.",
            "content": {
                "application/json": {
                    "example": {
                        "learningOutcome": [
                            {
                                "title": {"en": "Explain basic first aid procedures."},
                                "reason": None,
                                "relatedESCOSkill": [
                                    {
                                        "uri": "http://data.europa.eu/esco/skill/example",
                                        "title": "Provide first aid",
                                        "framework": "ESCO",
                                        "similarityScore": 0.82,
                                        "validation": {
                                            "method": "semantic",
                                            "reason": "The skill describes the same practical capability.",
                                        },
                                    }
                                ],
                                "relatedSkill": [],
                            }
                        ],
                        "learningOutcomeSummary": None,
                        "warnings": [],
                        "classificationSuggestions": None,
                        "stats": None,
                    }
                }
            },
        },
        202: {
            "model": MetadataJobAccepted,
            "description": "Recommendation accepted for background processing.",
        },
        429: {"description": "The request or job queue is at capacity."},
        502: {"description": "Invalid upstream response."},
        503: {"description": "Required provider or framework unavailable."},
        504: {"description": "Recommendation timed out."},
    },
)
async def recommend_elm_loq_metadata(
    response: Response,
    http_request: Request,
    request: ElmLoqMetadataRequest = Body(
        ...,
        openapi_examples={
            "providedOutcomes": {
                "summary": "Recommend skills for supplied outcomes",
                "value": {
                    "sourceLanguage": "en",
                    "learningOutcome": [
                        {"title": "Explain basic first aid procedures."}
                    ],
                    "options": {
                        "frameworks": ["ESCO"],
                        "validation": "semantic",
                        "summary": "disabled",
                    },
                },
            },
            "extractFromCourse": {
                "summary": "Extract outcomes and suggest classifications",
                "value": {
                    "course": {
                        "title": "First Aid Basics",
                        "description": (
                            "Participants learn to assess common emergencies, "
                            "provide basic first aid and contact emergency services."
                        ),
                    },
                    "sourceLanguage": "en",
                    "targetLanguages": ["de"],
                    "options": {
                        "frameworks": ["ESCO", "DigComp"],
                        "classifications": ["EQF", "ISCED_F"],
                    },
                },
            },
        },
    ),
    recommender: ElmLoqMetadataRecommender = Depends(get_recommender),
    store: Optional[MetadataJobStore] = Depends(get_metadata_store),
) -> ElmLoqMetadataResponse | MetadataJobAccepted | JSONResponse:
    request_id = str(uuid4())
    response.headers["X-Request-ID"] = request_id
    if "respond-async" in {
        preference.split(";", 1)[0].strip().lower()
        for preference in http_request.headers.get("Prefer", "").split(",")
    }:
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="Background recommendation jobs are unavailable.",
            )
        try:
            job_id, token, _ = await asyncio.to_thread(
                store.create_job, request.model_dump(by_alias=True, exclude_none=True)
            )
        except JobQueueFull as error:
            raise HTTPException(
                status_code=429,
                detail="The recommendation job queue is full. Retry later.",
                headers={"Retry-After": "5"},
            ) from error
        except (psycopg2.Error, ValueError) as error:
            logger.error("Metadata job submission failed: request_id=%s", request_id)
            raise HTTPException(
                status_code=503, detail="The recommendation job store is unavailable."
            ) from error
        status_url = str(http_request.url_for("get_metadata_job", job_id=job_id))
        events_url = str(http_request.url_for("stream_metadata_job", job_id=job_id))
        response.status_code = 202
        response.headers["Location"] = status_url
        response.headers["X-Job-Token"] = token
        response.headers["Cache-Control"] = "no-store"
        return MetadataJobAccepted(
            job_id=job_id,
            job_token=token,
            status_url=status_url,
            events_url=events_url,
        )
    sync_slot = False
    if store is not None:
        try:
            sync_slot = await asyncio.to_thread(
                store.try_acquire_sync_slot, UUID(request_id)
            )
        except psycopg2.Error as error:
            logger.error(
                "Metadata admission coordinator failed: request_id=%s", request_id
            )
            raise HTTPException(
                status_code=503,
                detail="The recommendation admission coordinator is unavailable.",
            ) from error
        if not sync_slot:
            raise HTTPException(
                status_code=429,
                detail="The recommendation service is at capacity. Retry later.",
                headers={"Retry-After": "5", "X-Request-ID": request_id},
            )
    try:
        result = await recommender.recommend(request)
        if isinstance(result, Response):
            return result
        if result.stats is None:
            return JSONResponse(
                content=result.model_dump(
                    mode="json", by_alias=True, exclude={"stats"}
                ),
                headers={"X-Request-ID": request_id},
            )
        return result
    except RecommendationError as error:
        logger.warning(
            "Metadata recommendation failed: request_id=%s error_type=%s",
            request_id,
            type(error).__name__,
        )
        if error.stats is not None:
            headers = {"X-Request-ID": request_id}
            if getattr(error, "retry_after_seconds", None) is not None:
                headers["Retry-After"] = str(error.retry_after_seconds)
            return JSONResponse(
                status_code=error.status_code,
                content={
                    "detail": str(error),
                    "stats": error.stats.model_dump(by_alias=True),
                },
                headers=headers,
            )
        headers = {"X-Request-ID": request_id}
        if getattr(error, "retry_after_seconds", None) is not None:
            headers["Retry-After"] = str(error.retry_after_seconds)
        raise HTTPException(
            status_code=error.status_code,
            detail=str(error),
            headers=headers,
        ) from error
    finally:
        if sync_slot and store is not None:
            try:
                await asyncio.to_thread(store.release_sync_slot, UUID(request_id))
            except psycopg2.Error:
                logger.error(
                    "Metadata admission release failed: request_id=%s", request_id
                )


def require_job_credentials(
    job_id: str,
    job_token: Optional[str],
    store: Optional[MetadataJobStore],
) -> tuple[Any, str]:
    identifier = valid_job_id(job_id)
    if identifier is None or not job_token or store is None:
        raise HTTPException(status_code=404, detail="Recommendation job not found.")
    return identifier, job_token


@router.get(
    "/recommend/elm-loq-metadata/jobs/{job_id}",
    response_model=MetadataJobStatus,
    name="get_metadata_job",
)
async def get_metadata_job(
    job_id: str,
    response: Response,
    job_token: Optional[str] = Header(default=None, alias="X-Job-Token"),
    store: Optional[MetadataJobStore] = Depends(get_metadata_store),
) -> dict[str, Any]:
    identifier, token = require_job_credentials(job_id, job_token, store)
    if store is None:
        raise HTTPException(status_code=404, detail="Recommendation job not found.")
    try:
        job = await asyncio.to_thread(store.get_job, identifier, token)
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=503, detail="The recommendation job store is unavailable."
        ) from error
    if job is None:
        raise HTTPException(status_code=404, detail="Recommendation job not found.")
    response.headers["Cache-Control"] = "no-store"
    return job


@router.get(
    "/recommend/elm-loq-metadata/jobs/{job_id}/events",
    name="stream_metadata_job",
)
async def stream_metadata_job(
    job_id: str,
    http_request: Request,
    job_token: Optional[str] = Header(default=None, alias="X-Job-Token"),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
    store: Optional[MetadataJobStore] = Depends(get_metadata_store),
) -> StreamingResponse:
    identifier, token = require_job_credentials(job_id, job_token, store)
    if store is None:
        raise HTTPException(status_code=404, detail="Recommendation job not found.")
    try:
        cursor = max(0, int(last_event_id or "0"))
        initial = await asyncio.to_thread(store.get_events, identifier, token, cursor)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid Last-Event-ID.") from error
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=503, detail="The recommendation job store is unavailable."
        ) from error
    if initial is None:
        raise HTTPException(status_code=404, detail="Recommendation job not found.")

    async def events():
        nonlocal cursor
        last_heartbeat = asyncio.get_running_loop().time()
        send_initial = True
        while not await http_request.is_disconnected():
            if send_initial:
                rows = initial
                send_initial = False
            else:
                rows = await asyncio.to_thread(
                    store.get_events, identifier, token, cursor
                )
            if rows is None:
                return
            if not rows:
                status = await asyncio.to_thread(
                    store.get_job, identifier, token, False
                )
                if status is None:
                    return
                if status["status"] in ("succeeded", "failed"):
                    payload = json.dumps(status, separators=(",", ":"))
                    yield f"event: terminal\ndata: {payload}\n\n"
                    return
            for event in rows:
                cursor = event["id"]
                payload = json.dumps(event["payload"], separators=(",", ":"))
                yield f"id: {cursor}\nevent: {event['event_type']}\ndata: {payload}\n\n"
                if event["event_type"] in ("completed", "failed"):
                    return
            if asyncio.get_running_loop().time() - last_heartbeat >= 15:
                yield ": keep-alive\n\n"
                last_heartbeat = asyncio.get_running_loop().time()
            await asyncio.sleep(1)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
