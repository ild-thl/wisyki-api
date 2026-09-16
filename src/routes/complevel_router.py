import logging

from fastapi import APIRouter, HTTPException

from ..models.CompetencyLevelClassifier import (
    CompLevelResponse,
    CompetencyLevelClassificationError,
    CompetencyLevelClassifier,
    CompetencyLevelTimeoutError,
    PredictCompLevelRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/predictCompLevel",
    response_model=CompLevelResponse,
    description="Predict the learning outcome competency level of a course. This endpoint is used in WISY@KI 2022 and only predicts three classes A, B, and C.",
)
async def predict_complevel(request: PredictCompLevelRequest):
    try:
        classifier = CompetencyLevelClassifier()
        return await classifier.classify(
            title=request.title,
            description=request.description,
            context="course",
        )
    except CompetencyLevelTimeoutError as error:
        logger.exception("Competency classification request timed out")
        raise HTTPException(
            status_code=504, detail="Competency classification timed out."
        ) from error
    except CompetencyLevelClassificationError as error:
        logger.exception(
            "Competency classification request failed: error_type=%s error=%s",
            error.__class__.__name__,
            str(error),
        )
        raise HTTPException(
            status_code=502, detail="Competency classification failed."
        ) from error
