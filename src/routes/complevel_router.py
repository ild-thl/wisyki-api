from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Literal
from ..models.ComplevelPredictor import ComplevelPredictor
from ..models.ComplevelModelTrainer import ComplevelModelTrainer

router = APIRouter()


class PredictCompLevelRequest(BaseModel):
    title: str = Field(default="")
    description: str = Field(default="")

class TextLabelItem(BaseModel):
    text: str = Field(...)
    label: str = Field(...)

class TrainCompLevelRequest(BaseModel):
    data: List[TextLabelItem] = Field(...)


class CompLevelResponse(BaseModel):
    class_probability: List[float] = Field(...)
    level: Literal["A", "B", "C", "D"] = Field(
        ..., description="The level must be one of 'A', 'B', 'C', or 'D'"
    )
    target_probability: float = Field(...)

    @validator("level")
    def check_level(cls, v):
        if v not in ["A", "B", "C", "D"]:
            raise ValueError('level must be one of "A", "B", "C", or "D"')
        return v


@router.post("/predictCompLevel", response_model=CompLevelResponse)
def predict_complevel(request: PredictCompLevelRequest):
    model = ComplevelPredictor()
    prediction = model.predict(request.title, request.description)
    return CompLevelResponse(class_probability=prediction["class_probability"], level=prediction["level"], target_probability=prediction["target_probability"])


@router.post("/trainCompLevel")
def train_complevel(request: TrainCompLevelRequest):
    trainer = ComplevelModelTrainer()
    training_stats = trainer.train(request.data)
    return training_stats


@router.get("/getCompLevelReport")
def report_complevel():
    trainer = ComplevelModelTrainer()
    report = trainer.getReport()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
