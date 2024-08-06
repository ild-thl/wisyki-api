from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Literal
from ..models.ComplevelPredictor import ComplevelPredictor, CompLevelResponse, PredictCompLevelRequest
from ..models.ComplevelModelTrainer import ComplevelModelTrainer

router = APIRouter()


@router.post("/predictCompLevel", response_model=CompLevelResponse, description="Predict the learning outcome competency level of a course. This endpoint is used in WISY@KI 2022 and only predicts three classes A, B, and C.")
def predict_complevel(request: PredictCompLevelRequest):
    model = ComplevelPredictor()
    prediction = model.predict(request.title, request.description)
    return CompLevelResponse(class_probability=prediction["class_probability"], level=prediction["level"], target_probability=prediction["target_probability"])

@router.get("/trainCompLevel", description="Initiate the training of the Competence Level model.")
def train_complevel():
    trainer = ComplevelModelTrainer()
    training_stats = trainer.train()
    return training_stats


@router.get("/getCompLevelReport", description="Get the training report of the Competence Level model.")
def report_complevel():
    trainer = ComplevelModelTrainer()
    report = trainer.getReport()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
