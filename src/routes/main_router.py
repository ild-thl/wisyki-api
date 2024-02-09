from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class About(BaseModel):
    about: str

class Health(BaseModel):
    status: str

@router.get("/", response_model=About)
def index():
    return {"about": "This is an API providing AI-predictions for identifying learning outcomes in course descriptions."}

@router.get("/health", response_model=Health)
def health():
    return {"status": "healthy"}