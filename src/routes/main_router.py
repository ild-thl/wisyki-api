from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class About(BaseModel):
    about: str

class Health(BaseModel):
    status: str

@router.get("/", response_model=About)
def index():
    return {
        "about": "This is an API providing AI-predictions learning outcomes, prerequisites, competency levels and keywords based on course descriptions. See api-docs at /docs or /redoc for more information.",
        "acknowledgement": "This API was developed as part of the project WISY@KI by the Institut für interakive System, University of apllied Science Lübeck and was funded by the Federal Ministry of Education and Research. This service uses the ESCO classification of the European Commission.",
    }

@router.get("/health", response_model=Health)
def health():
    return {"status": "healthy"}