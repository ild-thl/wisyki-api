from fastapi import APIRouter, Depends, HTTPException, Path
from starlette.requests import Request
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Tuple, Optional, Any
import requests
from ..models.SkillRetriever import SkillRetriever
from ..models.SkillProfileExtractor import (
    get_recommendation,
    extract_skill_profile,
)
from enum import Enum
import os


router = APIRouter()


def get_db(req: Request):
    return req.app.state.DB


def get_embedding_functions(req: Request):
    return req.app.state.EMBEDDING_FUNCTIONS


def get_reranker(req: Request):
    return req.app.state.RERANKER


def get_skilldb(req: Request):
    return req.app.state.SKILLDB


def get_domains(req: Request):
    return req.app.state.DOMAINS


class ProfileRequest(BaseModel):
    mistral_api_key: Optional[str] = Field(
        default=None,
        description="An API key for Mistral. If use_llm or llm_validation is true, this key enables usage of propriatary Mistral models.",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="An API key for OpenAI. If use_llm or llm_validation is true, this key enables usage of propriatary OpenAI models.",
    )


class ExtractProfileRequest(ProfileRequest):
    doc: str = Field(
        ...,
        description="The document to extract the skill profile from.",
    )


class ProfileRecommendationRequest(ProfileRequest):
    profile: BaseModel = Field(
        ...,
        description="The skill profile to get a recommendation for.",
    )


class EducationalAchievement(BaseModel):
    type: Optional[str] = Field(
        None, description="The type of educational achievement."
    )
    title: Optional[str] = Field(
        None, description="The title of the educational achievement."
    )
    field_of_study: Optional[str] = Field(None, description="The field of study.")
    institution: Optional[str] = Field(
        None, description="The institution where the degree was obtained."
    )


class WorkExperience(BaseModel):
    job_title: Optional[str] = Field(None, description="The job title.")
    duration: Optional[str] = Field(None, description="The duration of the job.")
    area_of_work: Optional[str] = Field(None, description="The area of work.")


class EQFLevel(Enum):
    EQF_1 = "EQF 1"
    EQF_2 = "EQF 2"
    EQF_3 = "EQF 3"
    EQF_4 = "EQF 4"
    EQF_5 = "EQF 5"
    EQF_6 = "EQF 6"
    EQF_7 = "EQF 7"
    EQF_8 = "EQF 8"


class LanguageProficiencyLevel(Enum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


class LanguageProficiency(BaseModel):
    language: Optional[str] = Field(None, description="The language.")
    proficiency_level: Optional[LanguageProficiencyLevel] = Field(
        None, description="The level of proficiency in the language."
    )


class SkillProfile(BaseModel):
    education_level: Optional[EQFLevel] = Field(
        None,
        description="The current education level of the person as an EQF level.",
    )
    educational_achievements: Optional[List[EducationalAchievement]] = Field(
        default=[],
        description="The educational achievements and attainments of the person. This can include degrees, courses, certifications, Microcredentials, etc.",
    )
    work_experience: Optional[List[WorkExperience]] = Field(
        default=[], description="The work experience of the person."
    )
    skills: Optional[List[str]] = Field(
        default=[],
        description="The skills of the person.",
    )
    languages: Optional[List[LanguageProficiency]] = Field(
        default=[],
        description="The languages the person knows.",
    )
    desired_skills: Optional[List[str]] = Field(
        default=[],
        description="The desired skills of the person.",
    )
    desired_occuptions: Optional[List[str]] = Field(
        default=[],
        description="The desired occupations of the person.",
    )
    interests: Optional[List[str]] = Field(
        default=[],
        description="The interests of the person. Besides professional interests, also personal interests can be included.",
    )


@router.post(
    "/extractSkillProfile",
    description="Extract a skill profile from a document.",
    response_model=SkillProfile,
)
async def post_extract_skill_profile(
    request: ExtractProfileRequest,
    db=Depends(get_db),
    embedding_functions=Depends(get_embedding_functions),
    reranker=Depends(get_reranker),
    skilldb=Depends(get_skilldb),
    domains=Depends(get_domains),
):
    if (
        request.openai_api_key is None
        and request.mistral_api_key is None
        and os.getenv("OPENAI_API_KEY")
    ):
        request.openai_api_key = os.getenv("OPENAI_API_KEY")

    profile = await extract_skill_profile(
        doc=request.doc,
        mistral_api_key=request.mistral_api_key,
        openai_api_key=request.openai_api_key,
    )

    return profile


async def predict_skills(
    los: List[str],
    embedding_function,
    reranker,
    skilldb,
    domains,
    request: ProfileRequest,
):
    skill_retriever = SkillRetriever(
        embedding_function,
        reranker,
        skilldb,
        domains,
        SkillRetrieverRequest(
            taxonomies=request.taxonomies,
            los=los,
            openai_api_key=request.openai_api_key,
            mistral_api_key=request.mistral_api_key,
            rerank=True,
            top_k=len(los),
        ),
    )

    # Predict skills for the extracted skills
    lo, lo_predictions = await skill_retriever.predict()
    lo_predictions = generate_skill_responses(lo_predictions)
    return SkillPredictionResponse(natural=lo, skills=lo_predictions)


class Recommendation(BaseModel):
    skills: Optional[List[str]] = Field(
        default=[],
        description="The recommended skills the person needs to aquire to reach their desired goals.",
    )
    occupations: Optional[List[str]] = Field(
        default=[],
        description="The recommended occupations for the person to pursue based in their talents and interests.",
    )
    education: Optional[List[str]] = Field(
        default=[],
        description="The recommended educational paths for the person to take to a aquire the desired skills and prepare for the desired occupation. This can include degrees, courses, certifications, etc.",
    )


@router.post(
    "/profileRecommendation",
    description="Get a recommendation based on a skill profile.",
    response_model=Recommendation,
)
async def post_profile_recommendation(
    request: ProfileRecommendationRequest,
    db=Depends(get_db),
    embedding_functions=Depends(get_embedding_functions),
    reranker=Depends(get_reranker),
    skilldb=Depends(get_skilldb),
    domains=Depends(get_domains),
):
    recommendation = await get_recommendation(request)

    return recommendation
