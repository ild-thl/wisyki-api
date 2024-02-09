from fastapi import APIRouter, Depends, HTTPException
from starlette.requests import Request
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Tuple, Optional, Any
import requests
from openai import AuthenticationError
from ..models.SkillRetriever import SkillRetriever


class BaseSkill(BaseModel):
    title: str
    uri: str


class SkillResponse(BaseSkill):
    score: float
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class SkillRequest(BaseSkill):
    valid: Optional[bool] = Field(default=True)


class SkillRetrieverRequest(BaseModel):
    skill_taxonomy: str = Field(default="ESCO")
    doc: Optional[str] = Field(default=None)
    los: List[str] = Field(default=[])
    skills: List[SkillRequest] = Field(default=[])
    filterconcepts: List[str] = Field(default=[])
    top_k: int = Field(default=20)
    strict: int = Field(default=0)
    trusted_score: float = Field(default=0.8)
    temperature: float = Field(default=0.1)
    use_llm: bool = Field(default=False)
    llm_validation: bool = Field(default=False)
    rerank: bool = Field(default=False)
    score_cutoff: float = Field(default=1)
    openai_api_key: Optional[str] = Field(default=None)
    mistral_api_key: Optional[str] = Field(default=None)

    # Ensure that skill_taxonomy is one of the available taxonomies.
    @validator("skill_taxonomy", pre=True, always=True)
    def check_skill_taxonomy(cls, v):
        if v not in ["ESCO", "DKZ", "GRETA"]:
            raise ValueError('skill_taxonomy must be one of "ESCO", "DKZ", "GRETA"')
        return v

    # Ensure that either doc or los is provided.
    @validator("doc", pre=True, always=True)
    def check_doc(cls, v, values):
        if "los" in values and not values["los"] and not v:
            raise ValueError('Either "doc" or "los" must be provided')
        return v

    # Ensure that either doc or los is provided.
    @validator("los", pre=True, always=True)
    def check_los(cls, v, values):
        if "doc" in values and not values["doc"] and not v:
            raise ValueError('Either "doc" or "los" must be provided')
        return v

    # Ensure that only one of llm_validation or rerank is true.
    @validator("llm_validation", pre=True, always=True)
    def check_llm_validation(cls, v, values):
        if "rerank" in values and values["rerank"] and v:
            raise ValueError('Only one of "llm_validation" or "rerank" can be true')
        return v

    # Ensure that only one of llm_validation or rerank is true.
    @validator("rerank", pre=True, always=True)
    def check_rerank(cls, v, values):
        if "llm_validation" in values and values["llm_validation"] and v:
            raise ValueError('Only one of "llm_validation" or "rerank" can be true')
        return v


class LegacySkillRetrieverRequest(SkillRetrieverRequest):
    trusted_score: float = Field(default=0.2)
    skillfit_validation: bool = Field(default=False)


class SkillRetrieverResponse(BaseModel):
    searchterms: List[str]
    results: List[SkillResponse]


class ValidationResult(BaseModel):
    uri: str
    title: str
    taxonomy: str
    valid: bool


class UpdateCourseSkillsRequest(BaseModel):
    id: Optional[str] = Field(default=None)
    doc: str
    validationResults: List[ValidationResult] = Field(default=[])


class UpdateCourseSkillsResponse(BaseModel):
    updated_courses: List[str]


class CourseSkill(BaseModel):
    course_text: str
    skill_id: str
    valid: bool


class GetCourseSkillsResponse(BaseModel):
    course_skills: List[CourseSkill]


class BaseEmbeddingRequest(BaseModel):
    model: str = Field(
        default="instructor-large",
        pattern="^(instructor-large|instructor-skillfit)$",
        description="The model field can only take the values 'instructor-large' or 'instructor-skillfit'",
    )


class GetEmbeddingsQueryRequest(BaseEmbeddingRequest):
    query: str
    query_instruction: str = Field(
        default="Represent the question for retrieving supporting documents: ",
        description="Instruction to use for embedding query.",
    )


class GetEmbeddingsDocumentsRequest(BaseEmbeddingRequest):
    docs: List[str]
    embed_instruction: str = Field(
        default="Represent the document for retrieval: ",
        description="Instruction to use for embedding documents.",
    )


router = APIRouter()


def get_db(req: Request):
    return req.app.state.DB


def get_embedding_functions(req: Request):
    return req.app.state.EMBEDDING_FUNCTIONS


def get_reranker(req: Request):
    return req.app.state.RERANKER


def get_skilldbs(req: Request):
    return req.app.state.SKILLDBS


# Legacy endpoint. Use /v2/chatsearch instead.
# In this version of the endpoint, the score is 1 - score, and 0 is the best score.
@router.post("/chatsearch", response_model=SkillRetrieverResponse)
async def chatsearch(
    request: LegacySkillRetrieverRequest,
    db=Depends(get_db),
    embedding_functions=Depends(get_embedding_functions),
    reranker=Depends(get_reranker),
    skilldbs=Depends(get_skilldbs),
):
    # set trusted_score and core_cutoff to 1- trusted_score and 1 - score_cutoff
    request.trusted_score = 1 - request.trusted_score
    request.score_cutoff = 1 - request.score_cutoff

    if request.skillfit_validation:
        request.rerank = True

    # Ensure that skill_taxonomy is one of the available taxonomies.
    available_taxonomies = skilldbs.keys()
    if request.skill_taxonomy not in available_taxonomies:
        raise HTTPException(
            status_code=400,
            detail=f'skill_taxonomy must be one of {", ".join(available_taxonomies)}',
        )

    embedding_function = embedding_functions["instructor-skillfit"]
    embedding_function.query_instruction = "Represent the learning outcome for retrieving relevant skills: ";
    embedding_function.embed_instruction = "Represent the skill for retrieval: ";

    predictor = SkillRetriever(
        embedding_function,
        reranker,
        skilldbs,
        request,
    )

    try:
        learning_outcomes, predictions = await predictor.predict()
    except requests.Timeout:
        raise HTTPException(status_code=408, detail="Request timed out.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    # predicted_skills to Skill objects
    predicted_skills = [
        SkillResponse(
            title=skill.title,
            uri=skill.uri,
            score=1 - skill.score,
            metadata=skill.metadata,
        )
        for skill in predictions
    ]

    return SkillRetrieverResponse(
        searchterms=learning_outcomes, results=predicted_skills
    )


@router.post("/v2/chatsearch", response_model=SkillRetrieverResponse)
async def chatsearch_v2(
    request: SkillRetrieverRequest,
    db=Depends(get_db),
    embedding_functions=Depends(get_embedding_functions),
    reranker=Depends(get_reranker),
    skilldbs=Depends(get_skilldbs),
):
    # Ensure that skill_taxonomy is one of the available taxonomies.
    available_taxonomies = skilldbs.keys()
    if request.skill_taxonomy not in available_taxonomies:
        raise HTTPException(
            status_code=400,
            detail=f'skill_taxonomy must be one of {", ".join(available_taxonomies)}',
        )

    embedding_function = embedding_functions["instructor-skillfit"]
    embedding_function.query_instruction = "Represent the learning outcome for retrieving relevant skills: ";
    embedding_function.embed_instruction = "Represent the skill for retrieval: ";

    predictor = SkillRetriever(
        embedding_function,
        reranker,
        skilldbs,
        request,
    )

    try:
        learning_outcomes, predictions = await predictor.predict()
    except requests.Timeout:
        raise HTTPException(status_code=408, detail="Request timed out.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    # predicted_skills to Skill objects
    predicted_skills = [
        SkillResponse(
            title=skill.title,
            uri=skill.uri,
            score=skill.score,
            metadata=skill.metadata,  # score is now 1 - score
        )
        for skill in predictions
    ]

    return SkillRetrieverResponse(
        searchterms=learning_outcomes, results=predicted_skills
    )


@router.post("/updateCourseSkills", response_model=UpdateCourseSkillsResponse)
def update_course_skills(request: List[UpdateCourseSkillsRequest], db=Depends(get_db)):
    updated_courses = []
    for item in request:
        course_id = db.update_course_skills(
            item.doc, [vr.dict() for vr in item.validationResults], item.id
        )
        updated_courses.append(course_id)

    return UpdateCourseSkillsResponse(updated_courses=updated_courses)


@router.get("/getCourseSkills", response_model=GetCourseSkillsResponse)
def get_course_skills(db=Depends(get_db)):
    course_skills = db.get_course_skills()
    return GetCourseSkillsResponse(
        course_skills=[
            CourseSkill(course_text=cs[0], skill_id=cs[1], valid=cs[2])
            for cs in course_skills
        ]
    )


@router.post("/embeddings/query", response_model=List[float])
def embed_query(
    request: GetEmbeddingsQueryRequest,
    embedding_functions=Depends(get_embedding_functions),
):
    embedding_function = embedding_functions[request.model]
    embedding_function.query_instruction = request.query_instruction
    return embedding_function.embed_query(request.query)


@router.post("/embeddings/documents", response_model=List[List[float]])
def embed_documents(
    request: GetEmbeddingsDocumentsRequest,
    embedding_functions=Depends(get_embedding_functions),
):
    embedding_function = embedding_functions[request.model]
    embedding_function.embed_instruction = request.embed_instruction
    return embedding_function.embed_documents(request.docs)
