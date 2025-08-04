from fastapi import APIRouter, Depends, HTTPException, Path
from starlette.requests import Request
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Tuple, Optional, Any
import requests
from openai import AuthenticationError
from ..models.SkillRetriever import SkillRetriever
from ..models.ComplevelPredictor import ComplevelPredictor, CompLevelResponse
from ..models.KeywordExtractor import KeywordExtractor
from ..models.LearningOpportunityExtractor import extract_learning_opportunity
import json
from fastapi import Body
import logging

logger = logging.getLogger("uvicorn.info")


class BaseSkill(BaseModel):
    title: str
    uri: str


class SkillResponse(BaseSkill):
    score: float = Field(
        ...,
        description="The score of the skill prediction indicating the confidence of the prediction.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the skill. Content varies based on the taxonomy.",
    )
    taxonomy: str
    source: Optional[str] = Field(
        default=None, description="The most probable source of the predicted skill."
    )


class SkillRequest(BaseSkill):
    valid: Optional[bool] = Field(
        default=True,
        description="Indicates whether the skill accurately represents the information in the supporting document. A skill is considered a valid match if it accurately reflects the skills described in the document.",
    )


class SkillRetrieverRequest(BaseModel):
    taxonomies: List[str] = Field(
        default=["ESCO"],
        description="The target taxonomies for skill predictions. Can be one or many of 'ESCO', 'DKZ', 'GRETA', 'DigCompEdu, 'DigCompESCO'.",
    )
    targets: List[str] = Field(
        default=["learning_outcomes"],
        description="The target metadata to be predicted. Can be one or more of 'learning_outcomes', 'prerequisites', 'comp_level', 'keywords'.",
    )
    doc: Optional[str] = Field(
        default=None, description="The document to retrieve skills from."
    )
    los: List[str] = Field(
        default=[], description="List of natural language learning outcomes."
    )
    prerequisites: List[str] = Field(
        default=[], description="List of natural language prerequisites."
    )
    skills: List[SkillRequest] = Field(
        default=[],
        description="A list of already validated skills. Currently these are only used for finetuning learning outcome predictions.",
    )
    filterconcepts: List[str] = Field(
        default=[],
        description="List of ESCO concepts to filter by. Predicted learning outomes will be part of these concepts.",
    )
    top_k: int = Field(
        default=20,
        description="The max number of skills to retrieve. Lowering this number can speed up the retrieval process, if combined with further validation steps.",
    )
    strict: int = Field(
        default=0, description="Strictness level for filtering prediction results."
    )
    trusted_score: float = Field(
        default=0.8,
        description="The minimum score for a skill to be considered as valid. Only used if no further validation steps are requested.",
    )
    temperature: float = Field(
        default=0.1,
        description="The temperature parameter for the llm based learning outcome extraction.",
    )
    use_llm: bool = Field(
        default=False,
        description="Whether to use a Large Language Model for learning outcome extraction.",
    )
    llm_validation: bool = Field(
        default=False,
        description="Whether to validate the skills using a Large Language Model.",
    )
    rerank: bool = Field(
        default=False,
        description="Whether to rerank the retreival results using a Cross-Encoder model.",
    )
    score_cutoff: float = Field(
        default=1,
        description="The minimum score for a skill to be included in the results.",
    )
    domain_specific_score_cutoff: float = Field(
        default=0.6,
        description="The minimum score for a domain-specific skill to be included in the results.",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="An API key for OpenAI. If use_llm or llm_validation is true, this key enables usage of propriatary OpenAI models.",
    )
    mistral_api_key: Optional[str] = Field(
        default=None,
        description="An API key for Mistral. If use_llm or llm_validation is true, this key enables usage of propriatary Mistral models.",
    )
    finetuned: bool = Field(
        default=True,
        description="Whether to use a fine-tuned model for skill retrieval.",
    )

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
    skill_taxonomy: str = Field(
        default="ESCO",
        description="The target taxonomy for skill predictions. Can be one of 'ESCO', 'DKZ', 'GRETA', 'DigCompEdu', 'DigCompESCO'.",
    )
    trusted_score: float = Field(
        default=0.2,
        description="The minimum score for a skill to be considered as valid.",
    )
    domain_specific_score_cutoff: float = Field(
        default=0.2,
        description="The minimum score for a domain-specific skill to be included in the results.",
    )
    skillfit_validation: bool = Field(
        default=False,
        description="Whether to validate the skills using a Cross-Encoder model.",
    )


class SkillPredictionResponse(BaseModel):
    natural: List[str] = Field(
        default=[],
        description="The natural language representation of the learning outcomes or prerequisites.",
    )
    skills: List[SkillResponse] = Field(default=[], description="The predicted skills.")
    comp_level: Optional[CompLevelResponse] = Field(
        default=None,
        description="The competence level of the learning outcomes or prerequisites.",
    )


class SkillRetrieverResponse(BaseModel):
    searchterms: List[str] = Field(
        default=[], description="The learning outcomes or prerequisites."
    )
    results: List[SkillResponse] = Field(
        default=[], description="The predicted skills."
    )
    stats: Optional[Dict[str, Any]] = Field(
        default=None, description="Statistics about the retrieval process."
    )


class SkillRetrieverResponseV2(BaseModel):
    learning_outcomes: Optional[SkillPredictionResponse] = Field(
        default=None, description="The predicted learning outcomes."
    )
    prerequisites: Optional[SkillPredictionResponse] = Field(
        default=None, description="The predicted prerequisites."
    )
    keywords: Optional[List[str]] = Field(
        default=None, description="The extracted keywords."
    )
    stats: Optional[Dict[str, Any]] = Field(
        default=None, description="Statistics about the retrieval process."
    )


class ValidationResult(BaseModel):
    uri: str = Field(..., description="The URI of the skill.")
    title: str = Field(..., description="The title of the skill.")
    taxonomy: str = Field(..., description="The taxonomy of the skill.")
    valid: bool = Field(
        ...,
        description="Indicates whether the skill accurately represents the information in the supporting document. A skill is considered a valid match if it accurately reflects the skills described in the document.",
    )


class UpdateCourseSkillsRequest(BaseModel):
    id: Optional[str] = Field(default=None, description="The ID of the course.")
    doc: str = Field(
        ...,
        description="The document representing the source of the validated skill predictions.",
    )
    validationResults: List[ValidationResult] = Field(
        default=[], description="The validation results for the skill predictions."
    )


class UpdateCourseSkillsResponse(BaseModel):
    updated_courses: List[str] = Field(
        default=[], description="The IDs of the updated courses."
    )


class CourseSkill(BaseModel):
    id: str = Field(..., description="The ID of the skill.")
    title: str = Field(..., description="The title of the skill.")
    taxonomy: str = Field(..., description="The taxonomy of the skill.")
    valid: bool = Field(
        ...,
        description="Indicates whether the skill accurately represents the information in the supporting document. A skill is considered a valid match if it accurately reflects the skills described in the document.",
    )


class GetCourseSkillsResponse(BaseModel):
    id: str = Field(..., description="The ID of the course.")
    doc: str = Field(
        ...,
        description="The document representing the source of the validated skill predictions.",
    )
    skills: List[CourseSkill] = Field(
        default=[], description="The courses learning outcomes as skills."
    )


class BaseEmbeddingRequest(BaseModel):
    model: str = Field(
        default="instructor-large",
        pattern="^(instructor-large|multilingual_e5_finetuned)$",
        description="The model field can only take the values 'instructor-large' or 'multilingual_e5_finetuned'",
    )


class GetEmbeddingsQueryRequest(BaseEmbeddingRequest):
    query: str = Field(..., description="The query to embed.")
    query_instruction: str = Field(
        default="query: ",
        description="Instruction to use for embedding query.",
    )


class GetEmbeddingsDocumentsRequest(BaseEmbeddingRequest):
    docs: List[str] = Field(..., description="The documents to embed.")
    embed_instruction: str = Field(
        default="passage: ",
        description="Instruction to use for embedding documents.",
    )


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


async def extract_lo(request: SkillRetrieverRequest):
    if request.use_llm and (len(request.los) <= 0 or len(request.prerequisites) <= 0):
        try:
            learning_opportunity = await extract_learning_opportunity(
                request.doc,
                request.mistral_api_key,
                request.openai_api_key,
                request.targets,
            )
            print("Extracted learning opportunity data.", learning_opportunity)
            if len(request.los) <= 0 and "learning_outcomes" in learning_opportunity:
                request.los = learning_opportunity["learning_outcomes"]
            if (
                len(request.prerequisites) <= 0
                and "prerequisites" in learning_opportunity
            ):
                request.prerequisites = learning_opportunity["prerequisites"]
        except requests.Timeout:
            raise HTTPException(status_code=408, detail="Request timed out.")
        except AuthenticationError:
            raise HTTPException(status_code=401, detail="Invalid API key.")

    return request


# Legacy endpoint. Use /v2/chatsearch instead.
# In this version of the endpoint, the score is 1 - score, and 0 is the best score.
@router.post(
    "/chatsearch",
    response_model=SkillRetrieverResponse,
    description="Legacy endpoint for compatibility with wisy@ki. Use /v2/chatsearch for more advanced features. In this version of the endpoint, the score is 1 - score, and 0 is the best score.",
)
async def chatsearch(
    request: LegacySkillRetrieverRequest,
    db=Depends(get_db),
    embedding_functions=Depends(get_embedding_functions),
    reranker=Depends(get_reranker),
    skilldb=Depends(get_skilldb),
    domains=Depends(get_domains),
):
    # set trusted_score and core_cutoff to 1- trusted_score and 1 - score_cutoff
    request.trusted_score = 1 - request.trusted_score
    request.score_cutoff = 1 - request.score_cutoff
    request.domain_specific_score_cutoff = 1 - request.domain_specific_score_cutoff

    if request.skill_taxonomy:
        request.taxonomies = [request.skill_taxonomy]

    if request.skillfit_validation:
        request.rerank = True

    if not request.finetuned:
        request.rerank = False

    embedding_function = embedding_functions["multilingual_e5_finetuned"]
    embedding_function.query_instruction = "query: "
    embedding_function.embed_instruction = "passage: "

    # request = await extract_lo(request)

    predictor = SkillRetriever(
        embedding_function,
        reranker,
        skilldb,
        domains,
        request,
    )

    try:
        learning_outcomes, predictions = await predictor.predict(
            target="learning_outcomes", get_sources="sources" in request.targets
        )
    except requests.Timeout:
        raise HTTPException(status_code=408, detail="Request timed out.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    # predicted_skills to Skill objects
    predicted_skills = [
        SkillResponse(
            title=skill.title,
            uri=skill.uri,
            score=1 - skill.score,  # score is 1 - score, and 0 is the best score
            taxonomy=skill.taxonomy,
            metadata=skill.metadata,
            source=skill.source,
        )
        for skill in predictions
    ]

    stats = {
        "used_models": predictor.used_models,
    }

    return SkillRetrieverResponse(
        searchterms=learning_outcomes, results=predicted_skills, stats=stats
    )


def generate_skill_responses(predictions):
    """
    Generate a list of SkillResponse objects based on the given predictions.

    Args:
        predictions (list): A list of Prediction objects.

    Returns:
        list: A list of SkillResponse objects.

    """
    return [
        SkillResponse(
            title=skill.title,
            uri=skill.uri,
            score=skill.score,
            taxonomy=skill.taxonomy,
            metadata=skill.metadata,
            source=skill.source,
        )
        for skill in predictions
    ]


@router.post(
    "/v2/chatsearch",
    response_model=SkillRetrieverResponseV2,
    description="""
This endpoint provides AI-based annotation of learning outcomes, prerequisites, and competence levels. 

Functionality:
- For long heterogeneous texts, a Large Language Model (LLM) can be used to extract learning outcomes and prerequisites to improve the quality of the skill retrieval.
- By default the THL provides open access to the Large Language Model Mixtral8x7b hosted in SH. Still we recommend to use your own API key to ensure availability and not overuse public resources.
- Uses a combination of specifically fine-tuned bi-encoder and cross-encoder models for best performance on the task of learning outcome based skill retrieval.
- Retrieves the most relevant skills from a vector store of standardized skills.
- Supported skill taxonomies includeESCO, DKZ, GRETA, DigCompESCO and DigCompEdu.
- Results can be further filtered by given ESCO concepts.
- Different options for faster or more elaborate valodaton. Try out different settings to find whar best suits your domain and use case.

Response:
By default, the response includes...
- a natural language representation of the learning outcomes or prerequisites
- the predicted skills for the learning outcomes or prerequisites
- the competence level of the learning outcomes or prerequisites
- statistics about the process
    """,
)
async def chatsearch_v2(
    request: SkillRetrieverRequest,
    db=Depends(get_db),
    embedding_functions=Depends(get_embedding_functions),
    reranker=Depends(get_reranker),
    skilldb=Depends(get_skilldb),
    domains=Depends(get_domains),
):
    if not request.finetuned:
        request.rerank = False

    # Set the query and embed instructions for the embedding function
    embedding_function = embedding_functions["multilingual_e5_finetuned"]
    embedding_function.query_instruction = "query: "
    embedding_function.embed_instruction = "passage: "

    # request = awaitextract_lo(request)

    # Create a SkillRetriever object
    predictor = SkillRetriever(
        embedding_function,
        reranker,
        skilldb,
        domains,
        request,
    )
    # Create a KeywordExtractor object
    extractor = KeywordExtractor(request)

    learning_outcomes, prerequisites = None, None

    try:
        lo, lo_predictions = None, []
        prereq, prereq_predictions = None, []

        # Extract learning outcomes if requested
        if "learning_outcomes" in request.targets:
            lo, lo_predictions = await predictor.predict(
                target="learning_outcomes", get_sources="sources" in request.targets
            )
            lo_predictions = generate_skill_responses(lo_predictions)

            complevelresponse = None
            if "comp_level" in request.targets and lo:
                complevelmodel = ComplevelPredictor()
                complevelprediction = complevelmodel.predict("", "\n".join(lo))
                complevelresponse = CompLevelResponse(
                    class_probability=complevelprediction["class_probability"],
                    level=complevelprediction["level"],
                    target_probability=complevelprediction["target_probability"],
                )
                predictor.add_model_stats(
                    "wisy@ki-naive-complevel",
                    "Predict course learning outcome competence level.",
                )

            learning_outcomes = SkillPredictionResponse(
                natural=lo, skills=lo_predictions, comp_level=complevelresponse
            )
        else:
            # If no learning outcomes are requested to be predicted, use the provided learning outcomes or document.
            complevelresponse = None
            if "comp_level" in request.targets:
                complevelmodel = ComplevelPredictor()
                document = "\n".join(request.los) if request.los else request.doc
                complevelprediction = complevelmodel.predict("", document)
                complevelresponse = CompLevelResponse(
                    class_probability=complevelprediction["class_probability"],
                    level=complevelprediction["level"],
                    target_probability=complevelprediction["target_probability"],
                )
                predictor.add_model_stats(
                    "wisy@ki-naive-complevel",
                    "Predict course learning outcome competence level.",
                )

            learning_outcomes = SkillPredictionResponse(
                natural=request.los, skills=[], comp_level=complevelresponse
            )

        # Extract prerequisites if requested
        if "prerequisites" in request.targets:
            prereq, prereq_predictions = await predictor.predict(
                target="prerequisites", get_sources="sources" in request.targets
            )
            prereq_predictions = generate_skill_responses(prereq_predictions)

            complevelresponse = None
            if "comp_level" in request.targets and prereq:
                complevelmodel = ComplevelPredictor()
                complevelprediction = complevelmodel.predict("", "\n".join(prereq))
                complevelresponse = CompLevelResponse(
                    class_probability=complevelprediction["class_probability"],
                    level=complevelprediction["level"],
                    target_probability=complevelprediction["target_probability"],
                )
                predictor.add_model_stats(
                    "wisy@ki-naive-complevel",
                    "Predict course prerequisites competence level.",
                )

            prerequisites = SkillPredictionResponse(
                natural=prereq, skills=prereq_predictions, comp_level=complevelresponse
            )
        elif request.prerequisites:
            # If no prerequisites are requested to be predicted, but some are provided, use the provided prerequisites.
            complevelresponse = None
            if "comp_level" in request.targets:
                complevelmodel = ComplevelPredictor()
                document = "\n".join(request.prerequisites)
                complevelprediction = complevelmodel.predict("", document)
                complevelresponse = CompLevelResponse(
                    class_probability=complevelprediction["class_probability"],
                    level=complevelprediction["level"],
                    target_probability=complevelprediction["target_probability"],
                )
                predictor.add_model_stats(
                    "wisy@ki-naive-complevel",
                    "Predict course prerequisites competence level.",
                )

            prerequisites = SkillPredictionResponse(
                natural=request.prerequisites, skills=[], comp_level=complevelresponse
            )

    except requests.Timeout:
        raise HTTPException(status_code=408, detail="Request timed out.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    # Extract keywords if requested
    keywords = None
    if "keywords" in request.targets:
        # Extract valid skills
        valid_skills = [skill.title for skill in request.skills if skill.valid]

        # Extract predicted skills
        predicted_skills = []
        if learning_outcomes:
            predicted_skills.extend(skill.title for skill in learning_outcomes.skills)
        if prerequisites:
            predicted_skills.extend(skill.title for skill in prerequisites.skills)

        # Combine valid and predicted skills
        candidate_keywords = valid_skills + predicted_skills

        # Prepare keyword document
        if request.doc:
            keyword_document = request.doc
        else:
            keyword_document = (
                f"Lernziele: {' '.join(lo)}\nVorraussetzungen: {' '.join(prereq)}"
            )

        # Extract keywords
        keywords = extractor.extract(keyword_document, candidate_keywords)

    # Generate stats
    stats = {
        "used_models": predictor.used_models + extractor.used_models,
    }

    return SkillRetrieverResponseV2(
        learning_outcomes=learning_outcomes,
        prerequisites=prerequisites,
        keywords=keywords,
        stats=stats,
    )


@router.post(
    "/updateCourseSkills",
    response_model=UpdateCourseSkillsResponse,
    description="Update the skills of a course.",
)
def update_course_skills(request: List[UpdateCourseSkillsRequest], db=Depends(get_db)):
    updated_courses = []
    for item in request:
        course_id = db.update_course_skills(
            item.doc, [vr.dict() for vr in item.validationResults], item.id
        )
        updated_courses.append(course_id)

    return UpdateCourseSkillsResponse(updated_courses=updated_courses)


@router.get(
    "/getCourseSkills/{course_id}",
    response_model=GetCourseSkillsResponse,
    description="Get the skills of a course.",
)
def get_course_skills(
    course_id: str = Path(..., description="The ID of the course."), db=Depends(get_db)
):
    skills = db.get_course_skills(course_id)
    if not skills:
        return GetCourseSkillsResponse(id=course_id, doc="", skills=[])

    return GetCourseSkillsResponse(
        id=course_id,
        doc=skills[0][0],
        skills=[
            CourseSkill(
                id=skill[1],
                title=skill[3],
                taxonomy=skill[4],
                valid=skill[2],
            )
            for skill in skills
        ],
    )


class CourseCompTrainingDataResponse(BaseModel):
    id: str
    query: str
    positive: List[str]
    negative: List[str]


class CourseCompTrainingDataRequest(BaseModel):
    skill_taxonomies: Optional[List[str]] = None


@router.get(
    "/getCourseCompTrainingData",
    response_model=List[CourseCompTrainingDataResponse],
    description="Get training data for the skill retrieval model.",
)
def get_course_comp_training_data(
    request: CourseCompTrainingDataRequest, db=Depends(get_db)
):
    training_data = db.get_course_comp_training_data(request.skill_taxonomies)

    filtered_training_data = []
    for td in training_data:
        positive = (
            list(set([s.replace(" (ESCO)", "") for s in td[2]]))
            if td[2] is not None
            else []
        )
        negative = (
            list(set([s.replace(" (ESCO)", "") for s in td[3]]))
            if td[3] is not None
            else []
        )

        if positive:
            if "local" in td[0]:
                continue
            filtered_training_data.append(
                CourseCompTrainingDataResponse(
                    id=td[0], query=td[1], positive=positive, negative=negative
                )
            )

    return filtered_training_data


@router.post(
    "/embeddings/query",
    response_model=List[float],
    description="Embed a query using the specified model.",
)
def embed_query(
    request: GetEmbeddingsQueryRequest,
    embedding_functions=Depends(get_embedding_functions),
):
    embedding_function = embedding_functions[request.model]
    embedding_function.query_instruction = request.query_instruction
    return embedding_function.embed_query(request.query)


@router.post(
    "/embeddings/documents",
    response_model=List[List[float]],
    description="Embed a list of documents using the specified model.",
)
def embed_documents(
    request: GetEmbeddingsDocumentsRequest,
    embedding_functions=Depends(get_embedding_functions),
):
    embedding_function = embedding_functions[request.model]
    embedding_function.embed_instruction = request.embed_instruction
    return embedding_function.embed_documents(request.docs)


class RerankRequest(BaseModel):
    kompetenzen: List[str]
    kurse: List[Dict[str, str]]  # [{"title": ..., "description": ...}, ...]

class RerankResponse(BaseModel):
    sorted_courses: List[Tuple[str, float]]

@router.post(
    "/rerank_courses",
    response_model=RerankResponse,
    description="Rerank courses for given competencies using the cross-encoder reranker."
)
def rerank_courses(
    request: RerankRequest,
    reranker=Depends(get_reranker)
):
    pairs = []
    for kompetenz in request.kompetenzen:
        for kurs in request.kurse:
            coursedata = kurs['title'] + " " + kurs['description']
            pairs.append((kompetenz, coursedata))

    logger.info(f"pairs to rerank: created")  
    # Nutze das bereits geladene Reranker-Modell
    scores = reranker.compute_score(pairs)
    logger.info(f"pairs to rerank: computed scores")

    # Finde den besten Score für jeden Kurs
    course_scores = {}
    pair_index = 0
    for kompetenz in request.kompetenzen:
        for kurs in request.kurse:
            score = scores[pair_index]
            pair_index += 1
            if kurs['title'] not in course_scores or score > course_scores[kurs['title']]:
                course_scores[kurs['title']] = score
    logger.info(f"sort courses by scores")
    sorted_courses = sorted(course_scores.items(), key=lambda item: item[1], reverse=True)
    logger.info(f"sorted courses")
    return RerankResponse(sorted_courses=sorted_courses)