from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from pydantic.alias_generators import to_camel

MetadataStage = Literal[
    "extraction", "validation", "summary", "translation", "classification"
]
DEFAULT_MIN_SIMILARITY = 0.4
Title = Annotated[str, StringConstraints(min_length=1, max_length=1000, pattern=r"\S")]
Text = Annotated[str, StringConstraints(min_length=1, max_length=4000, pattern=r"\S")]
Language = Annotated[
    str,
    StringConstraints(
        min_length=2,
        max_length=35,
        pattern=r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*$",
    ),
]
Framework = Literal[
    "ESCO", "DigCompESCO", "GRETA", "DigComp", "DigCompEdu", "FutureSkills2030"
]


class MetadataModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
        allow_inf_nan=False,
    )


LocalizedTextMap = dict[Language, Text]
LocalizedTitleMap = dict[Language, Title]


class CourseInput(MetadataModel):
    title: Optional[Title] = Field(
        default=None,
        description="Course title in sourceLanguage.",
    )
    short_description: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=4000, pattern=r"\S")]
    ] = Field(default=None, description="Additional course context.")
    description: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=12000, pattern=r"\S")]
    ] = Field(
        default=None,
        description=(
            "Source text used to extract learning outcomes; required when outcomes "
            "are not supplied in strict mode. Inferred mode can use an identified course title."
        ),
    )
    learning_outcome_summary: Optional[Text] = Field(
        default=None,
        description="Course learning outcome summary in sourceLanguage.",
    )
    target_group: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=4000, pattern=r"\S")]
    ] = Field(default=None, description="Intended learner group; context only.")
    entry_requirement: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=4000, pattern=r"\S")]
    ] = Field(default=None, description="Entry requirements; context only.")
    course_content_summary: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=8000, pattern=r"\S")]
    ] = Field(default=None, description="Summary of course content; context only.")


class OutcomeInput(MetadataModel):
    title: Title = Field(description=("Authored outcome title in sourceLanguage."))


class RecommendationOptions(MetadataModel):
    frameworks: list[Framework] = Field(
        default_factory=lambda: ["ESCO"],
        min_length=1,
        max_length=6,
        description="Skill frameworks to search; defaults to ESCO.",
    )
    candidate_limit: int = Field(
        default=12,
        ge=1,
        le=48,
        strict=True,
        description="Maximum retrieved candidates per outcome and framework, before similarity filtering. Does not cap accepted skills.",
    )
    extraction: Literal["auto", "disabled"] = Field(
        default="auto",
        description="Extract outcomes when none are supplied, or require supplied outcomes.",
    )
    validation: Literal["llm", "semantic"] = Field(
        default="llm",
        description="Use LLM validation or accept candidates by the similarity threshold only.",
    )
    evidence_mode: Literal["strict", "inferred"] = Field(
        default="strict",
        description="Strict requires clear source support. Inferred permits justified contextual assumptions during extraction and LLM validation, not claims of attainment or formal recognition.",
    )
    summary: Literal["auto", "disabled"] = Field(
        default="auto",
        description="Preserve a supplied summary or generate one when absent.",
    )
    classifications: list[Literal["EQF", "ISCED_F"]] = Field(
        default_factory=list,
        max_length=2,
        description=(
            "Optional course-level recommendations. EQF is a candidate level, not an "
            "official qualification fact; ISCED_F uses the imported EU concept scheme."
        ),
    )
    stats: Literal["none", "summary", "full"] = Field(
        default="none",
        description=(
            "none omits statistics, summary returns aggregate metrics, and full "
            "includes per-call and token usage."
        ),
    )
    min_similarity: Optional[float] = Field(
        default=DEFAULT_MIN_SIMILARITY,
        ge=-1,
        le=1,
        strict=True,
        description=(
            "Cosine similarity floor applied before validation. Defaults to 0.4; "
            "this is not a calibrated confidence or relevance guarantee."
        ),
    )

    @model_validator(mode="after")
    def validate_options(self) -> "RecommendationOptions":
        if len(set(self.frameworks)) != len(self.frameworks):
            raise ValueError("Frameworks must be unique.")
        if len(set(self.classifications)) != len(self.classifications):
            raise ValueError("Classifications must be unique.")
        return self

    @property
    def effective_min_similarity(self) -> float:
        return (
            self.min_similarity
            if self.min_similarity is not None
            else DEFAULT_MIN_SIMILARITY
        )


class ElmLoqMetadataRequest(MetadataModel):
    course: Optional[CourseInput] = None
    source_language: Language = Field(
        description="BCP 47 language tag identifying the source course/outcome text."
    )
    target_languages: list[Language] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Requested output languages. An empty list returns generated text in "
            "sourceLanguage."
        ),
    )
    purpose: Literal["learning_opportunity", "credential"] = Field(
        default="learning_opportunity",
        description=(
            "Controls wording: potential outcomes for an opportunity, or neutral "
            "capability statements for a credential. It does not assert learner "
            "attainment."
        ),
    )
    learning_outcome: list[OutcomeInput] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Optional authored outcomes. When omitted, outcomes are extracted from "
            "course information unless extraction is disabled."
        ),
    )
    learning_outcome_summary: Optional[Text] = Field(
        default=None,
        description=(
            "Existing summary in sourceLanguage. Preserved instead of "
            "generating a replacement."
        ),
    )
    options: RecommendationOptions = Field(
        default_factory=RecommendationOptions,
        description="Retrieval, validation, generation, and response-statistics options.",
    )

    @model_validator(mode="after")
    def validate_source(self) -> "ElmLoqMetadataRequest":
        if not self.learning_outcome:
            if self.options.extraction == "disabled":
                raise ValueError("Supplied learningOutcome values are required.")
            if self.course is None or not (
                self.course.description is not None
                or (
                    self.options.evidence_mode == "inferred"
                    and self.course.title is not None
                )
            ):
                raise ValueError(
                    "A course description or learningOutcome is required; inferred mode also accepts a course title."
                )
        if len(set(self.target_languages)) != len(self.target_languages):
            raise ValueError("Target languages must be unique.")
        return self

    @property
    def output_languages(self) -> list[Language]:
        return self.target_languages or [self.source_language]

    @property
    def primary_language(self) -> Language:
        return self.source_language


class ExtractedOutcome(MetadataModel):
    title: Title
    reason: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=240, pattern=r"\S")]
    ] = None


class ExtractionResult(MetadataModel):
    learning_outcome: list[ExtractedOutcome] = Field(max_length=20)
    learning_outcome_summary: Optional[Text] = None


class ProposedClassifications(MetadataModel):
    eqf_level: int | None = Field(default=None, ge=1, le=8)
    isced_f_code: str | None = Field(default=None, pattern=r"^\d{4}$")
    eqf_reason: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=1000, pattern=r"\S")]
    ] = None
    isced_f_reason: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=1000, pattern=r"\S")]
    ] = None

    @model_validator(mode="after")
    def require_reasons_for_selected_classifications(self) -> "ProposedClassifications":
        if self.eqf_level is not None and not self.eqf_reason:
            raise ValueError("An EQF proposal requires a brief rationale.")
        if self.isced_f_code is not None and not self.isced_f_reason:
            raise ValueError("An ISCED-F proposal requires a brief rationale.")
        return self


class ClassificationResult(MetadataModel):
    eqf_level: int | None = Field(default=None, ge=1, le=8)
    isced_f_code: str | None = Field(default=None, pattern=r"^\d{4}$")
    eqf_reason: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=1000, pattern=r"\S")]
    ] = None
    isced_f_reason: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=1000, pattern=r"\S")]
    ] = None

    @model_validator(mode="after")
    def require_reasons_for_selected_classifications(self) -> "ClassificationResult":
        if self.eqf_level is not None and not self.eqf_reason:
            raise ValueError("An EQF proposal requires a brief rationale.")
        if self.isced_f_code is not None and not self.isced_f_reason:
            raise ValueError("An ISCED-F proposal requires a brief rationale.")
        return self


class ClassifiedExtractionResult(ExtractionResult):
    classifications: ProposedClassifications = Field(
        default_factory=ProposedClassifications
    )


class SupportedSkillDecision(MetadataModel):
    pair_id: str
    reason: str = Field(min_length=1, max_length=240, pattern=r"\S")


class ValidationResult(MetadataModel):
    supported: list[SupportedSkillDecision] = Field(max_length=96)


class SummaryResult(MetadataModel):
    text: Text


class LocalizedOutcome(MetadataModel):
    id: str
    title: LocalizedTitleMap


class LocalizationResult(MetadataModel):
    learning_outcomes: list[LocalizedOutcome] = Field(max_length=20)
    learning_outcome_summary: Optional[LocalizedTextMap] = None


class EQFLevelSuggestion(MetadataModel):
    level: int = Field(ge=1, le=8)
    uri: str
    title: str
    reason: str


class ISCEDFSuggestion(MetadataModel):
    uri: str
    code: str = Field(pattern=r"^\d{4}$")
    title: str
    similarity_score: Optional[float] = Field(default=None, ge=-1, le=1)
    reason: str


class ClassificationSuggestions(MetadataModel):
    eqf_level: Optional[EQFLevelSuggestion] = None
    isced_f: list[ISCEDFSuggestion] = Field(default_factory=list, max_length=1)


class SkillValidation(MetadataModel):
    method: Literal["llm", "semantic"]
    reason: str


class RecommendedSkill(MetadataModel):
    uri: str
    title: str
    framework: Framework
    similarity_score: float
    validation: SkillValidation


class LearningOutcomeRecommendation(MetadataModel):
    title: LocalizedTitleMap
    reason: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=240, pattern=r"\S")]
    ] = Field(
        default=None,
        description="Extraction rationale; inferred mode identifies contextual assumptions. Not verified evidence.",
    )
    related_esco_skill: list[RecommendedSkill] = Field(
        default_factory=list, alias="relatedESCOSkill"
    )
    related_skill: list[RecommendedSkill] = Field(default_factory=list)


class RecommendedOutcome(LearningOutcomeRecommendation):
    id: str


class RecommendedSummary(MetadataModel):
    text: LocalizedTextMap


class RecommendationWarning(MetadataModel):
    code: str
    message: str
    outcome_index: Optional[int] = Field(
        default=None,
        description="Zero-based index into learningOutcome when a warning is specific to one outcome.",
    )


class ModelUse(MetadataModel):
    stage: str
    model: str


class TokenUsage(MetadataModel):
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    cached_input_tokens: Optional[int] = Field(default=None, ge=0)
    reasoning_tokens: Optional[int] = Field(default=None, ge=0)


class LlmCall(MetadataModel):
    id: int
    stage: MetadataStage
    requested_model: str
    model: str
    attempt: int
    repair: bool
    status: Literal["pending", "received", "error", "cancelled"] = "pending"
    started_at_ms: int
    duration_ms: int = 0
    prompt_characters: int
    error_type: Optional[str] = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class UsageSummary(MetadataModel):
    request_count: int = 0
    response_count: int = 0
    retry_count: int = 0
    repair_count: int = 0
    call_duration_ms: int = 0
    unknown_usage_requests: int = 0
    usage_complete: bool = True
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class LlmUsage(MetadataModel):
    totals: UsageSummary = Field(default_factory=UsageSummary)
    by_stage: dict[str, UsageSummary] = Field(default_factory=dict)
    by_model: dict[str, UsageSummary] = Field(default_factory=dict)
    calls: list[LlmCall] = Field(default_factory=list)


class RecommendationStats(MetadataModel):
    pipeline_version: str = "elm-loq-metadata-v4"
    validation: Literal["llm", "semantic"]
    extraction: Literal["provided", "extracted"]
    summary: Literal["provided", "generated", "disabled", "no_outcomes"]
    score_metric: str = "cosine_similarity"
    candidates_retrieved: int = 0
    classification_candidates_retrieved: int = 0
    candidates_validated: int = 0
    used_models: list[ModelUse] = Field(default_factory=list)
    stage_duration_ms: dict[str, int] = Field(default_factory=dict)
    total_duration_ms: int = 0
    llm_wall_duration_ms: int = 0
    llm_queue_wait_ms: int = 0
    llm: LlmUsage = Field(default_factory=LlmUsage)


class RecommendationStatsSummary(MetadataModel):
    pipeline_version: str
    validation: Literal["llm", "semantic"]
    extraction: Literal["provided", "extracted"]
    summary: Literal["provided", "generated", "disabled", "no_outcomes"]
    score_metric: str
    candidates_retrieved: int
    classification_candidates_retrieved: int
    candidates_validated: int
    used_models: list[ModelUse]
    stage_duration_ms: dict[str, int]
    total_duration_ms: int
    llm_wall_duration_ms: int
    llm_queue_wait_ms: int


class ElmLoqMetadataResponse(MetadataModel):
    learning_outcome: list[LearningOutcomeRecommendation]
    learning_outcome_summary: Optional[LocalizedTextMap] = None
    warnings: list[RecommendationWarning]
    classification_suggestions: Optional[ClassificationSuggestions] = None
    stats: Optional[RecommendationStats | RecommendationStatsSummary] = None
