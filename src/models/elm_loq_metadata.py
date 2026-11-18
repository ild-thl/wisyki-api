from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from pydantic.alias_generators import to_camel


MetadataStage = Literal["extraction", "validation", "summary", "translation"]
Title = Annotated[
    str, StringConstraints(min_length=1, max_length=1000, pattern=r"\S")
]
Text = Annotated[
    str, StringConstraints(min_length=1, max_length=4000, pattern=r"\S")
]
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
    title: Optional[Title | LocalizedTitleMap] = Field(
        default=None,
        description="Course title, optionally provided as a language-keyed map.",
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
            "are not supplied and extraction is enabled."
        ),
    )
    learning_outcome_summary: Optional[LocalizedTextMap] = Field(
        default=None,
        description="Existing learning outcome summary keyed by BCP 47 language tag.",
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
    title: LocalizedTitleMap = Field(
        description=(
            "Outcome title text keyed by BCP 47 language tag. Include sourceLanguage "
            "when the title is written in that language."
        )
    )


class RecommendationOptions(MetadataModel):
    frameworks: list[Framework] = Field(
        default_factory=lambda: ["ESCO"], min_length=1, max_length=6
    )
    max_skills_per_framework: int = Field(default=3, ge=1, le=10, strict=True)
    extraction: Literal["auto", "disabled"] = "auto"
    validation: Literal["llm", "semantic"] = "llm"
    summary: Literal["auto", "disabled"] = "auto"
    stats: Literal["none", "summary", "full"] = Field(
        default="none",
        description=(
            "none omits statistics, summary returns aggregate metrics, and full "
            "includes per-call and token usage."
        ),
    )
    min_similarity: Optional[float] = Field(
        default=None,
        ge=-1,
        le=1,
        strict=True,
        description=(
            "Required in semantic mode. An explicit, caller-selected cosine similarity "
            "floor, not a calibrated confidence or relevance guarantee."
        ),
    )

    @model_validator(mode="after")
    def validate_options(self) -> "RecommendationOptions":
        if len(set(self.frameworks)) != len(self.frameworks):
            raise ValueError("Frameworks must be unique.")
        if self.validation == "semantic" and self.min_similarity is None:
            raise ValueError("Semantic validation requires an explicit minSimilarity.")
        return self


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
    learning_outcome_summary: Optional[LocalizedTextMap] = Field(
        default=None,
        description=(
            "Existing summary keyed by BCP 47 language tag. Preserved instead of "
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
            if self.course is None or self.course.description is None:
                raise ValueError("A course description or learningOutcome is required.")
        if len(set(self.target_languages)) != len(self.target_languages):
            raise ValueError("Target languages must be unique.")
        return self

    @property
    def output_languages(self) -> list[Language]:
        return self.target_languages or [self.source_language]

    @property
    def primary_language(self) -> Language:
        return self.source_language


class Evidence(MetadataModel):
    field: str = Field(min_length=1, max_length=150)
    quote: str = Field(min_length=1, max_length=12000, pattern=r"\S")


class ExtractedOutcome(MetadataModel):
    title: Title
    evidence: list[Evidence] = Field(min_length=1, max_length=3)


class ExtractionResult(MetadataModel):
    learning_outcome: list[ExtractedOutcome] = Field(max_length=20)


class SkillDecision(MetadataModel):
    outcome_id: str
    candidate_id: str
    decision: Literal["supported", "unsupported", "uncertain"]
    reason: Optional[str] = Field(default=None, min_length=1, max_length=240, pattern=r"\S")

    @model_validator(mode="after")
    def require_supported_reason(self) -> "SkillDecision":
        if self.decision == "supported" and self.reason is None:
            raise ValueError("A supported association requires a short evidence-based reason.")
        return self


class SupportedSkillDecision(MetadataModel):
    candidate_id: str
    reason: str = Field(min_length=1, max_length=240, pattern=r"\S")


class OutcomeSkillDecisions(MetadataModel):
    outcome_id: str
    supported: list[SupportedSkillDecision] = Field(max_length=48)
    unsupported: list[str] = Field(max_length=48)
    uncertain: list[str] = Field(max_length=48)


class ValidationResult(MetadataModel):
    outcomes: list[OutcomeSkillDecisions] = Field(max_length=20)

    @property
    def decisions(self) -> list[SkillDecision]:
        decisions = []
        for outcome in self.outcomes:
            decisions.extend(
                SkillDecision(
                    outcome_id=outcome.outcome_id,
                    candidate_id=item.candidate_id,
                    decision="supported",
                    reason=item.reason,
                )
                for item in outcome.supported
            )
            for decision, identifiers in (
                ("unsupported", outcome.unsupported), ("uncertain", outcome.uncertain)
            ):
                decisions.extend(
                    SkillDecision(
                        outcome_id=outcome.outcome_id,
                        candidate_id=identifier,
                        decision=decision,
                    )
                    for identifier in identifiers
                )
        return decisions


class SummaryResult(MetadataModel):
    text: Text


class LocalizedOutcome(MetadataModel):
    id: str
    title: LocalizedTitleMap


class LocalizationResult(MetadataModel):
    learning_outcomes: list[LocalizedOutcome] = Field(max_length=20)
    learning_outcome_summary: Optional[LocalizedTextMap] = None


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
    origin: Literal["provided", "extracted"]
    evidence: list[Evidence]
    related_esco_skill: list[RecommendedSkill] = Field(
        default_factory=list, alias="relatedESCOSkill"
    )
    related_skill: list[RecommendedSkill] = Field(default_factory=list)


class RecommendedOutcome(LearningOutcomeRecommendation):
    id: str


class RecommendedSummary(MetadataModel):
    text: LocalizedTextMap
    origin: Literal["provided", "generated"]


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
    pipeline_version: str = "elm-loq-metadata-v3"
    validation: Literal["llm", "semantic"]
    extraction: Literal["provided", "extracted"]
    summary: Literal["provided", "generated", "disabled", "no_outcomes"]
    score_metric: str = "cosine_similarity"
    candidates_retrieved: int = 0
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
    candidates_validated: int
    used_models: list[ModelUse]
    stage_duration_ms: dict[str, int]
    total_duration_ms: int
    llm_wall_duration_ms: int
    llm_queue_wait_ms: int


class ElmLoqMetadataResponse(MetadataModel):
    learning_outcome: list[LearningOutcomeRecommendation]
    learning_outcome_summary: Optional[LocalizedTextMap]
    learning_outcome_summary_origin: Optional[Literal["provided", "generated"]] = None
    warnings: list[RecommendationWarning]
    stats: Optional[RecommendationStats | RecommendationStatsSummary] = None
