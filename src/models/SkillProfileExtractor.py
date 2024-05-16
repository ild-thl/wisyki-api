from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from .get_chat_llm import get_llm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os


class EducationalAchievement(BaseModel):
    type: Optional[str] = Field(None, description="The type of educational achievement.")
    title: Optional[str] = Field(None, description="The title of the educational achievement.")
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
        description="The educational achievements and attainments of the person. This can include degrees, courses, microcredentials, etc.",
    )
    work_experience: Optional[List[WorkExperience]] = Field(
        default=[], description="The work experience of the person."
    )
    skills: Optional[List[str]] = Field(
        default=[],
        description="The skills of the person.",
    )
    desired_skills: Optional[List[str]] = Field(
        default=[],
        description="The desired skills of the person.",
    )
    languages: Optional[List[LanguageProficiency]] = Field(
        default=[],
        description="The languages the person knows.",
    )
    desired_occuptions: Optional[List[str]] = Field(
        default=[],
        description="The desired occupations of the person.",
    )
    interests: Optional[List[str]] = Field(
        default=[],
        description="The interests of the person. Besides professional interests, also personal interests can be included.",
    )


async def extract_skill_profile(doc: str, mistral_api_key: str = None, openai_api_key: str = None) -> SkillProfile:
    # Extract the skill profile from the document
    template = (
        "Folgendes Dokument ist gegeben:"
        "{doc}"

        "Analysiere das gegebene Dokument und extrahiere die relevanten und explizit enthaltenen Metadaten."
        "{format_instructions}"

        "Gebe nun die extrahierten Profildaten in deutscher Sprache und konform zum angegeben Schema aus."
        "Nicht deutsche Texte sollten ins Deutsche übersetzt werden."
    )

    parser = JsonOutputParser(pydantic_object=SkillProfile)
    model, model_name = get_llm(openai_api_key=openai_api_key, mistral_api_key=mistral_api_key, use_most_competent_llm=False, max_tokens=2048)
    prompt = PromptTemplate(
        template=template,
        input_variables=["doc"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    return chain.invoke({"doc": doc})


class Recommendation(BaseModel):
    skills: Optional[List[str]] = Field(
        default=[],
        description="The recommended skills the person needs to aquire to reach their desired goals."
    )
    occupations: Optional[List[str]] = Field(
        default=[],
        description="The recommended occupations for the person to pursue based in their talents and interests."
    )
    education: Optional[List[str]] = Field(
        default=[],
        description="The recommended educational paths for the person to take to a aquire the desired skills and prepare for the desired occupation. This can include degrees, courses, certifications, etc."
    )


async def get_recommendation(profile: SkillProfile) -> Recommendation:
    # Get a recommendation based on the skill profile
    template = (
        "Folgendes Profil ist gegeben:\n"
        "{profile}\n\n"
        "Analysiere das gegebene Profil und gebe eine weitergehende Empfehlung aus. Spreche nur eine Empfehlung aus, wenn sie das Profil bereichert.\n\n"
        "{format_instructions}\n\n"
        "Gebe nun die Empfehlung in deutscher Sprache und konform zum angegeben Schema aus.\n"
    )

    parser = JsonOutputParser(pydantic_object=Recommendation)
    model, model_name = get_llm()
    prompt = PromptTemplate(
        template=template,
        input_variables=["profile"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    return chain.invoke({"profile": profile})
