from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, List, Tuple, Optional, Any
from .get_chat_llm import get_llm
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import os

class LearningOutcomes(BaseModel):
    learning_outcomes: List[str] = Field(
        default=[],
        description="The learning outcomes as knowledge and skills that participants will acquire by participating in the described learning opportunity.",
    )

class Prerequisites(BaseModel):
    prerequisites: List[str] = Field(
        default=[],
        description="The prerequisites as in knowledge and skills that are required to participate in the described learning opportunity. Empty if not specified or if the learning opportunity is open to total beginners.",
    )

class LearningOutcomesAndPrerequisites(LearningOutcomes, Prerequisites):
    pass

async def extract_learning_opportunity(
    doc: str, mistral_api_key: str = None, openai_api_key: str = None, targets: List[str] = None
) -> Dict[str, Any]:
    """
    Extracts structured learning opportunity data from a given document.

    Args:
        doc (str): The document from which to extract the structured learning opportunity data.
        mistral_api_key (str): The API key for the Mistral API.
        openai_api_key (str): The API key for the OpenAI API.

    Returns:
        LearningOpporunity: The extracted structured learning opportunity data.
    """
    pydantic_object = LearningOutcomesAndPrerequisites
    if "learning_outcomes" in targets and "prerequisites" in targets:
        pydantic_object = LearningOutcomesAndPrerequisites
    elif "learning_outcomes" in targets:
        pydantic_object = LearningOutcomes
    elif "prerequisites" in targets:
        pydantic_object = Prerequisites
    else:
        raise ValueError("At least one target must be specified.")
    
    # Initialize the output parser with the LearningOpportunity model.
    parser = PydanticOutputParser(pydantic_object=pydantic_object)

    # Create a prompt template for extracting structured learning opportunity data from a document.
    prompt = PromptTemplate(
        template=(
            "Folgendes Dokument ist gegeben:"
            "{doc}"
            "Analysiere das gegebene Dokument und extrahiere die relevanten und explizit enthaltenen Metadaten."
            "{format_instructions}"
            "Gebe nun die extrahierten Metadaten in deutscher Sprache und konform zum angegeben Schema aus."
            "Nicht deutsche Texte sollten ins Deutsche übersetzt werden."
        ),
        input_variables=["doc"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Init the connection to the llm model.
    model, model_name = get_llm(
        openai_api_key=openai_api_key,
        mistral_api_key=mistral_api_key,
        use_most_competent_llm=True,
        max_tokens=2048,
    )

    # Build and invoke the chain to extract the structured data.
    chain = prompt | model | parser
    result = chain.invoke({"doc": doc})
    return result.dict()
