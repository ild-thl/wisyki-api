from .get_chat_llm import get_llm
import re
from typing import List
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document


class LLMKeywordExtractor:
    def __init__(self, chain, prompt):
        self.chain = chain
        self.prompt = prompt

    def extract_keywords(
        self, document: str, candidate_keywords: List[str]
    ) -> List[str]:
        result = self.chain.run(
            input_documents=[Document(page_content=document)],
            question=self.prompt,
        )
        return clean_keyword_response(result)


def clean_keyword_response(response: str) -> List[str]:
    """Extract only the comma-separated content inside keyword tags."""
    if not isinstance(response, str) or not response.strip():
        return []

    tagged_match = re.search(
        r"<keywords>\s*(.*?)\s*</keywords>", response, flags=re.IGNORECASE | re.DOTALL
    )
    if not tagged_match:
        return []

    content = tagged_match.group(1)
    keywords = re.split(r"[,;\n]", content)

    cleaned = []
    seen = set()
    for keyword in keywords:
        keyword = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", keyword).strip(" \t\"'")
        if not keyword:
            continue
        normalized = keyword.casefold()
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(keyword)
    return cleaned


class KeywordExtractor:
    def __init__(self, request):
        """
        Initialize the KeywordExtractor object.

        Parameters:
        - request: The request object containing the request parameters.
        """
        self.openai_api_key = request.openai_api_key
        self.mistral_api_key = request.mistral_api_key
        self.used_models = []
        llm, self.model_name = get_llm(self.openai_api_key, self.mistral_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        prompt = (
            "Folgendes Dokument liegt vor:"
            "[DOCUMENT]"
            ""
            "Mit diesen potenziellen Schlüsselwörtern:"
            "[CANDIDATES]"
            ""
            "Auf Grundlage der obigen Informationen, optimieren Sie bitte die potenziellen Schlüsselwörter, um das Thema des Dokuments bestmöglich zu repräsentieren."
            ""
            "Bitte verwenden Sie das folgende Format und trennen Sie die Schlüsselwörter durch Kommas:"
            "<keywords>keyword1, keyword2</keywords>"
        )
        self.kw_model = LLMKeywordExtractor(chain=chain, prompt=prompt)

    def add_model_stats(self, model_name: str, reason: str):
        self.used_models.append({"model": model_name, "reason": reason})

    def extract(self, document: str, candidate_keywords: List[str]) -> List[str]:
        """
        Extracts keywords from a document using KeyBERT.

        Args:
            document (str): A document from which to extract keywords.
            candidate_keywords (List[str]): A list of candidate keywords.

        Returns:
            List[str]: A list of extracted keywords.
        """
        # Create LLM.
        self.add_model_stats(
            self.model_name, "Keyword extraction with KeyBERT & KeyLLM"
        )

        # Extract keywords.
        keywords = self.kw_model.extract_keywords(document, candidate_keywords)

        return keywords
