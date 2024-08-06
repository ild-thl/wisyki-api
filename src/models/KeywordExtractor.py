from .get_chat_llm import get_llm
from typing import List
from keybert.llm import LangChain
from keybert import KeyLLM
from langchain.chains.question_answering import load_qa_chain

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
            "<keywords>"
        )
        # Load it in KeyLLM.
        self.kw_model = KeyLLM(llm=LangChain(chain, prompt=prompt))
    
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
        self.add_model_stats(self.model_name, "Keyword extraction with KeyBERT & KeyLLM")

        # Extract keywords.
        keywords = self.kw_model.extract_keywords(document, candidate_keywords)

        if len(keywords) == 1:
            keywords = keywords[0]

        # if <keywords> or </keywords> in keywords, remove them
        keywords = [keyword.replace("<keywords>", "").replace("</keywords>", "") for keyword in keywords]
        
        return keywords