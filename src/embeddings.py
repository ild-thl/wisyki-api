"""
Embeddings module for HuggingFace Inference API integration.

This module provides a unified embedding wrapper that's used throughout the application
for vectorizing documents and queries consistently.
"""

import os
import requests
from typing import Optional
from langchain_core.embeddings import Embeddings


class HuggingFaceInferenceEmbeddings(Embeddings):
    """Wrapper for HuggingFace Inference API embeddings"""

    def __init__(self, api_key: str, model: str):
        """
        Initialize the embeddings.

        Args:
            api_key: HuggingFace API key
            model: The inference endpoint URL or model name
        """
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            self.model,
            json={"inputs": texts},
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()

        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                return result
            elif isinstance(result[0], dict) and "embedding" in result[0]:
                return [item["embedding"] for item in result]
        return result

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]


def load_embedding_function(
    api_key: Optional[str] = None, api_host: Optional[str] = None
) -> HuggingFaceInferenceEmbeddings:
    """
    Load HuggingFace embedding function from external API.

    Args:
        api_key: HuggingFace API key (default: from HF_EMBEDDING_API_KEY env var)
        api_host: Inference endpoint URL (default: from HF_EMBEDDING_HOST env var)

    Returns:
        HuggingFaceInferenceEmbeddings instance

    Raises:
        ValueError: If API key is not provided or available
    """
    if api_host is None:
        api_host = os.getenv("HF_EMBEDDING_HOST", "https://hf-text-inf.dev.dlc.sh")

    if api_key is None:
        api_key = os.getenv("HF_EMBEDDING_API_KEY")

    if not api_key:
        raise ValueError("HF_EMBEDDING_API_KEY environment variable is not set")

    return HuggingFaceInferenceEmbeddings(api_key=api_key, model=api_host)
