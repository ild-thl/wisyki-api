"""
Embeddings module for HuggingFace Inference API integration.

This module provides a unified embedding wrapper that's used throughout the application
for vectorizing documents and queries consistently.

Prefix handling
---------------
Some embedding models (notably the e5 family) are trained with instruction prefixes:
queries must be prefixed with ``query: `` and documents with ``passage: `` for
retrieval to work correctly. These prefixes are a property of the *model*, not of
individual callers, so they are configured once alongside the model and applied at
the root embedding calls only. No other part of the codebase should append prefixes
itself; doing so would double-prefix and corrupt the vectors.

The prefixes are read from environment variables so that a different model (or a
model without prefix requirements) can be configured without code changes:

- ``HF_EMBEDDING_QUERY_PREFIX`` (default: ``query: ``)
- ``HF_EMBEDDING_PASSAGE_PREFIX`` (default: ``passage: ``)

Set either to an empty string to disable prefixing for that call type.
"""

import os
import requests
from typing import Optional
from langchain_core.embeddings import Embeddings


class HuggingFaceInferenceEmbeddings(Embeddings):
    """Wrapper for HuggingFace Inference API embeddings"""

    def __init__(
        self,
        api_key: str,
        model: str,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        """
        Initialize the embeddings.

        Args:
            api_key: HuggingFace API key
            model: The inference endpoint URL or model name
            query_prefix: Prefix applied to every query before embedding.
            passage_prefix: Prefix applied to every document before embedding.
        """
        self.api_key = api_key
        self.model = model
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def _post(self, texts: list[str]) -> list[list[float]]:
        """POST texts to the inference endpoint and normalize the response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            self.model,
            json={"inputs": texts},
            headers=headers,
            timeout=(5, 30),
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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs (passage prefix applied at the root call)."""
        if self.passage_prefix:
            texts = [f"{self.passage_prefix}{text}" for text in texts]
        return self._post(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed query text (query prefix applied at the root call)."""
        if self.query_prefix:
            text = f"{self.query_prefix}{text}"
        return self._post([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of query texts (query prefix applied at the root call)."""
        if self.query_prefix:
            texts = [f"{self.query_prefix}{text}" for text in texts]
        return self._post(texts)


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

    return HuggingFaceInferenceEmbeddings(
        api_key=api_key,
        model=api_host,
        query_prefix=os.getenv("HF_EMBEDDING_QUERY_PREFIX", "query: "),
        passage_prefix=os.getenv("HF_EMBEDDING_PASSAGE_PREFIX", "passage: "),
    )
