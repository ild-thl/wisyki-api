import os
from urllib.parse import urlsplit

from langchain_core.language_models import BaseChatModel
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from .llm_transcript import MarkdownLLMTranscript


def is_mistral_api_base(api_base: str | None) -> bool:
    if not api_base:
        return False
    try:
        hostname = urlsplit(api_base).hostname
    except ValueError:
        return False
    return hostname in {"api.mistral.ai", "codestral.mistral.ai"}


def create_chat_model(
    *,
    model: str,
    api_base: str | None,
    api_key: str | None,
    seed: int | None = None,
    **kwargs,
) -> BaseChatModel:
    transcript_path = os.getenv("LLM_TRANSCRIPT_PATH", "").strip()
    if transcript_path:
        kwargs["callbacks"] = [MarkdownLLMTranscript(transcript_path)]
    if is_mistral_api_base(api_base):
        return ChatMistralAI(
            model=model,
            base_url=api_base,
            api_key=api_key,
            random_seed=seed,
            **kwargs,
        )
    return ChatOpenAI(
        model=model,
        openai_api_base=api_base,
        openai_api_key=api_key,
        seed=seed,
        **kwargs,
    )
