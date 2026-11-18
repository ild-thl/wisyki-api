import hashlib
import os
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .elm_loq_metadata import MetadataStage


@dataclass(frozen=True)
class MetadataLLMSettings:
    validation_batch_size: int = 24
    validation_max_input_chars: int = 24000
    concurrency: int = 2
    rate_limit_requests: int = 2
    rate_limit_window_seconds: int = 10
    synchronous_rate_wait_seconds: int = 30
    response_format: Literal["prompt", "json_object"] = "prompt"

    def __post_init__(self):
        bounds = (
            ("validation_batch_size", self.validation_batch_size, 1, 48),
            ("validation_max_input_chars", self.validation_max_input_chars, 4000, 100000),
            ("concurrency", self.concurrency, 1, 4),
            ("rate_limit_requests", self.rate_limit_requests, 1, 1000),
            ("rate_limit_window_seconds", self.rate_limit_window_seconds, 1, 3600),
            ("synchronous_rate_wait_seconds", self.synchronous_rate_wait_seconds, 1, 120),
        )
        for name, value, minimum, maximum in bounds:
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        if self.response_format not in ("prompt", "json_object"):
            raise ValueError("Unsupported metadata LLM response format.")

    @classmethod
    def from_env(cls) -> "MetadataLLMSettings":
        response_format = os.getenv("ELM_LOQ_METADATA_RESPONSE_FORMAT", "prompt")
        if response_format not in ("prompt", "json_object"):
            raise ValueError("Unsupported metadata LLM response format.")
        return cls(
            validation_batch_size=int(
                os.getenv("ELM_LOQ_METADATA_VALIDATION_BATCH_SIZE", "24")
            ),
            validation_max_input_chars=int(
                os.getenv("ELM_LOQ_METADATA_VALIDATION_MAX_INPUT_CHARS", "24000")
            ),
            concurrency=int(os.getenv("ELM_LOQ_METADATA_LLM_CONCURRENCY", "2")),
            rate_limit_requests=int(
                os.getenv("ELM_LOQ_METADATA_LLM_REQUESTS_PER_WINDOW", "2")
            ),
            rate_limit_window_seconds=int(
                os.getenv("ELM_LOQ_METADATA_LLM_RATE_WINDOW_SECONDS", "10")
            ),
            synchronous_rate_wait_seconds=int(
                os.getenv("ELM_LOQ_METADATA_SYNC_RATE_WAIT_SECONDS", "30")
            ),
            response_format=response_format,
        )


def metadata_provider_bucket(stage: MetadataStage) -> str:
    prefix = f"LLM_{stage.upper()}"
    api_base = os.getenv(f"{prefix}_API_URL") or os.getenv("LLM_API_URL", "")
    api_key = os.getenv(f"{prefix}_API_KEY") or os.getenv("LLM_API_KEY", "")
    bucket = os.getenv("ELM_LOQ_METADATA_LLM_RATE_LIMIT_BUCKET")
    material = bucket or f"{api_base.rstrip('/')}:{api_key}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def get_metadata_llm(stage: MetadataStage) -> tuple[BaseChatModel, str]:
    prefix = f"LLM_{stage.upper()}"
    default_model = (
        os.getenv("LLM_MODEL_COMPLEX") if stage == "validation" else None
    ) or os.getenv("LLM_MODEL")
    model_name = os.getenv(f"{prefix}_MODEL") or default_model
    api_base = os.getenv(f"{prefix}_API_URL") or os.getenv("LLM_API_URL")
    api_key = os.getenv(f"{prefix}_API_KEY") or os.getenv("LLM_API_KEY")
    if not model_name or not api_key or not api_base:
        raise ValueError("A metadata LLM model, API URL and API key must be configured.")
    return (
        ChatOpenAI(
            model=model_name,
            openai_api_base=api_base,
            openai_api_key=api_key,
            temperature=0.1,
            max_tokens=4096,
            max_retries=0,
            model_kwargs={"seed": 42},
        ),
        model_name,
    )
