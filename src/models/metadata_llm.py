import hashlib
import os
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from .chat_model_factory import create_chat_model
from .elm_loq_metadata import MetadataStage


@dataclass(frozen=True)
class MetadataLLMSettings:
    validation_max_input_chars: int = 24000
    concurrency: int = 2
    rate_limit_requests: int = 2
    rate_limit_window_seconds: int = 10
    synchronous_rate_wait_seconds: int = 30
    validation_timeout_seconds: float = 120.0
    validation_max_output_tokens: int = 16384

    def __post_init__(self):
        bounds = (
            (
                "validation_max_input_chars",
                self.validation_max_input_chars,
                4000,
                100000,
            ),
            ("concurrency", self.concurrency, 1, 4),
            ("rate_limit_requests", self.rate_limit_requests, 1, 1000),
            ("rate_limit_window_seconds", self.rate_limit_window_seconds, 1, 3600),
            (
                "synchronous_rate_wait_seconds",
                self.synchronous_rate_wait_seconds,
                1,
                120,
            ),
            ("validation_timeout_seconds", self.validation_timeout_seconds, 5, 3600),
            (
                "validation_max_output_tokens",
                self.validation_max_output_tokens,
                1024,
                65536,
            ),
        )
        for name, value, minimum, maximum in bounds:
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}.")

    @classmethod
    def from_env(cls) -> "MetadataLLMSettings":
        return cls(
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
            validation_timeout_seconds=float(
                os.getenv("ELM_LOQ_METADATA_VALIDATION_TIMEOUT_SECONDS", "120")
            ),
            validation_max_output_tokens=int(
                os.getenv("ELM_LOQ_METADATA_VALIDATION_MAX_OUTPUT_TOKENS", "16384")
            ),
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
        raise ValueError(
            "A metadata LLM model, API URL and API key must be configured."
        )
    max_tokens = (
        MetadataLLMSettings.from_env().validation_max_output_tokens
        if stage == "validation"
        else 4096
    )
    return (
        create_chat_model(
            model=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=0.1,
            max_tokens=max_tokens,
            max_retries=0,
            seed=42,
        ),
        model_name,
    )
