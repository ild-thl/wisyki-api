from collections import defaultdict
from typing import Optional

from langchain_core.messages import BaseMessage

from .elm_loq_metadata import LlmCall, LlmUsage, TokenUsage, UsageSummary


def token_count(value: object) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def response_usage(response: BaseMessage) -> TokenUsage:
    raw = response.response_metadata.get("token_usage")
    if isinstance(raw, dict) and raw:
        prompt_details = raw.get("prompt_tokens_details")
        completion_details = raw.get("completion_tokens_details")
        usage = TokenUsage(
            input_tokens=token_count(raw.get("prompt_tokens")),
            output_tokens=token_count(raw.get("completion_tokens")),
            total_tokens=token_count(raw.get("total_tokens")),
            cached_input_tokens=(
                token_count(prompt_details.get("cached_tokens"))
                if isinstance(prompt_details, dict)
                else None
            ),
            reasoning_tokens=(
                token_count(completion_details.get("reasoning_tokens"))
                if isinstance(completion_details, dict)
                else None
            ),
        )
    else:
        normalized = getattr(response, "usage_metadata", None)
        usage = TokenUsage(
            **{
                field: token_count(normalized.get(field))
                for field in ("input_tokens", "output_tokens", "total_tokens")
            }
        ) if isinstance(normalized, dict) else TokenUsage()
    if (
        usage.total_tokens is None
        and usage.input_tokens is not None
        and usage.output_tokens is not None
    ):
        usage.total_tokens = usage.input_tokens + usage.output_tokens
    return usage


def summarize_calls(calls: list[LlmCall]) -> UsageSummary:
    tokens = {}
    for field in TokenUsage.model_fields:
        counts = [getattr(call.token_usage, field) for call in calls]
        tokens[field] = (
            sum(count for count in counts if count is not None)
            if all(count is not None for count in counts)
            else None
        )
    unknown = sum(
        any(
            count is None
            for count in (
                call.token_usage.input_tokens,
                call.token_usage.output_tokens,
                call.token_usage.total_tokens,
            )
        )
        for call in calls
    )
    return UsageSummary(
        request_count=len(calls),
        response_count=sum(call.status == "received" for call in calls),
        retry_count=sum(call.attempt > 1 for call in calls),
        repair_count=sum(call.repair for call in calls),
        call_duration_ms=sum(call.duration_ms for call in calls),
        unknown_usage_requests=unknown,
        usage_complete=unknown == 0,
        token_usage=TokenUsage(**tokens),
    )


def aggregate_usage(calls: list[LlmCall]) -> LlmUsage:
    stages: dict[str, list[LlmCall]] = defaultdict(list)
    models: dict[str, list[LlmCall]] = defaultdict(list)
    for call in calls:
        stages[call.stage].append(call)
        models[call.model].append(call)
    return LlmUsage(
        calls=calls,
        totals=summarize_calls(calls),
        by_stage={stage: summarize_calls(items) for stage, items in stages.items()},
        by_model={model: summarize_calls(items) for model, items in models.items()},
    )


def llm_wall_duration(calls: list[LlmCall]) -> int:
    elapsed = 0
    last_end = 0
    for call in sorted(calls, key=lambda item: item.started_at_ms):
        end = call.started_at_ms + call.duration_ms
        elapsed += max(0, end - max(last_end, call.started_at_ms))
        last_end = max(last_end, end)
    return elapsed
