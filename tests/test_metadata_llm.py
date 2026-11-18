import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage

from src.models.elm_loq_metadata import ElmLoqMetadataRequest, LlmCall, TokenUsage
from src.models.ElmLoqMetadataRecommender import (
    ElmLoqMetadataRecommender,
    RecommendationError,
)
from src.models.get_chat_llm import get_default_llm
from src.models.llm_usage import aggregate_usage, llm_wall_duration, response_usage
from src.models.metadata_llm import MetadataLLMSettings, get_metadata_llm
from test_elm_loq_metadata import (
    outcome, query_result, record, request_data, validation_response,
)


def response(value, input_tokens=100, output_tokens=20, **details):
    return AIMessage(
        content=json.dumps(value),
        response_metadata={
            "model_name": "resolved-model",
            "token_usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                **details,
            },
        },
    )


def decisions(messages):
    pairs = json.loads(messages[1].content)["data"]["pairs"]
    return validation_response(
        [
            {
                "outcomeId": pair["outcomeId"],
                "candidateId": pair["candidateId"],
                "decision": "supported",
                "reason": "Supported by this outcome.",
            }
            for pair in pairs
        ]
    )


class TokenAccountingTests(unittest.TestCase):
    def test_reads_raw_token_cache_and_reasoning_details_without_double_counting(self):
        usage = response_usage(response(
            {},
            prompt_tokens_details={"cached_tokens": 40},
            completion_tokens_details={"reasoning_tokens": 5},
        ))
        self.assertEqual(
            usage,
            TokenUsage(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                cached_input_tokens=40,
                reasoning_tokens=5,
            ),
        )

    def test_normalized_usage_and_missing_or_invalid_counts(self):
        normalized = AIMessage(
            content="{}",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        self.assertEqual(response_usage(normalized).total_tokens, 15)
        self.assertIsNone(response_usage(AIMessage(content="{}")).input_tokens)
        partial = AIMessage(
            content="{}",
            response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        )
        self.assertEqual(response_usage(partial).total_tokens, 15)
        invalid = AIMessage(
            content="{}",
            response_metadata={
                "token_usage": {"prompt_tokens": True, "completion_tokens": -1}
            },
        )
        self.assertIsNone(response_usage(invalid).input_tokens)
        self.assertIsNone(response_usage(invalid).total_tokens)

    def test_aggregation_marks_unknown_attempt_usage_instead_of_reporting_zero(self):
        calls = [
            LlmCall(
                id=1, stage="validation", requested_model="alias", model="resolved",
                attempt=1, repair=False, started_at_ms=0, duration_ms=20,
                prompt_characters=100, status="received",
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            LlmCall(
                id=2, stage="validation", requested_model="alias", model="alias",
                attempt=2, repair=True, started_at_ms=10, duration_ms=20,
                prompt_characters=100, status="error",
            ),
        ]
        usage = aggregate_usage(calls)
        self.assertEqual(usage.totals.request_count, 2)
        self.assertEqual(usage.totals.response_count, 1)
        self.assertEqual(usage.totals.retry_count, 1)
        self.assertEqual(usage.totals.repair_count, 1)
        self.assertEqual(usage.totals.unknown_usage_requests, 1)
        self.assertFalse(usage.totals.usage_complete)
        self.assertIsNone(usage.totals.token_usage.input_tokens)
        self.assertEqual(usage.by_model["resolved"].token_usage.total_tokens, 15)
        self.assertEqual(usage.by_stage["validation"].request_count, 2)
        self.assertEqual(llm_wall_duration(calls), 30)
        self.assertEqual(usage.totals.call_duration_ms, 40)
        self.assertEqual(aggregate_usage([]).totals.token_usage.total_tokens, 0)


class ModelConfigurationTests(unittest.TestCase):
    def test_stage_overrides_and_complex_model_do_not_require_new_default_settings(self):
        values = {
            "LLM_MODEL": "small",
            "LLM_MODEL_COMPLEX": "large",
            "LLM_API_URL": "http://llm.invalid",
            "LLM_API_KEY": "test-key",
            "LLM_SUMMARY_MODEL": "tiny",
            "LLM_SUMMARY_API_URL": "http://summary.invalid",
            "LLM_SUMMARY_API_KEY": "summary-test-key",
        }
        with patch.dict("os.environ", values, clear=True), patch(
            "src.models.metadata_llm.ChatOpenAI"
        ) as constructor:
            for stage, expected in (
                ("extraction", "small"), ("validation", "large"), ("summary", "tiny")
            ):
                _, name = get_metadata_llm(stage)
                self.assertEqual(name, expected)
                self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
            self.assertEqual(
                constructor.call_args.kwargs["openai_api_base"], "http://summary.invalid"
            )
            self.assertEqual(
                constructor.call_args.kwargs["openai_api_key"], "summary-test-key"
            )
        values["LLM_VALIDATION_MODEL"] = "specialist"
        with patch.dict("os.environ", values, clear=True), patch(
            "src.models.metadata_llm.ChatOpenAI"
        ):
            self.assertEqual(get_metadata_llm("validation")[1], "specialist")

    def test_legacy_default_uses_complex_model_only_when_requested_and_configured(self):
        with patch.dict(
            "os.environ", {"LLM_MODEL": "small", "LLM_MODEL_COMPLEX": "large"}, clear=True
        ), patch("src.models.get_chat_llm.ChatOpenAI"):
            self.assertEqual(get_default_llm()[1], "small")
            self.assertEqual(get_default_llm(use_most_competent_llm=True)[1], "large")
        with patch.dict("os.environ", {"LLM_MODEL": "small"}, clear=True), patch(
            "src.models.get_chat_llm.ChatOpenAI"
        ):
            self.assertEqual(get_default_llm(use_most_competent_llm=True)[1], "small")

    def test_invalid_budget_configuration_is_rejected(self):
        for values in (
            {"ELM_LOQ_METADATA_LLM_CONCURRENCY": "0"},
            {"ELM_LOQ_METADATA_VALIDATION_BATCH_SIZE": "49"},
            {"ELM_LOQ_METADATA_VALIDATION_MAX_INPUT_CHARS": "10"},
            {"ELM_LOQ_METADATA_RESPONSE_FORMAT": "unsupported"},
        ):
            with self.subTest(values=values), patch.dict("os.environ", values, clear=True):
                with self.assertRaises(ValueError):
                    MetadataLLMSettings.from_env()


class PipelineAccountingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.embedding = Mock()
        self.embedding.embed_documents.side_effect = lambda texts: [
            [1.0, float(index)] for index, _ in enumerate(texts)
        ]
        self.collection = Mock()
        self.collection.metadata = {"hnsw:space": "cosine"}
        self.collection.query.return_value = query_result([[record()]])
        self.model = Mock()
        self.model.ainvoke = AsyncMock()
        self.factory = Mock(return_value=(self.model, "requested-model"))

    def service(self, **settings):
        return ElmLoqMetadataRecommender(
            self.embedding, self.collection, self.factory,
            settings=MetadataLLMSettings(**settings),
        )

    def request(self, **changes):
        return ElmLoqMetadataRequest.model_validate(request_data(
            options={
                "validation": "llm",
                "extraction": "disabled",
                "summary": "disabled",
                "stats": "full",
            },
            **changes,
        ))

    async def test_every_repair_and_retry_is_counted_with_returned_model_and_usage(self):
        self.model.ainvoke.side_effect = [
            response({"outcomes": []}, 100, 10),
            asyncio.TimeoutError(),
            response(validation_response([{
                "outcomeId": "lo-1", "candidateId": "c1",
                "decision": "supported", "reason": "Direct evidence.",
            }]), 110, 20),
        ]
        with patch("src.models.ElmLoqMetadataRecommender.asyncio.sleep", new=AsyncMock()):
            result = await self.service().recommend(self.request())
        stats = result.stats
        self.assertEqual(stats.llm.totals.request_count, 3)
        self.assertEqual(stats.llm.totals.repair_count, 2)
        self.assertEqual(stats.llm.totals.retry_count, 1)
        self.assertEqual(stats.llm.by_model["resolved-model"].token_usage.total_tokens, 240)
        self.assertIsNone(stats.llm.totals.token_usage.total_tokens)
        self.assertEqual(stats.llm.calls[0].requested_model, "requested-model")
        self.assertEqual(stats.llm.calls[0].model, "resolved-model")
        self.assertEqual(stats.llm.calls[1].status, "error")
        self.assertIn("LLM_USAGE_INCOMPLETE", [item.code for item in result.warnings])

    async def test_complete_usage_is_aggregated_and_missing_details_remain_unknown(self):
        self.model.ainvoke.side_effect = lambda messages: response(decisions(messages), 80, 20)
        result = await self.service().recommend(self.request())
        totals = result.stats.llm.totals
        self.assertEqual(totals.token_usage.total_tokens, 100)
        self.assertTrue(totals.usage_complete)
        self.assertIsNone(totals.token_usage.cached_input_tokens)
        self.assertEqual(len(result.stats.used_models), 1)

    async def test_prompt_tables_deduplicate_shared_outcomes_and_skill_definitions(self):
        self.collection.query.return_value = query_result([[record()], [record()]])
        seen = []

        async def validate(messages):
            payload = json.loads(messages[1].content)["data"]
            seen.append(payload)
            return response(decisions(messages))

        self.model.ainvoke.side_effect = validate
        result = await self.service().recommend(self.request(
            learningOutcome=[outcome(), outcome("Use rigs.")]
        ))
        self.assertEqual(len(seen[0]["outcomes"]), 2)
        self.assertEqual(len(seen[0]["skills"]), 1)
        self.assertEqual(len(seen[0]["pairs"]), 2)
        self.assertNotIn("evidence", seen[0]["pairs"][0])
        self.assertEqual(result.stats.llm.totals.request_count, 1)
        self.assertEqual(result.stats.candidates_validated, 2)

    async def test_validation_parallelism_is_bounded_and_batch_ids_remain_isolated(self):
        self.collection.query.return_value = query_result([[record()]] * 4)
        active = 0
        maximum = 0
        two_started = asyncio.Event()

        async def validate(messages):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            if active == 2:
                two_started.set()
            await asyncio.wait_for(two_started.wait(), timeout=1)
            try:
                return response(decisions(messages))
            finally:
                active -= 1

        self.model.ainvoke.side_effect = validate
        result = await self.service(validation_batch_size=1, concurrency=2).recommend(
            self.request(learningOutcome=[
                outcome(f"Outcome {index}") for index in range(4)
            ])
        )
        self.assertEqual(maximum, 2)
        self.assertEqual(result.stats.llm.totals.request_count, 4)
        self.assertTrue(all(item.related_esco_skill for item in result.learning_outcome))

    async def test_failed_stage_cancels_and_accounts_for_inflight_sibling_calls(self):
        self.collection.query.return_value = query_result([[record()], [record()]])
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def validate(messages):
            pairs = json.loads(messages[1].content)["data"]["pairs"]
            if pairs[0]["outcomeId"] == "lo-1":
                await asyncio.wait_for(sibling_started.wait(), 1)
                return response({"outcomes": []})
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        self.model.ainvoke.side_effect = validate
        with self.assertRaises(RecommendationError) as caught:
            await self.service(validation_batch_size=1).recommend(self.request(
                learningOutcome=[
                    outcome("First outcome."), outcome("Use rigs.")
                ]
            ))
        self.assertTrue(sibling_cancelled.is_set())
        calls = caught.exception.stats.llm.calls
        self.assertEqual(len(calls), 3)
        self.assertIn("cancelled", [call.status for call in calls])
        self.assertFalse(caught.exception.stats.llm.totals.usage_complete)

    async def test_stage_models_have_separate_accounting_and_summary_overlaps_validation(self):
        summary_started = asyncio.Event()
        models = {}
        names = {"extraction": "small", "validation": "smart", "summary": "tiny"}
        extraction = {
            "learningOutcome": [{
                "title": "Animate with keyframes.",
                "evidence": [{"field": "course.description", "quote": "Use keyframes."}],
            }]
        }

        async def validate(messages):
            await asyncio.wait_for(summary_started.wait(), 1)
            return decisions(messages)

        for stage, name in names.items():
            model = Mock()

            async def invoke(messages, stage=stage, name=name):
                if stage == "extraction":
                    value = extraction
                elif stage == "summary":
                    summary_started.set()
                    value = {"text": "Animate with keyframes."}
                else:
                    value = await validate(messages)
                result = response(value)
                result.response_metadata["model_name"] = name
                return result

            model.ainvoke = AsyncMock(side_effect=invoke)
            models[stage] = model
        self.factory.side_effect = lambda stage: (models[stage], names[stage])
        result = await self.service().recommend(ElmLoqMetadataRequest.model_validate({
            "course": {"description": "Use keyframes."},
            "sourceLanguage": "en",
            "options": {"stats": "full"},
        }))
        self.assertEqual(set(result.stats.llm.by_model), {"small", "smart", "tiny"})
        self.assertEqual(result.stats.llm.totals.request_count, 3)
        self.assertEqual(result.stats.llm.totals.token_usage.total_tokens, 360)
        self.assertTrue(all(
            item.request_count == 1 for item in result.stats.llm.by_stage.values()
        ))

    async def test_context_budget_rejects_an_oversized_pair_without_truncating_source(self):
        with self.assertRaises(RecommendationError) as caught:
            await self.service(validation_max_input_chars=4000).recommend(self.request(
                course={"description": "x" * 5000}
            ))
        self.assertEqual(caught.exception.status_code, 422)
        self.factory.assert_not_called()

    async def test_optional_json_mode_is_explicit_and_retains_raw_usage(self):
        self.model.bind.return_value = self.model
        self.model.ainvoke.side_effect = lambda messages: response(decisions(messages))
        result = await self.service(response_format="json_object").recommend(self.request())
        self.model.bind.assert_called_once_with(response_format={"type": "json_object"})
        self.assertEqual(result.stats.llm.totals.token_usage.total_tokens, 120)
