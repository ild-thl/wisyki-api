import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.models.elm_loq_metadata import (
    ElmLoqMetadataRequest,
    LlmCall,
    TokenUsage,
    ValidationResult,
)
from src.models.ElmLoqMetadataRecommender import (
    ElmLoqMetadataRecommender,
    RecommendationError,
    RecommendationTimeout,
    RecommendationUnavailable,
    gather_or_cancel,
)
from src.models.get_chat_llm import get_default_llm
from src.models.chat_model_factory import create_chat_model
from src.models.llm_transcript import MarkdownLLMTranscript
from src.models.llm_usage import aggregate_usage, llm_wall_duration, response_usage
from src.models.metadata_llm import MetadataLLMSettings, get_metadata_llm
from test_elm_loq_metadata import (
    outcome,
    query_result,
    record,
    request_data,
    validation_response,
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
    outcomes = json.loads(messages[1].content)["data"]["outcomes"]
    candidates = [
        candidate for outcome in outcomes for candidate in outcome["skillCandidates"]
    ]
    return validation_response(
        [
            {
                "pairId": candidate["pairId"],
                "decision": "supported",
                "reason": "Supported by this outcome.",
            }
            for candidate in candidates
        ]
    )


class TokenAccountingTests(unittest.TestCase):
    def test_reads_raw_token_cache_and_reasoning_details_without_double_counting(self):
        usage = response_usage(
            response(
                {},
                prompt_tokens_details={"cached_tokens": 40},
                completion_tokens_details={"reasoning_tokens": 5},
            )
        )
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
            response_metadata={
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}
            },
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
                id=1,
                stage="validation",
                requested_model="alias",
                model="resolved",
                attempt=1,
                repair=False,
                started_at_ms=0,
                duration_ms=20,
                prompt_characters=100,
                status="received",
                token_usage=TokenUsage(
                    input_tokens=10, output_tokens=5, total_tokens=15
                ),
            ),
            LlmCall(
                id=2,
                stage="validation",
                requested_model="alias",
                model="alias",
                attempt=2,
                repair=True,
                started_at_ms=10,
                duration_ms=20,
                prompt_characters=100,
                status="error",
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


class TranscriptTests(unittest.TestCase):
    def test_writes_readable_prompts_and_structured_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "debug" / "llm.md"
            transcript = MarkdownLLMTranscript(str(path))
            transcript.on_chat_model_start(
                {"kwargs": {"model": "mistral-small-latest"}},
                [
                    [
                        SystemMessage(content="Select supported classifications."),
                        HumanMessage(
                            content=json.dumps(
                                {
                                    "language": "en",
                                    "data": {"course": {"title": "UX design"}},
                                }
                            )
                        ),
                    ]
                ],
                run_id="request-1",
            )
            transcript.on_llm_end(
                LLMResult(
                    generations=[
                        [
                            ChatGeneration(
                                message=AIMessage(
                                    content="",
                                    tool_calls=[
                                        {
                                            "name": "ClassificationResult",
                                            "args": {"iscedFCode": "0211"},
                                            "id": "call-1",
                                            "type": "tool_call",
                                        }
                                    ],
                                )
                            ),
                        ]
                    ]
                ),
                run_id="request-1",
            )

            content = path.read_text(encoding="utf-8")

        self.assertIn("### System", content)
        self.assertIn("- **title:** UX design", content)
        self.assertIn("### Structured response: ClassificationResult", content)
        self.assertIn("- **iscedFCode:** 0211", content)
        self.assertNotIn('{"language"', content)

    def test_provider_factory_attaches_transcript_only_when_configured(self):
        with patch.dict(os.environ, {"LLM_TRANSCRIPT_PATH": "/tmp/llm.md"}), patch(
            "src.models.chat_model_factory.ChatOpenAI"
        ) as constructor:
            create_chat_model(
                model="test-model",
                api_base="https://llm.example/v1",
                api_key="test-key",
            )

        callbacks = constructor.call_args.kwargs["callbacks"]
        self.assertEqual(len(callbacks), 1)
        self.assertIsInstance(callbacks[0], MarkdownLLMTranscript)


class ModelConfigurationTests(unittest.TestCase):
    def test_chat_model_factory_selects_mistral_and_maps_seed(self):
        with patch.dict(os.environ, {"LLM_TRANSCRIPT_PATH": ""}), patch(
            "src.models.chat_model_factory.ChatMistralAI"
        ) as mistral, patch("src.models.chat_model_factory.ChatOpenAI") as openai:
            create_chat_model(
                model="mistral-small-latest",
                api_base="https://api.mistral.ai/v1",
                api_key="test-key",
                seed=42,
                max_tokens=1024,
            )

        mistral.assert_called_once_with(
            model="mistral-small-latest",
            base_url="https://api.mistral.ai/v1",
            api_key="test-key",
            random_seed=42,
            max_tokens=1024,
        )
        openai.assert_not_called()

    def test_chat_model_factory_uses_openai_for_other_api_bases(self):
        with patch.dict(os.environ, {"LLM_TRANSCRIPT_PATH": ""}), patch(
            "src.models.chat_model_factory.ChatMistralAI"
        ) as mistral, patch("src.models.chat_model_factory.ChatOpenAI") as openai:
            create_chat_model(
                model="model-alias",
                api_base="https://llm.example/v1",
                api_key="test-key",
                seed=42,
            )

        openai.assert_called_once_with(
            model="model-alias",
            openai_api_base="https://llm.example/v1",
            openai_api_key="test-key",
            seed=42,
        )
        mistral.assert_not_called()

    def test_stage_overrides_and_complex_model_do_not_require_new_default_settings(
        self,
    ):
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
            "src.models.metadata_llm.create_chat_model"
        ) as constructor:
            for stage, expected in (
                ("extraction", "small"),
                ("validation", "large"),
                ("summary", "tiny"),
            ):
                _, name = get_metadata_llm(stage)
                self.assertEqual(name, expected)
                self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
                self.assertEqual(constructor.call_args.kwargs["seed"], 42)
            self.assertEqual(
                constructor.call_args.kwargs["api_base"], "http://summary.invalid"
            )
            self.assertEqual(
                constructor.call_args.kwargs["api_key"], "summary-test-key"
            )
        values["LLM_VALIDATION_MODEL"] = "specialist"
        with patch.dict("os.environ", values, clear=True), patch(
            "src.models.metadata_llm.create_chat_model"
        ):
            self.assertEqual(get_metadata_llm("validation")[1], "specialist")

    def test_validation_stage_gets_a_larger_output_budget_than_other_stages(self):
        values = {
            "LLM_MODEL": "small",
            "LLM_MODEL_COMPLEX": "large",
            "LLM_API_URL": "http://llm.invalid",
            "LLM_API_KEY": "test-key",
        }
        with patch.dict("os.environ", values, clear=True), patch(
            "src.models.metadata_llm.create_chat_model"
        ) as constructor:
            get_metadata_llm("extraction")
            self.assertEqual(constructor.call_args.kwargs["max_tokens"], 4096)
            get_metadata_llm("validation")
            self.assertEqual(constructor.call_args.kwargs["max_tokens"], 16384)
        values["ELM_LOQ_METADATA_VALIDATION_MAX_OUTPUT_TOKENS"] = "8192"
        with patch.dict("os.environ", values, clear=True), patch(
            "src.models.metadata_llm.create_chat_model"
        ) as constructor:
            get_metadata_llm("validation")
            self.assertEqual(constructor.call_args.kwargs["max_tokens"], 8192)

    def test_legacy_default_uses_complex_model_only_when_requested_and_configured(self):
        with patch.dict(
            "os.environ",
            {"LLM_MODEL": "small", "LLM_MODEL_COMPLEX": "large"},
            clear=True,
        ), patch("src.models.get_chat_llm.create_chat_model"):
            self.assertEqual(get_default_llm()[1], "small")
            self.assertEqual(get_default_llm(use_most_competent_llm=True)[1], "large")
        with patch.dict("os.environ", {"LLM_MODEL": "small"}, clear=True), patch(
            "src.models.get_chat_llm.create_chat_model"
        ):
            self.assertEqual(get_default_llm(use_most_competent_llm=True)[1], "small")

    def test_invalid_budget_configuration_is_rejected(self):
        for values in (
            {"ELM_LOQ_METADATA_LLM_CONCURRENCY": "0"},
            {"ELM_LOQ_METADATA_VALIDATION_MAX_INPUT_CHARS": "10"},
            {"ELM_LOQ_METADATA_VALIDATION_TIMEOUT_SECONDS": "1"},
            {"ELM_LOQ_METADATA_VALIDATION_TIMEOUT_SECONDS": "4000"},
            {"ELM_LOQ_METADATA_VALIDATION_MAX_OUTPUT_TOKENS": "512"},
            {"ELM_LOQ_METADATA_VALIDATION_MAX_OUTPUT_TOKENS": "70000"},
        ):
            with self.subTest(values=values), patch.dict(
                "os.environ", values, clear=True
            ):
                with self.assertRaises(ValueError):
                    MetadataLLMSettings.from_env()

    def test_response_format_environment_variable_is_ignored(self):
        with patch.dict(
            "os.environ",
            {"ELM_LOQ_METADATA_RESPONSE_FORMAT": "json_schema"},
            clear=True,
        ):
            self.assertFalse(hasattr(MetadataLLMSettings.from_env(), "response_format"))


class PipelineAccountingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.embedding = Mock()
        self.embedding.embed_documents.side_effect = lambda texts: [
            [1.0, float(index)] for index, _ in enumerate(texts)
        ]
        self.embedding.embed_queries.side_effect = (
            self.embedding.embed_documents.side_effect
        )
        self.collection = Mock()
        self.collection.metadata = {"hnsw:space": "cosine"}
        self.collection.query.return_value = query_result([[record()]])
        self.model = Mock()
        self.model.ainvoke = AsyncMock()
        self.factory = Mock(return_value=(self.model, "requested-model"))
        self.create_agent = patch(
            "src.models.ElmLoqMetadataRecommender.create_agent",
            side_effect=self.build_agent,
        )
        self.create_agent_mock = self.create_agent.start()
        self.addCleanup(self.create_agent.stop)

    def build_agent(self, model, response_format):
        async def ainvoke(state):
            received = await model.ainvoke(state["messages"])
            return {
                "messages": [*state["messages"], received],
                "structured_response": json.loads(received.content),
            }

        return SimpleNamespace(ainvoke=ainvoke)

    def service(self, **settings):
        return ElmLoqMetadataRecommender(
            self.embedding,
            self.collection,
            self.factory,
            settings=MetadataLLMSettings(**settings),
        )

    def request(self, **changes):
        options = changes.pop("options", {})
        return ElmLoqMetadataRequest.model_validate(
            request_data(
                options={
                    "validation": "llm",
                    "extraction": "disabled",
                    "summary": "disabled",
                    "stats": "full",
                    **options,
                },
                **changes,
            )
        )

    async def test_every_repair_and_retry_is_counted_with_returned_model_and_usage(
        self,
    ):
        self.model.ainvoke.side_effect = [
            response({"outcomes": []}, 100, 10),
            asyncio.TimeoutError(),
            response(
                validation_response(
                    [
                        {
                            "outcomeId": "lo-1",
                            "candidateId": "c1",
                            "decision": "supported",
                            "reason": "Direct evidence.",
                        }
                    ]
                ),
                110,
                20,
            ),
        ]
        with patch(
            "src.models.ElmLoqMetadataRecommender.asyncio.sleep", new=AsyncMock()
        ):
            result = await self.service().recommend(self.request())
        stats = result.stats
        self.assertEqual(stats.llm.totals.request_count, 3)
        self.assertEqual(stats.llm.totals.repair_count, 2)
        self.assertEqual(stats.llm.totals.retry_count, 1)
        self.assertEqual(
            stats.llm.by_model["resolved-model"].token_usage.total_tokens, 240
        )
        self.assertIsNone(stats.llm.totals.token_usage.total_tokens)
        self.assertEqual(stats.llm.calls[0].requested_model, "requested-model")
        self.assertEqual(stats.llm.calls[0].model, "resolved-model")
        self.assertEqual(stats.llm.calls[1].status, "error")
        self.assertIn("LLM_USAGE_INCOMPLETE", [item.code for item in result.warnings])

    async def test_complete_usage_is_aggregated_and_missing_details_remain_unknown(
        self,
    ):
        self.model.ainvoke.side_effect = lambda messages: response(
            decisions(messages), 80, 20
        )
        result = await self.service().recommend(self.request())
        totals = result.stats.llm.totals
        self.assertEqual(totals.token_usage.total_tokens, 100)
        self.assertTrue(totals.usage_complete)
        self.assertIsNone(totals.token_usage.cached_input_tokens)
        self.assertEqual(len(result.stats.used_models), 1)

    async def test_prompt_tables_deduplicate_shared_outcomes_and_skill_definitions(
        self,
    ):
        self.collection.query.return_value = query_result([[record()], [record()]])
        seen = []

        async def validate(messages):
            payload = json.loads(messages[1].content)["data"]
            seen.append(payload)
            return response(decisions(messages))

        self.model.ainvoke.side_effect = validate
        result = await self.service().recommend(
            self.request(learningOutcome=[outcome(), outcome("Use rigs.")])
        )
        self.assertEqual(len(seen[0]["outcomes"]), 2)
        self.assertEqual(
            [len(item["skillCandidates"]) for item in seen[0]["outcomes"]], [1, 1]
        )
        candidate = seen[0]["outcomes"][0]["skillCandidates"][0]
        self.assertEqual(set(candidate), {"pairId", "title", "definition"})
        self.assertNotIn("evidence", candidate)
        self.assertEqual(result.stats.llm.totals.request_count, 1)
        self.assertEqual(result.stats.candidates_validated, 2)

    async def test_validation_uses_one_request_for_all_outcome_pairs(self):
        self.collection.query.return_value = query_result([[record()]] * 4)
        payloads = []

        async def validate(messages):
            payloads.append(json.loads(messages[1].content)["data"])
            return response(decisions(messages))

        self.model.ainvoke.side_effect = validate
        result = await self.service(concurrency=2).recommend(
            self.request(
                learningOutcome=[outcome(f"Outcome {index}") for index in range(4)],
                options={
                    "validation": "llm",
                    "extraction": "disabled",
                    "summary": "disabled",
                    "stats": "full",
                },
            )
        )
        self.assertEqual(len(payloads), 1)
        self.assertEqual(
            sum(len(item["skillCandidates"]) for item in payloads[0]["outcomes"]), 4
        )
        self.assertEqual(result.stats.llm.totals.request_count, 1)
        self.assertTrue(
            all(item.related_esco_skill for item in result.learning_outcome)
        )

    async def test_failed_parallel_stage_cancels_inflight_sibling(self):
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def fail():
            await sibling_started.wait()
            raise RecommendationError("Parallel stage failed.")

        async def sibling():
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        with self.assertRaises(RecommendationError):
            await gather_or_cancel(fail(), sibling())
        self.assertTrue(sibling_cancelled.is_set())

    async def test_stage_models_have_separate_accounting_and_extraction_overlaps_validation(
        self,
    ):
        validation_started = asyncio.Event()
        models = {}
        names = {"extraction": "small", "validation": "smart"}
        extraction = {
            "learningOutcome": [{"title": "Animate with keyframes."}],
            "learningOutcomeSummary": "Animate with keyframes.",
        }

        async def validate(messages):
            await asyncio.wait_for(validation_started.wait(), 1)
            return decisions(messages)

        for stage, name in names.items():
            model = Mock()

            async def invoke(messages, stage=stage, name=name):
                if stage == "extraction":
                    validation_started.set()
                    value = extraction
                else:
                    value = await validate(messages)
                result = response(value)
                result.response_metadata["model_name"] = name
                return result

            model.ainvoke = AsyncMock(side_effect=invoke)
            models[stage] = model
        self.factory.side_effect = lambda stage: (models[stage], names[stage])
        result = await self.service().recommend(
            ElmLoqMetadataRequest.model_validate(
                {
                    "course": {"description": "Use keyframes."},
                    "sourceLanguage": "en",
                    "options": {"stats": "full"},
                }
            )
        )
        self.assertEqual(set(result.stats.llm.by_model), {"small", "smart"})
        self.assertEqual(result.stats.llm.totals.request_count, 2)
        self.assertEqual(result.stats.llm.totals.token_usage.total_tokens, 240)
        self.assertEqual(
            result.learning_outcome_summary["en"], "Animate with keyframes."
        )
        self.assertTrue(
            all(item.request_count == 1 for item in result.stats.llm.by_stage.values())
        )

    async def test_agent_receives_schema_and_repairs_unknown_candidate_ids(self):
        invalid = validation_response(
            [
                {
                    "outcomeId": "lo-1",
                    "candidateId": "unknown",
                    "decision": "supported",
                    "reason": "Wrong ID.",
                }
            ]
        )
        valid = validation_response(
            [
                {
                    "outcomeId": "lo-1",
                    "candidateId": "c1",
                    "decision": "supported",
                    "reason": "Direct evidence.",
                }
            ]
        )
        self.model.ainvoke.side_effect = [
            response(invalid, 80, 20),
            response(valid, 90, 30),
        ]

        result = await self.service().recommend(self.request())

        self.assertEqual(self.create_agent_mock.call_count, 2)
        self.assertIs(
            self.create_agent_mock.call_args.kwargs["response_format"], ValidationResult
        )
        self.assertEqual(result.stats.llm.totals.request_count, 2)
        self.assertEqual(result.stats.llm.totals.repair_count, 1)
        self.assertEqual(result.stats.llm.totals.token_usage.total_tokens, 220)
        self.assertEqual(len(result.learning_outcome[0].related_esco_skill), 1)

    async def test_validation_repairs_skill_table_id_to_valid_pair_id(self):
        self.model.ainvoke.side_effect = [
            response(
                {
                    "supported": [
                        {
                            "pairId": "skill-1",
                            "reason": "Wrong table ID.",
                        }
                    ]
                }
            ),
            response(
                {
                    "supported": [
                        {
                            "pairId": "p-c1",
                            "reason": "Direct evidence.",
                        }
                    ]
                }
            ),
        ]

        result = await self.service().recommend(self.request())

        self.assertEqual(self.model.ainvoke.await_count, 2)
        self.assertEqual(len(result.learning_outcome[0].related_esco_skill), 1)
        self.assertEqual(result.stats.llm.totals.repair_count, 1)

    async def test_invalid_pairs_after_repair_are_omitted_without_failing_outcomes(
        self,
    ):
        invalid = response(
            validation_response(
                [
                    {
                        "outcomeId": "lo-1",
                        "candidateId": "c999",
                        "decision": "supported",
                        "reason": "Unlisted pair.",
                    }
                ]
            )
        )
        self.model.ainvoke.side_effect = [invalid, invalid]

        result = await self.service().recommend(self.request())

        self.assertEqual(self.model.ainvoke.await_count, 2)
        self.assertEqual(len(result.learning_outcome), 1)
        self.assertEqual(result.learning_outcome[0].related_esco_skill, [])
        self.assertIn(
            "INVALID_SKILL_PAIRS_OMITTED", [warning.code for warning in result.warnings]
        )

    async def test_validation_stage_uses_the_longer_validation_timeout(self):
        timeouts = []

        async def validate(messages):
            await asyncio.sleep(0)
            return response(decisions(messages))

        real_wait_for = asyncio.wait_for

        async def spy(coro, timeout):
            timeouts.append(timeout)
            return await real_wait_for(coro, timeout=timeout)

        self.model.ainvoke.side_effect = validate
        with patch("src.models.ElmLoqMetadataRecommender.asyncio.wait_for", new=spy):
            await self.service(validation_timeout_seconds=90).recommend(self.request())
        # The outer request budget (120 s default) wraps the validation call (90 s).
        self.assertEqual(timeouts, [120, 90.0])

    async def test_agent_construction_failure_is_not_counted_as_a_provider_call(self):
        self.create_agent_mock.side_effect = NotImplementedError()
        with self.assertRaises(RecommendationUnavailable) as caught:
            await self.service().recommend(self.request())
        self.assertEqual(caught.exception.stats.llm.totals.request_count, 0)
        self.model.ainvoke.assert_not_called()

    async def test_validation_timeout_expires_and_reports_the_stage(self):
        async def hang(messages):
            await asyncio.Event().wait()

        self.model.ainvoke.side_effect = hang
        with patch(
            "src.models.ElmLoqMetadataRecommender.asyncio.sleep", new=AsyncMock()
        ):
            with self.assertRaises(RecommendationTimeout) as caught:
                await self.service(validation_timeout_seconds=5).recommend(
                    self.request()
                )
        self.assertIn("validation", str(caught.exception))
        calls = caught.exception.stats.llm.calls
        self.assertEqual([call.stage for call in calls], ["validation", "validation"])
        self.assertTrue(all(call.status == "error" for call in calls))

    async def test_non_validation_stages_keep_the_default_stage_timeout(self):
        timeouts = []
        real_wait_for = asyncio.wait_for

        async def spy(coro, timeout):
            timeouts.append(timeout)
            return await real_wait_for(coro, timeout=timeout)

        self.model.ainvoke.side_effect = lambda messages: response(
            {
                "learningOutcome": [
                    {
                        "title": "Animate with keyframes.",
                    }
                ],
                "learningOutcomeSummary": "Animate with keyframes.",
            }
        )
        with patch("src.models.ElmLoqMetadataRecommender.asyncio.wait_for", new=spy):
            await self.service(validation_timeout_seconds=90).recommend(
                ElmLoqMetadataRequest.model_validate(
                    {
                        "course": {"description": "Use keyframes."},
                        "sourceLanguage": "en",
                        "options": {
                            "stats": "full",
                            "validation": "semantic",
                            "minSimilarity": 0.5,
                        },
                    }
                )
            )
        # Only the extraction call runs; the outer request budget wraps it.
        self.assertEqual(timeouts, [120, 30])
