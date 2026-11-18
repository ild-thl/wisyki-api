import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.embeddings import HuggingFaceInferenceEmbeddings
from src.models.elm_loq_metadata import ElmLoqMetadataRequest
from src.models.ElmLoqMetadataRecommender import (
    ElmLoqMetadataRecommender,
    RecommendationError,
    RecommendationInputError,
    RecommendationTimeout,
    RecommendationUnavailable,
)
from src.models.SkillPrediction import SkillPrediction
from src.models.SkillRetriever import SkillRetriever
from src.routes.metadata_router import get_recommender, router


ESCO_ANIMATION = "http://data.europa.eu/esco/skill/eee58e6d-4587-42f0-b9a0-fe055cfe2044"
ESCO_RENDERING = "http://data.europa.eu/esco/skill/e8171178-9a60-40a6-ad3b-510e10cfebad"


def record(uri=ESCO_ANIMATION, title="Animation", framework="ESCO", distance=0.2):
    metadata = {
        "uri": uri,
        "title": title,
        "taxonomy": framework,
        "description": "A framework definition",
        "area": "An area",
        "activities": "Activities",
        "category": "A category",
        "URI": uri,
        "Kompetenzfacette": title,
        "ID": "greta-1",
        "Kompetenzaspekt": "An aspect",
        "Kompetenzbereich": "An area",
        "Kompetenzanforderungen": "Requirements",
        "Kompetenzbeschreibung": "A definition",
    }
    return (Document(page_content=f"{title}\nA framework definition", metadata=metadata), distance)


def query_result(rows):
    return {
        "documents": [[document.page_content for document, _ in row] for row in rows],
        "metadatas": [[document.metadata for document, _ in row] for row in rows],
        "distances": [[distance for _, distance in row] for row in rows],
    }


def outcome(title="Animate with keyframes."):
    return {"title": {"en": title}}


def request_data(**changes):
    result = {
        "learningOutcome": [outcome()],
        "sourceLanguage": "en",
        "targetLanguages": [],
        "options": {
            "validation": "semantic",
            "minSimilarity": 0.5,
            "summary": "disabled",
            "extraction": "disabled",
            "stats": "full",
        },
    }
    result.update(changes)
    return result


def message(value):
    return AIMessage(content=json.dumps(value))


def validation_response(decisions):
    groups = {}
    for item in decisions:
        group = groups.setdefault(
            item["outcomeId"],
            {
                "outcomeId": item["outcomeId"],
                "supported": [],
                "unsupported": [],
                "uncertain": [],
            },
        )
        if item["decision"] == "supported":
            group["supported"].append({
                "candidateId": item["candidateId"], "reason": item["reason"]
            })
        else:
            group[item["decision"]].append(item["candidateId"])
    return {"outcomes": list(groups.values())}


class RecommendationTests(unittest.IsolatedAsyncioTestCase):
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
        self.factory = Mock(return_value=(self.model, "fake-model"))
        self.service = ElmLoqMetadataRecommender(
            self.embedding, self.collection, self.factory, stage_timeout_seconds=0.1
        )

    async def recommend(self, data=None):
        return await self.service.recommend(
            ElmLoqMetadataRequest.model_validate(data or request_data())
        )

    def use_llm(self):
        return request_data(
            options={
                "validation": "llm",
                "extraction": "disabled",
                "summary": "disabled",
                "stats": "full",
            }
        )

    async def test_preserves_authored_metadata_without_constructing_llm_or_mutating_input(self):
        data = request_data(
            learningOutcome=[
                outcome("  Animate using keyframes.  "),
                outcome("Use a rig."),
            ],
            learningOutcomeSummary={"en": " Authored summary. "},
        )
        self.collection.query.return_value = query_result([[record()], [record()]])
        request = ElmLoqMetadataRequest.model_validate(data)
        before = request.model_dump()
        result = await self.service.recommend(request)
        self.assertEqual(request.model_dump(), before)
        self.assertEqual(len(result.learning_outcome), 2)
        self.assertEqual(
            result.learning_outcome[0].title["en"], "  Animate using keyframes.  "
        )
        self.assertEqual(
            result.learning_outcome_summary["en"], " Authored summary. "
        )
        self.assertEqual(result.learning_outcome_summary_origin, "provided")
        self.factory.assert_not_called()
        self.assertEqual(self.embedding.embed_documents.call_count, 1)

    async def test_reports_recommendation_progress_in_pipeline_order(self):
        progress = []

        async def report(stage, percent):
            progress.append((stage, percent))

        await self.service.recommend(
            ElmLoqMetadataRequest.model_validate(request_data()),
            progress_callback=report,
        )
        self.assertEqual(
            [stage for stage, _ in progress],
            [
                "reading_document",
                "identifying_skills",
                "validating_results",
                "localizing_metadata",
                "completing",
            ],
        )

    async def test_validates_each_outcome_edge_without_cross_contamination(self):
        data = self.use_llm()
        data["learningOutcome"] = [
            outcome("Animate objects."),
            outcome("Render objects."),
        ]
        self.collection.query.return_value = query_result(
            [[record(), record(ESCO_RENDERING, "Rendering")]] * 2
        )

        async def validate(messages):
            payload = json.loads(messages[1].content)["data"]
            pairs = payload["pairs"]
            skills = {skill["id"]: skill for skill in payload["skills"]}
            outcome_titles = {
                item["id"]: next(iter(item["title"].values()))
                for item in payload["outcomes"]
            }
            return message(validation_response(
                [
                        {
                            "outcomeId": pair["outcomeId"],
                            "candidateId": pair["candidateId"],
                            "decision": (
                                "supported"
                                if (
                                    "Animate" in outcome_titles[pair["outcomeId"]]
                                    and skills[pair["skillId"]]["title"] == "Animation"
                                )
                                or (
                                    "Render" in outcome_titles[pair["outcomeId"]]
                                    and skills[pair["skillId"]]["title"] == "Rendering"
                                )
                                else "unsupported"
                            ),
                            "reason": "Evidence-specific decision.",
                        }
                        for pair in pairs
                ]
            ))

        self.model.ainvoke.side_effect = validate
        result = await self.recommend(data)
        self.assertEqual(
            [[skill.uri for skill in item.related_esco_skill] for item in result.learning_outcome],
            [[ESCO_ANIMATION], [ESCO_RENDERING]],
        )
        self.assertEqual(result.stats.candidates_validated, 4)

    async def test_deduplicates_before_validation_but_keeps_shared_skill_across_outcomes(self):
        self.collection.query.return_value = query_result(
            [
                [record(distance=0.3), record(distance=0.1)],
                [record(distance=0.2)],
            ]
        )
        result = await self.recommend(
            request_data(learningOutcome=[outcome(), outcome("Use rigs.")])
        )
        self.assertEqual(
            [len(item.related_esco_skill) for item in result.learning_outcome], [1, 1]
        )
        self.assertAlmostEqual(
            result.learning_outcome[0].related_esco_skill[0].similarity_score, 0.9
        )
        self.assertEqual(result.stats.candidates_retrieved, 3)

    async def test_batches_duplicate_text_once_without_merging_outcome_identity(self):
        result = await self.recommend(
            request_data(learningOutcome=[outcome(), outcome()])
        )
        self.embedding.embed_documents.assert_called_once_with(["Animate with keyframes."])
        self.assertEqual(len(result.learning_outcome), 2)
        self.assertTrue(all(item.related_esco_skill for item in result.learning_outcome))

    async def test_separates_esco_subset_and_other_framework_and_preserves_urn(self):
        self.collection.query.side_effect = [
            query_result([[record()]]),
            query_result([[record(framework="DigCompESCO", distance=0.1)]]),
            query_result([[record("urn:digcomp3.0:Competence/1.1", framework="DigComp")]]),
        ]
        data = request_data()
        data["options"]["frameworks"] = ["ESCO", "DigCompESCO", "DigComp"]
        result = await self.recommend(data)
        item = result.learning_outcome[0]
        self.assertEqual(len(item.related_esco_skill), 1)
        self.assertEqual(item.related_esco_skill[0].framework, "DigCompESCO")
        self.assertEqual(item.related_skill[0].uri, "urn:digcomp3.0:Competence/1.1")

    async def test_empty_validated_matches_do_not_remove_outcome(self):
        data = self.use_llm()
        self.model.ainvoke.return_value = message(validation_response(
            [{
                    "outcomeId": "lo-1",
                    "candidateId": "c1",
                    "decision": "uncertain",
                    "reason": "No teaching evidence.",
            }]
        ))
        result = await self.recommend(data)
        self.assertEqual(len(result.learning_outcome), 1)
        self.assertEqual(result.learning_outcome[0].related_esco_skill, [])
        self.assertIn("NO_SKILL_MATCH", [warning.code for warning in result.warnings])

    async def test_rejects_unknown_crosswired_duplicate_or_missing_decisions(self):
        valid = {
            "outcomeId": "lo-1",
            "candidateId": "c1",
            "decision": "supported",
            "reason": "Evidence.",
        }
        for decisions in (
            [],
            [{**valid, "candidateId": "invented"}],
            [{**valid, "outcomeId": "another-outcome"}],
            [valid, valid],
        ):
            with self.subTest(decisions=decisions):
                self.model.ainvoke.return_value = message(validation_response(decisions))
                self.model.ainvoke.reset_mock()
                with self.assertRaises(RecommendationError):
                    await self.recommend(self.use_llm())
                self.assertEqual(self.model.ainvoke.await_count, 2)

    async def test_extracts_verified_quotes_and_generates_summary_without_skill_labels(self):
        data = {
            "course": {"description": "We teach animation using keyframes."},
            "sourceLanguage": "en",
            "options": {
                "validation": "semantic",
                "minSimilarity": 0.5,
                "stats": "full",
            },
        }
        self.collection.query.return_value = query_result([[record(title="Corpus-only label")]])
        self.model.ainvoke.side_effect = [
            message({"learningOutcome": [{
                "title": "Animate using keyframes.",
                "evidence": [{"field": "course.description", "quote": "animation using keyframes"}],
            }]}),
            message({"text": "Create animations using keyframes."}),
        ]
        result = await self.recommend(data)
        self.assertEqual(result.learning_outcome[0].origin, "extracted")
        self.assertEqual(result.learning_outcome_summary_origin, "generated")
        summary_prompt = self.model.ainvoke.call_args.args[0][1].content
        self.assertNotIn("Corpus-only label", summary_prompt)
        self.assertEqual(
            [item.stage for item in result.stats.used_models],
            ["extraction", "summary"],
        )
        self.assertEqual(
            result.learning_outcome[0].title["en"],
            "Animate using keyframes.",
        )
        self.assertEqual(
            result.learning_outcome_summary["en"],
            "Create animations using keyframes.",
        )

    async def test_localizes_supplied_outcomes_and_summary_in_one_stage(self):
        data = request_data(
            sourceLanguage="de",
            targetLanguages=["de", "en-GB"],
            learningOutcome=[
                {"title": {"de": "3D-Animationen erstellen."}}
            ],
            learningOutcomeSummary={"de": "Grundlagen der 3D-Animation."},
        )
        self.model.ainvoke.return_value = message({
            "learningOutcomes": [{
                "id": "lo-1",
                "title": {
                    "de": "3D-Animationen erstellen.",
                    "en-GB": "Create 3D animations.",
                },
            }],
            "learningOutcomeSummary": {
                "de": "Grundlagen der 3D-Animation.",
                "en-GB": "Learn the fundamentals of 3D animation.",
            },
        })

        result = await self.recommend(data)

        self.assertEqual(
            result.learning_outcome[0].title,
            {"de": "3D-Animationen erstellen.", "en-GB": "Create 3D animations."},
        )
        self.assertEqual(
            result.learning_outcome_summary,
            {
                "de": "Grundlagen der 3D-Animation.",
                "en-GB": "Learn the fundamentals of 3D animation.",
            },
        )
        self.assertEqual([item.stage for item in result.stats.used_models], ["translation"])
        self.assertEqual(self.model.ainvoke.await_count, 1)
        self.assertEqual(self.embedding.embed_documents.call_count, 1)

    async def test_credential_generation_uses_neutral_purpose_and_localizes_generated_text(self):
        data = {
            "course": {
                "title": "3D-Animation in Blender",
                "description": "Keyframes werden zur Animation von Objekten verwendet.",
                "targetGroup": "Einsteiger.",
                "entryRequirement": "Blender-Grundkenntnisse.",
                "courseContentSummary": "Keyframes und Objekte animieren.",
            },
            "sourceLanguage": "de",
            "targetLanguages": ["de", "en-GB"],
            "purpose": "credential",
            "options": {
                "validation": "semantic",
                "minSimilarity": 0.5,
                "stats": "full",
            },
        }
        self.model.ainvoke.side_effect = [
            message({"learningOutcome": [{
                "title": "3D-Objekte mit Keyframes animieren.",
                "evidence": [{
                    "field": "course.description",
                    "quote": "Keyframes werden zur Animation von Objekten verwendet",
                }],
            }]}),
            message({"text": "Fähigkeit, 3D-Objekte mit Keyframes zu animieren."}),
            message({
                "learningOutcomes": [{
                    "id": "lo-1",
                    "title": {
                        "de": "3D-Objekte mit Keyframes animieren.",
                        "en-GB": "Animate 3D objects using keyframes.",
                    },
                }],
                "learningOutcomeSummary": {
                    "de": "Fähigkeit, 3D-Objekte mit Keyframes zu animieren.",
                    "en-GB": "Ability to animate 3D objects using keyframes.",
                },
            }),
        ]

        result = await self.recommend(data)

        self.assertEqual(
            result.learning_outcome[0].title["en-GB"],
            "Animate 3D objects using keyframes.",
        )
        self.assertEqual(
            result.learning_outcome_summary["en-GB"],
            "Ability to animate 3D objects using keyframes.",
        )
        extraction_prompt = self.model.ainvoke.call_args_list[0].args[0][0].content
        summary_prompt = self.model.ainvoke.call_args_list[1].args[0][0].content
        self.assertIn("capability represented by the credential", extraction_prompt)
        self.assertIn("any particular learner", summary_prompt)
        extraction_data = json.loads(
            self.model.ainvoke.call_args_list[0].args[0][1].content
        )["data"]
        self.assertIn("course.targetGroup", extraction_data)
        self.assertIn("course.entryRequirement", extraction_data)
        self.assertEqual(
            [item.stage for item in result.stats.used_models],
            ["extraction", "summary", "translation"],
        )

    async def test_target_group_is_context_not_outcome_evidence(self):
        self.model.ainvoke.side_effect = [
            message({"learningOutcome": [{
                "title": "Beginners can use Blender.",
                "evidence": [{
                    "field": "course.targetGroup",
                    "quote": "Beginners",
                }],
            }]}),
            message({"learningOutcome": []}),
        ]
        result = await self.recommend({
            "sourceLanguage": "en",
            "course": {
                "description": "A course about 3D animation.",
                "targetGroup": "Beginners.",
            },
            "options": {"stats": "full"},
        })
        self.assertEqual(result.learning_outcome, [])
        self.assertEqual(self.model.ainvoke.await_count, 2)

    async def test_rejects_duplicate_or_unknown_empty_outcome_groups(self):
        valid = validation_response([{
            "outcomeId": "lo-1", "candidateId": "c1",
            "decision": "supported", "reason": "Direct evidence.",
        }])["outcomes"][0]
        for identifier in ("lo-1", "unknown"):
            with self.subTest(identifier=identifier):
                self.model.ainvoke.return_value = message({"outcomes": [
                    valid,
                    {
                        "outcomeId": identifier, "supported": [],
                        "unsupported": [], "uncertain": [],
                    },
                ]})
                with self.assertRaises(RecommendationError):
                    await self.recommend(self.use_llm())

    async def test_fabricated_quote_is_an_explicit_failure_after_one_repair(self):
        self.model.ainvoke.return_value = message({"learningOutcome": [{
            "title": "Animate objects.",
            "evidence": [{"field": "course.description", "quote": "A fabricated quote"}],
        }]})
        with self.assertRaises(RecommendationError):
            await self.recommend({
                "sourceLanguage": "en",
                "course": {"description": "We teach keyframes."},
            })
        self.assertEqual(self.model.ainvoke.await_count, 2)
        self.collection.query.assert_not_called()

    async def test_empty_extraction_skips_retrieval_and_summary(self):
        self.model.ainvoke.return_value = message({"learningOutcome": []})
        result = await self.recommend({
            "sourceLanguage": "en",
            "course": {"description": "Room 12, Tuesday."},
            "options": {"stats": "full"},
        })
        self.assertEqual(result.learning_outcome, [])
        self.assertIsNone(result.learning_outcome_summary)
        self.assertEqual(result.stats.summary, "no_outcomes")
        self.assertEqual(result.warnings[0].code, "NO_SUPPORTED_LEARNING_OUTCOMES")
        self.collection.query.assert_not_called()
        self.assertEqual(self.model.ainvoke.await_count, 1)

    async def test_recovers_once_from_invalid_schema_using_original_evidence(self):
        self.model.ainvoke.side_effect = [
            AIMessage(content="not JSON"),
            message({"learningOutcome": []}),
        ]
        result = await self.recommend({
            "sourceLanguage": "en",
            "course": {"description": "On campus Tuesday."},
        })
        self.assertEqual(result.learning_outcome, [])
        self.assertEqual(self.model.ainvoke.await_count, 2)

    async def test_accepts_fully_fenced_json_without_spending_a_repair_call(self):
        self.model.ainvoke.return_value = AIMessage(
            content='```json\n{"learningOutcome": []}\n```'
        )
        result = await self.recommend({
            "sourceLanguage": "en",
            "course": {"description": "On campus Tuesday."},
        })
        self.assertEqual(result.learning_outcome, [])
        self.assertEqual(self.model.ainvoke.await_count, 1)

    async def test_schema_or_provider_failure_does_not_log_course_content(self):
        text = "Private course marker never to be logged"
        self.model.ainvoke.return_value = AIMessage(content=text)
        with self.assertLogs("src.models.ElmLoqMetadataRecommender", level="WARNING") as logs:
            with self.assertRaises(RecommendationError):
                await self.recommend({
                    "sourceLanguage": "en",
                    "course": {"description": text},
                })
        self.assertNotIn(text, "\n".join(logs.output))

    async def test_summary_failure_is_not_a_successful_partial_response(self):
        data = request_data()
        data["options"]["summary"] = "auto"
        self.model.ainvoke.return_value = message({"text": ""})
        with self.assertRaises(RecommendationError):
            await self.recommend(data)

    async def test_request_state_is_not_shared_between_recommendations(self):
        first, second = await asyncio.gather(self.recommend(), self.recommend())
        self.assertIsNot(first.learning_outcome[0], second.learning_outcome[0])
        first.learning_outcome[0].related_esco_skill.clear()
        self.assertEqual(len(second.learning_outcome[0].related_esco_skill), 1)

    async def test_similarity_floor_and_caps_are_per_framework_and_outcome(self):
        self.collection.query.return_value = query_result([[
            record(distance=0.6),
            record(ESCO_RENDERING, "Rendering", distance=0.2),
            record(distance=0.1),
        ]])
        data = request_data()
        data["options"]["maxSkillsPerFramework"] = 1
        result = await self.recommend(data)
        self.assertEqual(
            [skill.uri for skill in result.learning_outcome[0].related_esco_skill],
            [ESCO_ANIMATION],
        )
        self.assertEqual(result.stats.candidates_retrieved, 3)

    async def test_one_framework_or_outcome_does_not_consume_anothers_result_cap(self):
        self.collection.query.side_effect = [
            query_result([
                [record(), record(ESCO_RENDERING, "Rendering", distance=0.1)],
                [record(distance=0.1), record(ESCO_RENDERING, "Rendering")],
            ]),
            query_result([
                [record("urn:test:digital-one", framework="DigComp")],
                [record("urn:test:digital-two", framework="DigComp")],
            ]),
        ]
        data = request_data(
            learningOutcome=[outcome(), outcome("Use rigs.")]
        )
        data["options"].update(frameworks=["ESCO", "DigComp"], maxSkillsPerFramework=1)
        result = await self.recommend(data)
        self.assertEqual(
            [
                (len(item.related_esco_skill), len(item.related_skill))
                for item in result.learning_outcome
            ],
            [(1, 1), (1, 1)],
        )
        self.assertEqual(result.learning_outcome[0].related_esco_skill[0].uri, ESCO_RENDERING)
        self.assertEqual(result.learning_outcome[1].related_esco_skill[0].uri, ESCO_ANIMATION)

    async def test_malformed_embeddings_and_records_are_not_empty_successes(self):
        for vectors in ([], [[0, 0]], [[float("nan"), 1]], [[True, 1]]):
            with self.subTest(vectors=vectors):
                self.embedding.embed_documents.side_effect = None
                self.embedding.embed_documents.return_value = vectors
                with self.assertRaises(RecommendationError):
                    await self.recommend()
        self.embedding.embed_documents.return_value = [[1, 0]]
        for rows in (
            {"documents": [[]], "metadatas": None, "distances": [[]]},
            query_result([[record(uri="urn:invented:esco")]]),
            query_result([[record(distance=float("nan"))]]),
            query_result([[record(framework="DigComp")]]),
        ):
            with self.subTest(rows=rows):
                self.collection.query.return_value = rows
                with self.assertRaises(RecommendationError):
                    await self.recommend()

    async def test_unavailable_framework_and_embedding_timeout_are_explicit(self):
        self.collection.query.return_value = query_result([[]])
        with self.assertRaises(RecommendationUnavailable):
            await self.recommend()
        self.embedding.embed_documents.side_effect = requests.Timeout()
        with self.assertRaises(RecommendationTimeout):
            await self.recommend()

    async def test_total_deadline_and_stage_timeout_are_not_successes(self):
        async def slow_model(messages):
            await asyncio.sleep(1)
            return message({"learningOutcome": []})

        self.model.ainvoke.side_effect = slow_model
        self.service.timeout_seconds = 0.02
        with self.assertRaises(RecommendationTimeout):
            await self.recommend({
                "sourceLanguage": "en",
                "course": {"description": "A course."},
            })

    async def test_stage_timeout_retries_once_without_semantic_fallback(self):
        self.model.ainvoke.side_effect = asyncio.TimeoutError()
        with patch(
            "src.models.ElmLoqMetadataRecommender.asyncio.sleep", new=AsyncMock()
        ):
            with self.assertRaises(RecommendationTimeout):
                await self.recommend(self.use_llm())
        self.assertEqual(self.model.ainvoke.await_count, 2)

    async def test_missing_provider_configuration_does_not_expose_internal_error(self):
        self.factory.side_effect = ValueError("Private configuration details.")
        with self.assertRaises(RecommendationUnavailable) as error:
            await self.recommend(self.use_llm())
        self.assertNotIn("Private configuration", str(error.exception))

    async def test_candidate_budget_is_checked_before_embedding(self):
        data = request_data(
            learningOutcome=[outcome(f"Outcome {index}") for index in range(20)]
        )
        data["options"].update(
            frameworks=["ESCO", "GRETA"], maxSkillsPerFramework=10, summary="auto"
        )
        with self.assertRaises(RecommendationInputError):
            await self.recommend(data)
        self.embedding.embed_documents.assert_not_called()
        self.factory.assert_not_called()

    async def test_native_chroma_batch_query_preserves_cosine_scores_and_framework_filters(self):
        import chromadb
        from chromadb.config import Settings
        from uuid import uuid4

        client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        name = f"metadata-test-{uuid4().hex}"
        collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        try:
            collection.add(
                ids=["esco", "digcomp"],
                embeddings=[[1.0, 0.0], [1.0, 0.0]],
                documents=["Animation definition", "Digital skill definition"],
                metadatas=[
                    {"uri": ESCO_ANIMATION, "title": "Animation", "taxonomy": "ESCO"},
                    record("urn:digcomp3.0:Competence/1.1", framework="DigComp")[0].metadata,
                ],
            )
            data = request_data()
            data["options"]["frameworks"] = ["ESCO", "DigComp"]
            service = ElmLoqMetadataRecommender(self.embedding, collection, self.factory)
            result = await service.recommend(ElmLoqMetadataRequest.model_validate(data))
            item = result.learning_outcome[0]
            self.assertEqual(item.related_esco_skill[0].uri, ESCO_ANIMATION)
            self.assertAlmostEqual(item.related_esco_skill[0].similarity_score, 1.0)
            self.assertEqual(item.related_skill[0].framework, "DigComp")
            self.factory.assert_not_called()
        finally:
            client.delete_collection(name)


class MetadataContractTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(router)
        self.recommender = Mock()
        self.recommender.recommend = AsyncMock()
        self.app.dependency_overrides[get_recommender] = lambda: self.recommender
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def test_target_languages_default_to_the_required_source_language(self):
        payload = request_data()
        payload.pop("targetLanguages")
        request = ElmLoqMetadataRequest.model_validate(payload)
        self.assertEqual(request.output_languages, ["en"])

    def test_invalid_inputs_fail_before_provider_calls(self):
        invalid = [
            {},
            {"course": {"description": "A course."}},
            {"course": {"description": "   "}},
            request_data(learningOutcome=[{"id": "client-id", **outcome()}]),
            request_data(language="en"),
            request_data(options={"validation": "semantic"}),
            request_data(options={"frameworks": ["ESCO", "ESCO"]}),
            request_data(options={"frameworks": ["unknown"]}),
            request_data(options={"maxSkillsPerFramework": True}),
            request_data(sourceLanguage="not_a_language"),
            request_data(targetLanguages=["en", "en"]),
            request_data(course={"description": "x" * 12001}),
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.client.post("/recommend/elm-loq-metadata", json=payload)
                self.assertEqual(response.status_code, 422)
        self.recommender.recommend.assert_not_called()

    def test_maps_dependency_errors_without_internal_details(self):
        for error, status in (
            (RecommendationError("Invalid upstream response."), 502),
            (RecommendationUnavailable("Provider unavailable."), 503),
            (RecommendationTimeout("Provider timed out."), 504),
            (RecommendationInputError("Candidate budget exceeded."), 422),
        ):
            with self.subTest(status=status):
                self.recommender.recommend.side_effect = error
                response = self.client.post("/recommend/elm-loq-metadata", json=request_data())
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json(), {"detail": str(error)})
                self.assertIn("X-Request-ID", response.headers)

    def test_http_response_preserves_outcome_associations_and_exact_elm_aliases(self):
        collection = Mock()
        collection.metadata = {"hnsw:space": "cosine"}
        collection.query.return_value = query_result([[record()]])
        embedding = Mock()
        embedding.embed_documents.return_value = [[1.0, 0.0]]
        service = ElmLoqMetadataRecommender(embedding, collection)
        self.app.dependency_overrides[get_recommender] = lambda: service
        response = self.client.post("/recommend/elm-loq-metadata", json=request_data())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("skills", body)
        item = body["learningOutcome"][0]
        self.assertNotIn("id", item)
        self.assertEqual(
            item["title"], {"en": "Animate with keyframes."}
        )
        self.assertEqual(item["relatedSkill"], [])
        self.assertEqual(item["relatedESCOSkill"][0]["uri"], ESCO_ANIMATION)
        self.assertAlmostEqual(item["relatedESCOSkill"][0]["similarityScore"], 0.8)
        self.assertIsNone(body["learningOutcomeSummary"])
        self.assertIn("X-Request-ID", response.headers)
        openapi = self.client.get("/openapi.json").json()
        request_schema = openapi["components"]["schemas"]["ElmLoqMetadataRequest"]
        self.assertIn("sourceLanguage", request_schema["required"])
        self.assertNotIn("language", request_schema["properties"])
        self.assertNotIn("targetLanguages", request_schema["required"])
        self.assertIn(
            "sourceLanguage",
            request_schema["properties"]["targetLanguages"]["description"],
        )
        self.assertNotIn(
            "id", openapi["components"]["schemas"]["OutcomeInput"]["properties"]
        )
        self.assertNotIn(
            "id",
            openapi["components"]["schemas"]["LearningOutcomeRecommendation"]["properties"],
        )

    def test_stats_are_omitted_or_reduced_according_to_request(self):
        collection = Mock()
        collection.metadata = {"hnsw:space": "cosine"}
        collection.query.return_value = query_result([[record()]])
        embedding = Mock()
        embedding.embed_documents.return_value = [[1.0, 0.0]]
        service = ElmLoqMetadataRecommender(embedding, collection)
        self.app.dependency_overrides[get_recommender] = lambda: service

        for level in ("default", "none", "summary", "full"):
            with self.subTest(level=level):
                data = request_data()
                if level == "default":
                    data["options"].pop("stats")
                else:
                    data["options"]["stats"] = level
                response = self.client.post("/recommend/elm-loq-metadata", json=data)
                self.assertEqual(response.status_code, 200)
                self.assertIn("X-Request-ID", response.headers)
                if level in ("default", "none"):
                    self.assertNotIn("stats", response.json())
                elif level == "summary":
                    self.assertIn("stats", response.json())
                    self.assertNotIn("llm", response.json()["stats"])
                else:
                    self.assertIn("llm", response.json()["stats"])

    def test_removed_chatsearch_v3_is_not_an_alias(self):
        response = self.client.post("/v3/chatsearch", json=request_data())
        self.assertEqual(response.status_code, 404)

    def test_failed_llm_schema_validation_still_exposes_reported_billable_usage(self):
        collection = Mock()
        collection.metadata = {"hnsw:space": "cosine"}
        collection.query.return_value = query_result([[record()]])
        embedding = Mock()
        embedding.embed_documents.return_value = [[1.0, 0.0]]
        model = Mock()
        model.ainvoke = AsyncMock(return_value=AIMessage(
            content="invalid JSON",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ))
        service = ElmLoqMetadataRecommender(
            embedding, collection, lambda stage: (model, "test-model")
        )
        self.app.dependency_overrides[get_recommender] = lambda: service
        data = request_data()
        data["options"]["validation"] = "llm"
        response = self.client.post("/recommend/elm-loq-metadata", json=data)
        self.assertEqual(response.status_code, 502)
        totals = response.json()["stats"]["llm"]["totals"]
        self.assertEqual(totals["requestCount"], 2)
        self.assertEqual(totals["tokenUsage"]["totalTokens"], 30)
        self.assertTrue(totals["usageComplete"])


class SharedIntegrationTests(unittest.TestCase):
    def test_framework_conversion_keeps_legacy_adapter_behavior(self):
        retriever = SkillRetriever.__new__(SkillRetriever)
        for framework in (
            "ESCO", "DigCompESCO", "GRETA", "DigComp", "DigCompEdu", "FutureSkills2030"
        ):
            with self.subTest(framework=framework):
                document, _ = record("urn:test:skill", "Expected title", framework)
                prediction = retriever.create_prediction((document, 0.8))
                self.assertEqual(prediction.uri, "urn:test:skill")
                self.assertEqual(prediction.title, "Expected title")
                self.assertEqual(prediction.score, 0.8)
                self.assertEqual(prediction.taxonomy, framework)
                self.assertEqual(prediction, SkillPrediction.from_document((document, 0.8)))

    def test_embedding_transport_keeps_payload_and_formats_with_finite_timeout(self):
        embedding = HuggingFaceInferenceEmbeddings("test-token", "http://embedding.invalid")
        for payload in ([[1.0, 0.5]], [{"embedding": [1.0, 0.5]}]):
            with self.subTest(payload=payload), patch("src.embeddings.requests.post") as post:
                post.return_value.json.return_value = payload
                self.assertEqual(embedding.embed_documents(["An outcome"]), [[1.0, 0.5]])
                self.assertEqual(post.call_args.kwargs["json"], {"inputs": ["An outcome"]})
                self.assertEqual(post.call_args.kwargs["timeout"], (5, 30))
                post.return_value.raise_for_status.assert_called_once()
