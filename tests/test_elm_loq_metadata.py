import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.embeddings import HuggingFaceInferenceEmbeddings, load_embedding_function
from src.models.elm_loq_metadata import (
    ClassifiedExtractionResult,
    ElmLoqMetadataRequest,
)
from src.models.ElmLoqMetadataRecommender import (
    ElmLoqMetadataRecommender,
    MetadataRecommendationRun,
    RecommendationError,
    RecommendationInputError,
    RecommendationTimeout,
    RecommendationUnavailable,
    eqf_vocabulary,
    isced_f_vocabulary,
)
from src.models.SkillPrediction import SkillPrediction
from src.models.metadata_llm import MetadataLLMSettings
from src.models.SkillRetriever import SkillRetriever
from src.routes.metadata_router import get_recommender, router

ESCO_ANIMATION = "http://data.europa.eu/esco/skill/eee58e6d-4587-42f0-b9a0-fe055cfe2044"
ESCO_RENDERING = "http://data.europa.eu/esco/skill/e8171178-9a60-40a6-ad3b-510e10cfebad"
ISCED_F_0611 = "http://data.europa.eu/snb/isced-f/0611"


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
    return (
        Document(page_content=f"{title}\nA framework definition", metadata=metadata),
        distance,
    )


def isced_f_record(uri=ISCED_F_0611, code="0611", distance=0.15):
    metadata = {
        "taxonomy": "ISCED_F",
        "scheme": "http://data.europa.eu/snb/isced-f/25831c2",
        "uri": uri,
        "code": code,
        "label_de": "Computernutzung",
        "label_en": "Computer use",
    }
    return (
        Document(page_content="ISCED-F 0611 Computernutzung", metadata=metadata),
        distance,
    )


def query_result(rows):
    return {
        "documents": [[document.page_content for document, _ in row] for row in rows],
        "metadatas": [[document.metadata for document, _ in row] for row in rows],
        "distances": [[distance for _, distance in row] for row in rows],
    }


def outcome(title="Animate with keyframes."):
    return {"title": title}


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
    return {
        "supported": [
            {
                "pairId": item.get("pairId") or f"p-{item['candidateId']}",
                "reason": item["reason"],
            }
            for item in decisions
            if item["decision"] == "supported"
        ]
    }


def fake_agent(model, response_format):
    async def ainvoke(state):
        received = await model.ainvoke(state["messages"])
        content = received.content.strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        try:
            structured = json.loads(content)
        except (TypeError, ValueError):
            structured = {"invalid": True}
        return {
            "messages": [*state["messages"], received],
            "structured_response": structured,
        }

    return SimpleNamespace(ainvoke=ainvoke)


class RecommendationTests(unittest.IsolatedAsyncioTestCase):
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
        self.factory = Mock(return_value=(self.model, "fake-model"))
        self.create_agent = patch(
            "src.models.ElmLoqMetadataRecommender.create_agent",
            side_effect=fake_agent,
        )
        self.create_agent.start()
        self.addCleanup(self.create_agent.stop)
        self.service = ElmLoqMetadataRecommender(
            self.embedding,
            self.collection,
            self.factory,
            stage_timeout_seconds=0.1,
            settings=MetadataLLMSettings(),
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

    def use_isced_candidates(self, candidates):
        skill_result = self.collection.query.return_value
        isced_result = query_result([candidates])

        def query(**kwargs):
            if kwargs.get("where") == {"taxonomy": "ISCED_F"}:
                return isced_result
            return skill_result

        self.collection.query.side_effect = query

    async def test_eqf_descriptors_load_from_authored_german_vocabulary(self):
        eqf_vocabulary.cache_clear()

        levels = eqf_vocabulary("de")

        self.assertEqual(set(levels), set(range(1, 9)))
        self.assertEqual(levels[4]["uri"], "http://data.europa.eu/snb/eqf/4")
        self.assertEqual(levels[4]["label"], "Niveau 4")
        self.assertIn("Breites Spektrum", levels[4]["definition"])

    async def test_eqf_descriptors_fall_back_to_german_for_unsupported_language(self):
        eqf_vocabulary.cache_clear()

        levels = eqf_vocabulary("fr")

        self.assertEqual(set(levels), set(range(1, 9)))
        self.assertEqual(levels[4]["label"], "Niveau 4")
        self.assertIn("Breites Spektrum", levels[4]["definition"])

    async def test_isced_retrieval_rejects_nonterminal_codes(self):
        candidate, distance = isced_f_record(
            uri="http://data.europa.eu/snb/isced-f/061",
            code="061",
        )
        self.collection.query.return_value = query_result([[(candidate, distance)]])

        request = ElmLoqMetadataRequest.model_validate(
            request_data(
                options={
                    "extraction": "disabled",
                    "summary": "disabled",
                    "classifications": ["ISCED_F"],
                }
            )
        )
        run = MetadataRecommendationRun(self.service, request)

        with self.assertRaises(RecommendationError):
            await run.retrieve_isced_f_candidates("Digital tools", 5)

    async def test_preserves_authored_metadata_without_constructing_llm_or_mutating_input(
        self,
    ):
        data = request_data(
            learningOutcome=[
                outcome("  Animate using keyframes.  "),
                outcome("Use a rig."),
            ],
            learningOutcomeSummary=" Authored summary. ",
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
        self.assertEqual(result.learning_outcome_summary["en"], " Authored summary. ")
        self.assertIsNone(result.classification_suggestions)
        self.factory.assert_not_called()
        self.assertEqual(self.embedding.embed_queries.call_count, 1)

    async def test_classifications_are_opt_in_and_return_only_controlled_candidates(
        self,
    ):
        self.model.ainvoke.return_value = message(
            {
                "eqfLevel": 4,
                "eqfReason": "The course develops independent use of digital tools.",
            }
        )
        self.use_isced_candidates(
            [
                isced_f_record(
                    uri="http://data.europa.eu/snb/isced-f/0613",
                    code="0613",
                    distance=0.3,
                ),
                isced_f_record(distance=0.15),
            ]
        )
        data = request_data(
            course={
                "title": "Digitale Werkzeuge",
                "description": "Teilnehmende nutzen digitale Anwendungen.",
            },
            options={
                "validation": "semantic",
                "minSimilarity": 0.5,
                "extraction": "disabled",
                "summary": "disabled",
                "stats": "full",
                "classifications": ["EQF", "ISCED_F"],
            },
        )

        result = await self.recommend(data)

        self.assertEqual(result.classification_suggestions.eqf_level.level, 4)
        self.assertEqual(
            result.classification_suggestions.eqf_level.reason,
            "The course develops independent use of digital tools.",
        )
        self.assertNotIn(
            "status",
            result.model_dump(by_alias=True)["classificationSuggestions"]["eqfLevel"],
        )
        self.assertEqual(result.classification_suggestions.isced_f[0].uri, ISCED_F_0611)
        self.assertEqual(result.classification_suggestions.isced_f[0].code, "0611")
        self.assertAlmostEqual(
            result.classification_suggestions.isced_f[0].similarity_score, 0.85
        )
        self.assertEqual(result.stats.classification_candidates_retrieved, 2)
        self.assertEqual(self.model.ainvoke.await_count, 1)
        llm_payload = json.loads(self.model.ainvoke.call_args.args[0][1].content)[
            "data"
        ]
        self.assertEqual(
            llm_payload["iscedFCandidates"],
            [
                {"code": "0613", "title": "Computer use"},
                {"code": "0611", "title": "Computer use"},
            ],
        )
        self.assertEqual(self.collection.query.call_count, 2)
        self.factory.assert_called_once_with("classification")

    async def test_isced_semantic_selection_abstains_below_the_similarity_floor(self):
        self.use_isced_candidates([isced_f_record(distance=0.8)])
        data = request_data(
            options={
                "validation": "semantic",
                "minSimilarity": 0.5,
                "extraction": "disabled",
                "summary": "disabled",
                "classifications": ["ISCED_F"],
            }
        )

        result = await self.recommend(data)

        self.assertEqual(len(result.learning_outcome), 1)
        self.assertEqual(result.classification_suggestions.isced_f, [])
        self.assertIn(
            "CLASSIFICATION_ABSTAINED", [item.code for item in result.warnings]
        )
        self.assertEqual(self.model.ainvoke.await_count, 0)
        self.factory.assert_not_called()

    async def test_isced_llm_selects_one_exact_candidate_from_code_title_context(self):
        self.use_isced_candidates(
            [
                isced_f_record(
                    uri="http://data.europa.eu/snb/isced-f/0613",
                    code="0613",
                    distance=0.1,
                ),
                isced_f_record(
                    uri="http://data.europa.eu/snb/isced-f/0211",
                    code="0211",
                    distance=0.2,
                ),
                isced_f_record(
                    uri="http://data.europa.eu/snb/isced-f/0611",
                    code="0611",
                    distance=0.3,
                ),
            ]
        )
        isced_reason = "The main learning activity is user-centred design. " + (
            "Evidence. " * 30
        )

        async def classify(messages):
            payload = json.loads(messages[1].content)["data"]
            if "iscedFCandidates" in payload:
                self.classification_candidates = payload["iscedFCandidates"]
                return message(
                    {
                        "eqfLevel": 4,
                        "eqfReason": "The course integrates planning and independent decisions.",
                        "iscedFCode": "0211",
                        "iscedFReason": isced_reason,
                    }
                )
            return message({"supported": []})

        self.model.ainvoke.side_effect = classify
        data = request_data(
            options={
                "validation": "llm",
                "extraction": "disabled",
                "summary": "disabled",
                "classifications": ["EQF", "ISCED_F"],
            }
        )
        result = await self.recommend(data)
        fields = result.classification_suggestions.isced_f
        self.assertEqual([item.code for item in fields], ["0211"])
        self.assertEqual(result.classification_suggestions.eqf_level.level, 4)
        self.assertEqual(len(self.classification_candidates), 3)
        self.assertTrue(
            all(
                set(candidate) == {"code", "title"}
                for candidate in self.classification_candidates
            )
        )
        self.assertAlmostEqual(fields[0].similarity_score, 0.8)
        body = result.model_dump(by_alias=True)["classificationSuggestions"]
        self.assertIn("title", body["eqfLevel"])
        self.assertEqual(
            body["eqfLevel"]["reason"],
            "The course integrates planning and independent decisions.",
        )
        self.assertIn("similarityScore", body["iscedF"][0])
        self.assertEqual(
            body["iscedF"][0]["reason"],
            isced_reason,
        )
        self.assertNotIn("label", body["eqfLevel"])
        self.assertNotIn("preferredLabel", body["iscedF"][0])

    async def test_isced_llm_accepts_valid_code_outside_semantic_candidates(self):
        code = "0613"
        uri, labels = isced_f_vocabulary()[code]
        self.use_isced_candidates(
            [
                isced_f_record(
                    uri="http://data.europa.eu/snb/isced-f/0212",
                    code="0212",
                    distance=0.1,
                )
            ]
        )
        self.collection.get.return_value = {
            "documents": ["ISCED-F 0613: Software and applications development."],
            "metadatas": [
                {
                    "taxonomy": "ISCED_F",
                    "scheme": "http://data.europa.eu/snb/isced-f/25831c2",
                    "uri": uri,
                    "code": code,
                    "label_de": labels["de"],
                    "label_en": labels["en"],
                }
            ],
        }
        async def classify(messages):
            payload = json.loads(messages[1].content)["data"]
            if "iscedFCandidates" in payload:
                return message(
                    {
                        "iscedFCode": code,
                        "iscedFReason": "The course develops a digital application prototype.",
                    }
                )
            return message({"supported": []})

        self.model.ainvoke.side_effect = classify

        result = await self.recommend(
            request_data(
                sourceLanguage="de",
                course={"description": "UX Design with a clickable Figma prototype."},
                options={
                    "validation": "llm",
                    "extraction": "disabled",
                    "summary": "disabled",
                    "classifications": ["ISCED_F"],
                },
            )
        )

        selected = result.classification_suggestions.isced_f[0]
        self.assertEqual(selected.code, code)
        self.assertEqual(selected.uri, uri)
        self.assertIsNone(selected.similarity_score)
        self.collection.get.assert_called_once_with(
            where={"taxonomy": "ISCED_F", "code": code},
            include=["documents", "metadatas"],
        )
        payload = json.loads(self.model.ainvoke.call_args_list[-1].args[0][1].content)[
            "data"
        ]
        self.assertEqual(payload["iscedFCandidates"][0]["code"], "0212")

    async def test_validation_receives_complete_context_without_outcome_quotes(self):
        context = "Course context " * 500 + "Distinct final section."
        data = self.use_llm()
        data["course"] = {"description": context, "targetGroup": "Beginners"}
        self.model.ainvoke.return_value = message({"supported": []})
        result = await self.recommend(data)
        messages = self.model.ainvoke.call_args.args[0]
        payload = json.loads(messages[1].content)["data"]
        self.assertEqual(payload["course"]["course.description"], context)
        self.assertEqual(payload["course"]["course.targetGroup"], "Beginners")
        self.assertNotIn("evidence", payload["outcomes"][0])
        self.assertLess(len(messages[0].content), 1500)
        self.assertEqual(result.learning_outcome[0].related_esco_skill, [])

    async def test_retrieval_limit_does_not_cap_accepted_skills(self):
        rows = [
            record(
                f"http://data.europa.eu/esco/skill/00000000-0000-4000-8000-{index:012d}",
                f"Skill {index}",
            )
            for index in range(12)
        ]
        self.collection.query.return_value = query_result([rows])
        data = self.use_llm()
        data["options"]["candidateLimit"] = 12
        self.model.ainvoke.return_value = message(
            validation_response(
                [
                    {
                        "outcomeId": "lo-1",
                        "candidateId": f"c{index + 1}",
                        "decision": "supported",
                        "reason": "Taught by this outcome.",
                    }
                    for index in range(12)
                ]
            )
        )
        result = await self.recommend(data)
        self.assertEqual(self.collection.query.call_args.kwargs["n_results"], 12)
        self.assertEqual(len(result.learning_outcome[0].related_esco_skill), 12)

    async def test_pair_id_keeps_candidate_attached_to_its_original_outcome(self):
        self.collection.query.return_value = query_result([[record()], [record()]])
        data = self.use_llm()
        data["learningOutcome"] = [outcome(), outcome("Use rigs.")]
        self.model.ainvoke.return_value = message(
            {
                "supported": [
                    {
                        "pairId": "p-c1",
                        "reason": "Direct evidence.",
                    }
                ]
            }
        )
        result = await self.recommend(data)
        self.assertEqual(len(result.learning_outcome), 2)
        self.assertEqual(len(result.learning_outcome[0].related_esco_skill), 1)
        self.assertEqual(result.learning_outcome[1].related_esco_skill, [])

    async def test_semantic_comparison_shares_retrieval_and_does_not_treat_llm_as_truth(
        self,
    ):
        from src.scripts.evaluate_elm_skills import compare_skills

        self.collection.query.return_value = query_result(
            [
                [
                    record(distance=0.6),
                    record(ESCO_RENDERING, "Rendering", distance=0.5),
                ]
            ]
        )
        self.model.ainvoke.return_value = message(
            {
                "supported": [
                    {
                        "pairId": "p-c1",
                        "reason": "Explicitly taught.",
                    }
                ]
            }
        )
        report = await compare_skills(
            ElmLoqMetadataRequest.model_validate(request_data()),
            self.service,
        )
        self.collection.query.assert_called_once()
        self.model.ainvoke.assert_awaited_once()
        self.assertEqual(report["variants"]["semantic-0.4"]["candidateCount"], 2)
        self.assertEqual(report["variants"]["semantic-0.5"]["candidateCount"], 1)
        self.assertEqual(
            report["variants"]["semantic-0.5"]["llmAcceptedButBelowCutoff"],
            [
                {"outcomeIndex": 0, "uri": ESCO_ANIMATION},
            ],
        )
        self.assertIn("not ground truth", report["notice"])

    async def test_comparison_preserves_semantic_variants_when_llm_fails(self):
        from src.scripts.evaluate_elm_skills import compare_skills

        self.model.ainvoke.return_value = message({"unsupported": []})
        report = await compare_skills(
            ElmLoqMetadataRequest.model_validate(request_data()),
            self.service,
        )
        self.assertEqual(len(report["variants"]["semantic-0.4"]["links"]), 1)
        self.assertEqual(
            report["variants"]["llm-0.4"]["errorType"], "RecommendationError"
        )
        self.assertNotIn("notAcceptedByLlm", report["variants"]["semantic-0.4"])

    async def test_extraction_returns_classifications_without_another_model_call(self):
        description = "Teilnehmende nutzen digitale Anwendungen."
        eqf_reason = "The course practices independent use of digital tools. " + (
            "Evidence. " * 30
        )
        self.use_isced_candidates([isced_f_record(distance=0.15)])
        self.model.ainvoke.return_value = message(
            {
                "learningOutcome": [{"title": description}],
                "classifications": {
                    "eqfLevel": 3,
                    "eqfReason": eqf_reason,
                },
            }
        )
        result = await self.recommend(
            {
                "course": {"description": description},
                "sourceLanguage": "de",
                "options": {
                    "classifications": ["EQF", "ISCED_F"],
                    "summary": "disabled",
                    "validation": "semantic",
                    "minSimilarity": 0.5,
                },
            }
        )
        self.assertEqual(result.classification_suggestions.eqf_level.level, 3)
        self.assertEqual(result.classification_suggestions.eqf_level.reason, eqf_reason)
        self.assertEqual(result.classification_suggestions.isced_f[0].uri, ISCED_F_0611)
        payload = json.loads(self.model.ainvoke.call_args.args[0][1].content)["data"]
        self.assertEqual(payload["iscedFCandidates"][0]["code"], "0611")
        self.model.ainvoke.assert_awaited_once()
        self.assertEqual(self.collection.query.call_count, 2)

    async def test_extraction_isced_proposal_bypasses_semantic_similarity_floor(self):
        code = "0613"
        uri, labels = isced_f_vocabulary()[code]
        self.use_isced_candidates(
            [
                isced_f_record(
                    uri="http://data.europa.eu/snb/isced-f/0212",
                    code="0212",
                    distance=0.1,
                )
            ]
        )
        reason = "The course teaches digital application design."
        self.collection.get.return_value = {
            "documents": ["ISCED-F 0613: Software and applications development."],
            "metadatas": [
                {
                    "taxonomy": "ISCED_F",
                    "scheme": "http://data.europa.eu/snb/isced-f/25831c2",
                    "uri": uri,
                    "code": code,
                    "label_de": labels["de"],
                    "label_en": labels["en"],
                }
            ],
        }
        self.model.ainvoke.return_value = message(
            {
                "learningOutcome": [{"title": "Design a digital application."}],
                "classifications": {
                    "iscedFCode": code,
                    "iscedFReason": reason,
                },
            }
        )

        result = await self.recommend(
            {
                "sourceLanguage": "de",
                "course": {"description": "UX Design with a clickable Figma prototype."},
                "learningOutcome": [],
                "options": {
                    "validation": "semantic",
                    "extraction": "auto",
                    "summary": "disabled",
                    "classifications": ["ISCED_F"],
                },
            }
        )

        selected = result.classification_suggestions.isced_f[0]
        self.assertEqual(selected.code, code)
        self.assertEqual(selected.uri, uri)
        self.assertIsNone(selected.similarity_score)
        self.assertEqual(selected.reason, reason)
        self.collection.get.assert_called_once_with(
            where={"taxonomy": "ISCED_F", "code": code},
            include=["documents", "metadatas"],
        )

    async def test_extraction_and_validation_combine_classifications_and_skill_pairs(
        self,
    ):
        description = "Participants create and evaluate digital media projects."
        skill_rows = [
            record(
                uri=(
                    "http://data.europa.eu/esco/skill/00000000-0000-4000-8000-"
                    f"{index:012d}"
                ),
                title=f"Skill {index}",
            )
            for index in range(1, 35)
        ]
        self.collection.query.return_value = query_result([skill_rows])
        self.use_isced_candidates([isced_f_record(distance=0.15)])

        async def extract_and_validate(messages):
            payload = json.loads(messages[1].content)["data"]
            if "Identify intended learning outcomes" in messages[0].content:
                self.extraction_payload = payload
                return message(
                    {
                        "learningOutcome": [{"title": description}],
                        "classifications": {
                            "eqfLevel": 3,
                            "eqfReason": "The course practices independent software use.",
                            "iscedFCode": "0611",
                            "iscedFReason": "The course explicitly teaches software use.",
                        },
                    }
                )
            self.validation_payload = payload
            return message(
                {
                    "supported": [
                        {
                            "pairId": "p-c1",
                            "reason": "Directly supported by the course outcome.",
                        }
                    ],
                }
            )

        self.model.ainvoke.side_effect = extract_and_validate
        data = request_data()
        data["learningOutcome"] = []
        data["course"] = {"description": description}
        data["options"] = {
            "validation": "llm",
            "extraction": "auto",
            "summary": "disabled",
            "stats": "full",
            "classifications": ["EQF", "ISCED_F"],
        }

        result = await self.recommend(data)

        self.assertEqual(self.model.ainvoke.await_count, 2)
        self.assertEqual(
            [call.args[0] for call in self.factory.call_args_list],
            ["extraction", "validation"],
        )
        self.assertNotIn("requestedClassifications", self.extraction_payload)
        self.assertNotIn("requestedClassifications", self.validation_payload)
        self.assertNotIn("iscedFCandidates", self.validation_payload)
        self.assertEqual(set(self.validation_payload), {"course", "outcomes"})
        nested_candidates = [
            candidate
            for item in self.validation_payload["outcomes"]
            for candidate in item["skillCandidates"]
        ]
        self.assertEqual(len(nested_candidates), 34)
        isced_query = next(
            call
            for call in self.collection.query.call_args_list
            if call.kwargs.get("where") == {"taxonomy": "ISCED_F"}
        )
        self.assertEqual(isced_query.kwargs["n_results"], 5)
        self.assertEqual(result.classification_suggestions.eqf_level.level, 3)
        self.assertEqual(result.classification_suggestions.isced_f[0].code, "0611")
        self.assertEqual(result.stats.candidates_validated, 34)

    async def test_classification_failure_keeps_skill_and_outcome_results(self):
        self.use_isced_candidates([isced_f_record(distance=0.15)])

        async def fail_classification(messages):
            payload = json.loads(messages[1].content)["data"]
            if "iscedFCandidates" in payload:
                return message(
                    {
                        "iscedFCode": "9999",
                        "iscedFReason": "The course teaches this area.",
                    }
                )
            return message(
                {
                    "supported": [
                        {
                            "pairId": "p-c1",
                            "reason": "Directly supported by the outcome.",
                        }
                    ]
                }
            )

        self.model.ainvoke.side_effect = fail_classification
        data = request_data(
            options={
                "validation": "llm",
                "minSimilarity": 0.5,
                "extraction": "disabled",
                "summary": "disabled",
                "classifications": ["ISCED_F"],
            }
        )

        result = await self.recommend(data)

        self.assertEqual(len(result.learning_outcome), 1)
        self.assertEqual(
            result.learning_outcome[0].related_esco_skill[0].uri, ESCO_ANIMATION
        )
        self.assertEqual(result.classification_suggestions.isced_f, [])
        self.assertIn(
            "INVALID_CLASSIFICATION_OMITTED", [item.code for item in result.warnings]
        )

    async def test_accepted_only_validation_omits_rejected_pairs(self):
        rows = [
            record(
                f"http://data.europa.eu/esco/skill/00000000-0000-4000-8000-{index:012d}",
                f"Skill {index}",
            )
            for index in range(24)
        ]
        self.collection.query.return_value = query_result([rows] * 4)
        self.model.ainvoke.return_value = message(
            {
                "supported": [
                    {
                        "pairId": "p-c1",
                        "reason": "Direct evidence.",
                    }
                ]
            }
        )
        data = request_data(
            learningOutcome=[outcome(f"Outcome {index}") for index in range(4)]
        )
        data["options"].update(validation="llm", candidateLimit=8, stats="full")

        result = await self.recommend(data)

        self.model.ainvoke.assert_awaited_once()
        self.assertEqual(result.stats.candidates_validated, 96)
        self.assertEqual(
            sum(len(item.related_esco_skill) for item in result.learning_outcome), 1
        )

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
            supported = []
            for item in payload["outcomes"]:
                outcome_title = next(iter(item["title"].values()))
                for candidate in item["skillCandidates"]:
                    matches = (
                        "Animate" in outcome_title and candidate["title"] == "Animation"
                    ) or (
                        "Render" in outcome_title and candidate["title"] == "Rendering"
                    )
                    if matches:
                        supported.append(
                            {
                                "pairId": candidate["pairId"],
                                "reason": "Evidence-specific decision.",
                            }
                        )
            return message({"supported": supported})

        self.model.ainvoke.side_effect = validate
        result = await self.recommend(data)
        self.assertEqual(
            [
                [skill.uri for skill in item.related_esco_skill]
                for item in result.learning_outcome
            ],
            [[ESCO_ANIMATION], [ESCO_RENDERING]],
        )
        self.assertEqual(result.stats.candidates_validated, 4)

    async def test_deduplicates_before_validation_but_keeps_shared_skill_across_outcomes(
        self,
    ):
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
        self.embedding.embed_queries.assert_called_once_with(
            ["Animate with keyframes."]
        )
        self.assertEqual(len(result.learning_outcome), 2)
        self.assertTrue(
            all(item.related_esco_skill for item in result.learning_outcome)
        )

    async def test_separates_esco_subset_and_other_framework_and_preserves_urn(self):
        self.collection.query.side_effect = [
            query_result([[record()]]),
            query_result([[record(framework="DigCompESCO", distance=0.1)]]),
            query_result(
                [[record("urn:digcomp3.0:Competence/1.1", framework="DigComp")]]
            ),
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
        self.model.ainvoke.return_value = message(
            validation_response(
                [
                    {
                        "outcomeId": "lo-1",
                        "candidateId": "c1",
                        "decision": "uncertain",
                        "reason": "No teaching evidence.",
                    }
                ]
            )
        )
        result = await self.recommend(data)
        self.assertEqual(len(result.learning_outcome), 1)
        self.assertEqual(result.learning_outcome[0].related_esco_skill, [])
        self.assertIn("NO_SKILL_MATCH", [warning.code for warning in result.warnings])

    async def test_omits_unknown_or_duplicate_accepted_pair_ids(self):
        valid = {
            "outcomeId": "lo-1",
            "candidateId": "c1",
            "decision": "supported",
            "reason": "Evidence.",
        }
        for decisions, expected_skill_count in (
            ([{**valid, "candidateId": "invented"}], 0),
            ([valid, valid], 1),
        ):
            with self.subTest(decisions=decisions):
                self.model.ainvoke.return_value = message(
                    validation_response(decisions)
                )
                self.model.ainvoke.reset_mock()
                result = await self.recommend(self.use_llm())
                self.assertEqual(self.model.ainvoke.await_count, 2)
                self.assertEqual(
                    len(result.learning_outcome[0].related_esco_skill),
                    expected_skill_count,
                )
                self.assertIn(
                    "INVALID_SKILL_PAIRS_OMITTED",
                    [warning.code for warning in result.warnings],
                )

    async def test_extracts_verified_quotes_and_summary_in_one_call_without_skill_labels(
        self,
    ):
        data = {
            "course": {"description": "Wir vermitteln Animationen mit Keyframes."},
            "sourceLanguage": "de",
            "options": {
                "validation": "semantic",
                "minSimilarity": 0.5,
                "stats": "full",
            },
        }
        self.collection.query.return_value = query_result(
            [[record(title="Corpus-only label")]]
        )
        self.model.ainvoke.return_value = message(
            {
                "learningOutcome": [
                    {
                        "title": "Animationen mit Keyframes erstellen.",
                    }
                ],
                "learningOutcomeSummary": (
                    "Die Teilnehmenden erstellen Animationen mit Keyframes und treffen "
                    "grundlegende Gestaltungsentscheidungen."
                ),
            }
        )
        result = await self.recommend(data)
        self.model.ainvoke.assert_awaited_once()
        extraction_messages = self.model.ainvoke.call_args.args[0]
        self.assertNotIn("Corpus-only label", extraction_messages[0].content)
        self.assertNotIn("Corpus-only label", extraction_messages[1].content)
        self.assertIn("learningOutcomeSummary", extraction_messages[0].content)
        self.assertIn("Write in de.", extraction_messages[0].content)
        self.assertIn("Prefer 3–7 outcomes when supported", extraction_messages[0].content)
        self.assertTrue(
            extraction_messages[0].content.rstrip().endswith(
                "Use an empty outcome list only when the evidence supports no intended capability."
            )
        )
        self.assertIn("2–3 sentences", extraction_messages[0].content)
        extraction_request = json.loads(extraction_messages[1].content)
        self.assertEqual(set(extraction_request), {"data"})
        self.assertEqual(set(extraction_request["data"]), {"course.description"})
        self.assertNotIn("purpose", extraction_messages[1].content)
        self.assertNotIn("outputLanguage", extraction_messages[1].content)
        self.assertNotIn('"languages"', extraction_messages[1].content)
        self.assertEqual(
            [item.stage for item in result.stats.used_models],
            ["extraction"],
        )
        self.assertEqual(
            result.learning_outcome[0].title["de"],
            "Animationen mit Keyframes erstellen.",
        )
        self.assertEqual(
            result.learning_outcome_summary["de"],
            "Die Teilnehmenden erstellen Animationen mit Keyframes und treffen "
            "grundlegende Gestaltungsentscheidungen.",
        )

    async def test_extraction_omits_summary_when_disabled_and_skips_summary_call(self):
        data = {
            "course": {"description": "We teach animation using keyframes."},
            "sourceLanguage": "en",
            "options": {
                "validation": "semantic",
                "minSimilarity": 0.5,
                "summary": "disabled",
                "stats": "full",
            },
        }
        self.model.ainvoke.return_value = message(
            {
                "learningOutcome": [
                    {
                        "title": "Animate using keyframes.",
                    }
                ],
            }
        )
        result = await self.recommend(data)
        self.model.ainvoke.assert_awaited_once()
        system_prompt = self.model.ainvoke.call_args.args[0][0].content
        self.assertIn("no summary was requested", system_prompt)
        self.assertIsNone(result.learning_outcome_summary)
        self.assertEqual(result.stats.summary, "disabled")

    async def test_extraction_summary_falls_back_to_a_summary_call_when_omitted(self):
        data = {
            "course": {"description": "We teach animation using keyframes."},
            "sourceLanguage": "en",
            "options": {
                "validation": "semantic",
                "minSimilarity": 0.5,
                "stats": "full",
            },
        }
        self.model.ainvoke.side_effect = [
            message(
                {
                    "learningOutcome": [
                        {
                            "title": "Animate using keyframes.",
                        }
                    ]
                }
            ),
            message({"text": "Create animations using keyframes."}),
        ]
        result = await self.recommend(data)
        self.assertEqual(self.model.ainvoke.await_count, 2)
        self.assertEqual(
            [item.stage for item in result.stats.used_models],
            ["extraction", "summary"],
        )
        self.assertEqual(
            result.learning_outcome_summary["en"],
            "Create animations using keyframes.",
        )

    async def test_localizes_supplied_outcomes_and_summary_in_one_stage(self):
        data = request_data(
            sourceLanguage="de",
            targetLanguages=["de", "en-GB"],
            learningOutcome=[{"title": "3D-Animationen erstellen."}],
            learningOutcomeSummary="Grundlagen der 3D-Animation.",
        )
        self.model.ainvoke.return_value = message(
            {
                "learningOutcomes": [
                    {
                        "id": "lo-1",
                        "title": {
                            "de": "3D-Animationen erstellen.",
                            "en-GB": "Create 3D animations.",
                        },
                    }
                ],
                "learningOutcomeSummary": {
                    "de": "Grundlagen der 3D-Animation.",
                    "en-GB": "Learn the fundamentals of 3D animation.",
                },
            }
        )

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
        self.assertEqual(
            [item.stage for item in result.stats.used_models], ["translation"]
        )
        self.assertEqual(self.model.ainvoke.await_count, 1)
        self.assertEqual(self.embedding.embed_queries.call_count, 1)

    async def test_credential_generation_uses_neutral_purpose_and_localizes_generated_text(
        self,
    ):
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
            message(
                {
                    "learningOutcome": [
                        {
                            "title": "3D-Objekte mit Keyframes animieren.",
                        }
                    ],
                    "learningOutcomeSummary": "Fähigkeit, 3D-Objekte mit Keyframes zu animieren.",
                }
            ),
            message(
                {
                    "learningOutcomes": [
                        {
                            "id": "lo-1",
                            "title": {
                                "de": "3D-Objekte mit Keyframes animieren.",
                                "en-GB": "Animate 3D objects using keyframes.",
                            },
                        }
                    ],
                    "learningOutcomeSummary": {
                        "de": "Fähigkeit, 3D-Objekte mit Keyframes zu animieren.",
                        "en-GB": "Ability to animate 3D objects using keyframes.",
                    },
                }
            ),
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
        self.assertIn("capability represented by the credential", extraction_prompt)
        self.assertIn("any particular learner", extraction_prompt)
        extraction_data = json.loads(
            self.model.ainvoke.call_args_list[0].args[0][1].content
        )["data"]
        self.assertIn("course.targetGroup", extraction_data)
        self.assertIn("course.entryRequirement", extraction_data)
        self.assertEqual(
            [item.stage for item in result.stats.used_models],
            ["extraction", "translation"],
        )
        self.assertEqual(self.model.ainvoke.await_count, 2)

    async def test_extraction_uses_context_without_requesting_quotes(self):
        self.model.ainvoke.return_value = message({"learningOutcome": []})
        result = await self.recommend(
            {
                "sourceLanguage": "en",
                "course": {
                    "description": "A course about 3D animation.",
                    "targetGroup": "Beginners.",
                },
                "options": {"stats": "full"},
            }
        )
        self.assertEqual(result.learning_outcome, [])
        self.model.ainvoke.assert_awaited_once()
        messages = self.model.ainvoke.call_args.args[0]
        self.assertIn("targetGroup and entryRequirement as context only", messages[0].content)
        self.assertNotIn('"evidence"', messages[1].content)

    async def test_inferred_extraction_requires_and_returns_a_reason(self):
        self.model.ainvoke.side_effect = [
            message({"learningOutcome": [{"title": "Apply first aid."}]}),
            message(
                {
                    "learningOutcome": [
                        {
                            "title": "Apply first aid.",
                            "reason": "Inferred from the identified first-aid course curriculum.",
                        }
                    ]
                }
            ),
        ]
        result = await self.recommend(
            {
                "course": {"title": "First aid"},
                "sourceLanguage": "en",
                "options": {
                    "evidenceMode": "inferred",
                    "validation": "semantic",
                    "summary": "disabled",
                },
            }
        )
        self.assertEqual(self.model.ainvoke.await_count, 2)
        self.assertIn("Inferred", result.learning_outcome[0].reason)
        prompt = self.model.ainvoke.call_args.args[0][0].content
        self.assertIn(
            "Evidence policy: You may infer likely intended capabilities", prompt
        )
        self.assertIn("Omit weak guesses.", prompt)

    async def test_skill_validation_mode_does_not_change_extraction_request(self):
        candidate = {
            "uri": ISCED_F_0611,
            "code": "0611",
            "preferred_label": {"en": "Computer use"},
            "similarity_score": 0.8,
        }
        result = ClassifiedExtractionResult.model_validate(
            {
                "learningOutcome": [
                    {
                        "title": "Use digital tools.",
                        "reason": "The course describes practical digital work.",
                    }
                ],
                "classifications": {
                    "iscedFCode": "0611",
                    "iscedFReason": "The course focuses on computer use.",
                },
            }
        )
        extraction_requests = []

        for validation in ("semantic", "llm"):
            request = ElmLoqMetadataRequest.model_validate(
                request_data(
                    course={"description": "A course using digital tools."},
                    learningOutcome=[],
                    options={
                        "validation": validation,
                        "evidenceMode": "inferred",
                        "summary": "disabled",
                        "classifications": ["ISCED_F"],
                    },
                )
            )
            run = MetadataRecommendationRun(self.service, request)
            run.isced_candidates = [candidate]
            run.structured = AsyncMock(return_value=result)

            outcomes = await run.outcomes()

            self.assertEqual(len(outcomes), 1)
            extraction_requests.append(run.structured.await_args.args[:4])

        self.assertEqual(extraction_requests[0], extraction_requests[1])

    async def test_strict_and_inferred_validation_have_distinct_policies(self):
        for mode in ("strict", "inferred"):
            with self.subTest(mode=mode):
                self.model.ainvoke.return_value = message({"supported": []})
                data = self.use_llm()
                data["options"]["evidenceMode"] = mode
                await self.recommend(data)
                prompt = self.model.ainvoke.call_args.args[0][0].content
                expected_policy = (
                    "Strict evidence policy: Do not require formal outcome phrasing"
                    if mode == "strict"
                    else "Evidence policy: You may infer likely intended capabilities"
                )
                self.assertIn(expected_policy, prompt)
                self.assertIn("Copy pairId exactly", prompt)
                self.assertIn("skillCandidates", prompt)
                self.assertIn("source language en", prompt)

    async def test_semantic_and_local_framework_choices_do_not_emit_information_warnings(
        self,
    ):
        self.collection.query.return_value = query_result(
            [
                [
                    record(
                        "urn:stifterverband:futureskills2030:Competence/1.4",
                        framework="FutureSkills2030",
                    ),
                ]
            ]
        )
        data = request_data()
        data["options"]["frameworks"] = ["FutureSkills2030"]
        result = await self.recommend(data)
        self.assertEqual(result.warnings, [])

    async def test_accepted_links_require_a_nonempty_reason(self):
        for reason in (None, "", "   "):
            with self.subTest(reason=reason):
                self.model.ainvoke.return_value = message(
                    {
                        "supported": [
                            {
                                "outcomeId": "lo-1",
                                "candidateId": "c1",
                                "reason": reason,
                            }
                        ]
                    }
                )
                with self.assertRaises(RecommendationError):
                    await self.recommend(self.use_llm())

    async def test_title_only_extraction_returns_no_origin_or_evidence(self):
        self.model.ainvoke.return_value = message(
            {
                "learningOutcome": [
                    {
                        "title": "Animate objects.",
                    }
                ]
            }
        )
        result = await self.recommend(
            {
                "sourceLanguage": "en",
                "course": {"description": "We teach keyframes."},
                "options": {"validation": "semantic", "summary": "disabled"},
            }
        )
        body = result.model_dump(by_alias=True)
        self.assertNotIn("learningOutcomeSummaryOrigin", body)
        self.assertNotIn("origin", body["learningOutcome"][0])
        self.assertNotIn("evidence", body["learningOutcome"][0])
        self.model.ainvoke.assert_awaited_once()

    async def test_empty_extraction_skips_retrieval_and_summary(self):
        self.model.ainvoke.return_value = message({"learningOutcome": []})
        result = await self.recommend(
            {
                "sourceLanguage": "en",
                "course": {"description": "Room 12, Tuesday."},
                "options": {"stats": "full"},
            }
        )
        self.assertEqual(result.learning_outcome, [])
        self.assertIsNone(result.learning_outcome_summary)
        self.assertEqual(result.stats.summary, "no_outcomes")
        self.assertEqual(result.warnings[0].code, "NO_SUPPORTED_LEARNING_OUTCOMES")
        self.collection.query.assert_not_called()
        self.assertEqual(self.model.ainvoke.await_count, 1)

    async def test_valid_empty_classified_extraction_is_not_repaired(self):
        self.model.ainvoke.return_value = message(
            {"learningOutcome": [], "classifications": {}}
        )
        result = await self.recommend(
            {
                "sourceLanguage": "en",
                "course": {"description": "Room 12, Tuesday."},
                "options": {"classifications": ["EQF"]},
            }
        )
        self.assertEqual(result.learning_outcome, [])
        self.assertEqual(result.warnings[0].code, "NO_SUPPORTED_LEARNING_OUTCOMES")
        self.assertEqual(self.model.ainvoke.await_count, 1)

    async def test_recovers_once_from_invalid_schema_using_original_evidence(self):
        self.model.ainvoke.side_effect = [
            AIMessage(content="not JSON"),
            message({"learningOutcome": []}),
        ]
        result = await self.recommend(
            {
                "sourceLanguage": "en",
                "course": {"description": "On campus Tuesday."},
            }
        )
        self.assertEqual(result.learning_outcome, [])
        self.assertEqual(self.model.ainvoke.await_count, 2)

    async def test_accepts_fully_fenced_json_without_spending_a_repair_call(self):
        self.model.ainvoke.return_value = AIMessage(
            content='```json\n{"learningOutcome": []}\n```'
        )
        result = await self.recommend(
            {
                "sourceLanguage": "en",
                "course": {"description": "On campus Tuesday."},
            }
        )
        self.assertEqual(result.learning_outcome, [])
        self.assertEqual(self.model.ainvoke.await_count, 1)

    async def test_schema_or_provider_failure_does_not_log_course_content(self):
        text = "Private course marker never to be logged"
        self.model.ainvoke.return_value = AIMessage(content=text)
        with self.assertLogs(
            "src.models.ElmLoqMetadataRecommender", level="WARNING"
        ) as logs:
            with self.assertRaises(RecommendationError):
                await self.recommend(
                    {
                        "sourceLanguage": "en",
                        "course": {"description": text},
                    }
                )
        self.assertNotIn(text, "\n".join(logs.output))

    async def test_invalid_extracted_summary_is_not_a_successful_partial_response(self):
        data = request_data()
        data["options"]["summary"] = "auto"
        self.model.ainvoke.return_value = message(
            {
                "learningOutcome": [
                    {
                        "title": "Animate with keyframes.",
                    }
                ],
                "learningOutcomeSummary": "",
            }
        )
        with self.assertRaises(RecommendationError):
            await self.recommend(data)

    async def test_balanced_validation_is_the_default_and_fits_96_pairs_in_one_call(
        self,
    ):
        rows = [
            record(
                f"http://data.europa.eu/esco/skill/00000000-0000-4000-8000-{index:012d}",
                f"Skill {index}",
            )
            for index in range(24)
        ]
        self.collection.query.return_value = query_result([rows] * 4)
        self.model.ainvoke.return_value = message(
            validation_response(
                [
                    {
                        "outcomeId": f"lo-{index + 1}",
                        "candidateId": f"c{index * 24 + 1}",
                        "decision": "supported",
                        "reason": "Direct evidence.",
                    }
                    for index in range(4)
                ]
                + [
                    {
                        "outcomeId": pair_outcome,
                        "candidateId": pair_candidate,
                        "decision": "unsupported",
                    }
                    for pair_outcome, pair_candidate in [
                        (f"lo-{index + 1}", f"c{index * 24 + offset}")
                        for index in range(4)
                        for offset in range(2, 25)
                    ]
                ]
            )
        )
        data = request_data(
            learningOutcome=[outcome(f"Outcome {index}") for index in range(4)]
        )
        data["options"].update(validation="llm", candidateLimit=8, stats="full")

        result = await self.recommend(data)

        self.model.ainvoke.assert_awaited_once()
        payload = json.loads(self.model.ainvoke.call_args.args[0][1].content)["data"]
        nested_candidates = [
            candidate
            for item in payload["outcomes"]
            for candidate in item["skillCandidates"]
        ]
        self.assertEqual(len(nested_candidates), 96)
        self.assertTrue(
            all("definition" in candidate for candidate in nested_candidates)
        )
        self.assertTrue(
            all(len(candidate["definition"]) <= 600 for candidate in nested_candidates)
        )
        self.assertNotIn("evidence", payload["outcomes"][0])
        self.assertEqual(result.stats.candidates_validated, 96)
        self.assertEqual(
            sum(len(item.related_esco_skill) for item in result.learning_outcome), 4
        )

    async def test_request_state_is_not_shared_between_recommendations(self):
        first, second = await asyncio.gather(self.recommend(), self.recommend())
        self.assertIsNot(first.learning_outcome[0], second.learning_outcome[0])
        first.learning_outcome[0].related_esco_skill.clear()
        self.assertEqual(len(second.learning_outcome[0].related_esco_skill), 1)

    async def test_similarity_floor_filters_without_capping_accepted_skills(self):
        self.collection.query.return_value = query_result(
            [
                [
                    record(distance=0.6),
                    record(ESCO_RENDERING, "Rendering", distance=0.2),
                    record(distance=0.1),
                ]
            ]
        )
        data = request_data()
        data["options"]["candidateLimit"] = 1
        result = await self.recommend(data)
        self.assertEqual(
            [skill.uri for skill in result.learning_outcome[0].related_esco_skill],
            [ESCO_ANIMATION, ESCO_RENDERING],
        )
        self.assertEqual(result.stats.candidates_retrieved, 3)

    async def test_default_similarity_floor_filters_candidates_before_validation(self):
        self.collection.query.return_value = query_result(
            [
                [
                    record(distance=0.61),
                    record(ESCO_RENDERING, "Rendering", distance=0.6),
                    record(
                        "http://data.europa.eu/esco/skill/00000000-0000-4000-8000-000000000003",
                        "Prototyping",
                        distance=0.59,
                    ),
                ]
            ]
        )
        data = request_data()
        data["options"].pop("minSimilarity")
        data["options"]["validation"] = "llm"
        self.model.ainvoke.return_value = message(
            validation_response(
                [
                    {
                        "outcomeId": "lo-1",
                        "candidateId": "c1",
                        "decision": "supported",
                        "reason": "Direct evidence.",
                    },
                    {
                        "outcomeId": "lo-1",
                        "candidateId": "c2",
                        "decision": "supported",
                        "reason": "Direct evidence.",
                    },
                ]
            )
        )

        result = await self.recommend(data)

        self.assertEqual(result.stats.candidates_retrieved, 3)
        self.assertEqual(result.stats.candidates_validated, 2)
        payload = json.loads(self.model.ainvoke.call_args.args[0][1].content)["data"]
        self.assertEqual(
            sum(len(item["skillCandidates"]) for item in payload["outcomes"]), 2
        )
        self.assertEqual(
            [skill.title for skill in result.learning_outcome[0].related_esco_skill],
            ["Prototyping", "Rendering"],
        )
        self.assertAlmostEqual(
            ElmLoqMetadataRequest.model_validate(data).options.effective_min_similarity,
            0.4,
        )

    async def test_shared_skills_remain_attached_to_each_outcome_without_a_cap(self):
        self.collection.query.side_effect = [
            query_result(
                [
                    [record(), record(ESCO_RENDERING, "Rendering", distance=0.1)],
                    [record(distance=0.1), record(ESCO_RENDERING, "Rendering")],
                ]
            ),
            query_result(
                [
                    [record("urn:test:digital-one", framework="DigComp")],
                    [record("urn:test:digital-two", framework="DigComp")],
                ]
            ),
        ]
        data = request_data(learningOutcome=[outcome(), outcome("Use rigs.")])
        data["options"].update(frameworks=["ESCO", "DigComp"], candidateLimit=1)
        result = await self.recommend(data)
        self.assertEqual(
            [
                (len(item.related_esco_skill), len(item.related_skill))
                for item in result.learning_outcome
            ],
            [(2, 1), (2, 1)],
        )
        self.assertEqual(
            result.learning_outcome[0].related_esco_skill[0].uri, ESCO_RENDERING
        )
        self.assertEqual(
            result.learning_outcome[1].related_esco_skill[0].uri, ESCO_ANIMATION
        )

    async def test_malformed_embeddings_and_records_are_not_empty_successes(self):
        for vectors in ([], [[0, 0]], [[float("nan"), 1]], [[True, 1]]):
            with self.subTest(vectors=vectors):
                self.embedding.embed_queries.side_effect = None
                self.embedding.embed_queries.return_value = vectors
                with self.assertRaises(RecommendationError):
                    await self.recommend()
        self.embedding.embed_queries.return_value = [[1, 0]]
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
        self.embedding.embed_queries.side_effect = requests.Timeout()
        with self.assertRaises(RecommendationTimeout):
            await self.recommend()

    async def test_total_deadline_and_stage_timeout_are_not_successes(self):
        async def slow_model(messages):
            await asyncio.sleep(1)
            return message({"learningOutcome": []})

        self.model.ainvoke.side_effect = slow_model
        self.service.timeout_seconds = 0.02
        with self.assertRaises(RecommendationTimeout):
            await self.recommend(
                {
                    "sourceLanguage": "en",
                    "course": {"description": "A course."},
                }
            )

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
            frameworks=["ESCO", "GRETA"], candidateLimit=24, summary="auto"
        )
        with self.assertRaises(RecommendationInputError):
            await self.recommend(data)
        self.embedding.embed_queries.assert_not_called()
        self.embedding.embed_documents.assert_not_called()
        self.factory.assert_not_called()

    async def test_native_chroma_batch_query_preserves_cosine_scores_and_framework_filters(
        self,
    ):
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
                    record("urn:digcomp3.0:Competence/1.1", framework="DigComp")[
                        0
                    ].metadata,
                ],
            )
            data = request_data()
            data["options"]["frameworks"] = ["ESCO", "DigComp"]
            service = ElmLoqMetadataRecommender(
                self.embedding, collection, self.factory
            )
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
            {"sourceLanguage": "en", "course": {"title": "First aid"}},
            request_data(learningOutcome=[{"id": "client-id", **outcome()}]),
            request_data(language="en"),
            request_data(options={"frameworks": ["ESCO", "ESCO"]}),
            request_data(options={"frameworks": ["unknown"]}),
            request_data(options={"candidateLimit": True}),
            request_data(options={"candidateLimit": 0}),
            request_data(options={"candidateLimit": 49}),
            request_data(options={"maxSkillsPerFramework": 8}),
            request_data(options={"validationDetail": "balanced"}),
            request_data(options={"validationDetail": "compact"}),
            request_data(options={"validationDetail": "explained"}),
            request_data(options={"maxIscedFCandidates": 5}),
            request_data(options={"evidenceMode": "unknown"}),
            request_data(learningOutcome=[{"title": {"en": "Localized input"}}]),
            request_data(learningOutcomeSummary={"en": "Localized summary"}),
            request_data(course={"title": {"en": "Localized title"}}),
            request_data(
                course={"learningOutcomeSummary": {"en": "Localized summary"}}
            ),
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
                response = self.client.post(
                    "/recommend/elm-loq-metadata", json=request_data()
                )
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json(), {"detail": str(error)})
                self.assertIn("X-Request-ID", response.headers)

    def test_http_response_preserves_outcome_associations_and_exact_elm_aliases(self):
        collection = Mock()
        collection.metadata = {"hnsw:space": "cosine"}
        collection.query.return_value = query_result([[record()]])
        embedding = Mock()
        embedding.embed_documents.return_value = [[1.0, 0.0]]
        embedding.embed_queries.return_value = [[1.0, 0.0]]
        service = ElmLoqMetadataRecommender(embedding, collection)
        self.app.dependency_overrides[get_recommender] = lambda: service
        response = self.client.post("/recommend/elm-loq-metadata", json=request_data())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("skills", body)
        item = body["learningOutcome"][0]
        self.assertNotIn("id", item)
        self.assertEqual(item["title"], {"en": "Animate with keyframes."})
        self.assertEqual(item["relatedSkill"], [])
        self.assertEqual(item["relatedESCOSkill"][0]["uri"], ESCO_ANIMATION)
        self.assertAlmostEqual(item["relatedESCOSkill"][0]["similarityScore"], 0.8)
        self.assertIsNone(body["learningOutcomeSummary"])
        self.assertIn("X-Request-ID", response.headers)
        openapi = self.client.get("/openapi.json").json()
        operation = openapi["paths"]["/recommend/elm-loq-metadata"]["post"]
        request_examples = operation["requestBody"]["content"]["application/json"][
            "examples"
        ]
        for item in request_examples.values():
            ElmLoqMetadataRequest.model_validate(item["value"])
        response_example = operation["responses"]["200"]["content"]["application/json"][
            "example"
        ]
        from src.models.elm_loq_metadata import ElmLoqMetadataResponse

        ElmLoqMetadataResponse.model_validate(response_example)
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
            openapi["components"]["schemas"]["LearningOutcomeRecommendation"][
                "properties"
            ],
        )

    def test_stats_are_omitted_or_reduced_according_to_request(self):
        collection = Mock()
        collection.metadata = {"hnsw:space": "cosine"}
        collection.query.return_value = query_result([[record()]])
        embedding = Mock()
        embedding.embed_documents.return_value = [[1.0, 0.0]]
        embedding.embed_queries.return_value = [[1.0, 0.0]]
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
        embedding.embed_queries.return_value = [[1.0, 0.0]]
        model = Mock()
        model.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="invalid JSON",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            )
        )
        service = ElmLoqMetadataRecommender(
            embedding, collection, lambda stage: (model, "test-model")
        )
        self.app.dependency_overrides[get_recommender] = lambda: service
        data = request_data()
        data["options"]["validation"] = "llm"
        with patch(
            "src.models.ElmLoqMetadataRecommender.create_agent",
            side_effect=fake_agent,
        ):
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
            "ESCO",
            "DigCompESCO",
            "GRETA",
            "DigComp",
            "DigCompEdu",
            "FutureSkills2030",
        ):
            with self.subTest(framework=framework):
                document, _ = record("urn:test:skill", "Expected title", framework)
                prediction = retriever.create_prediction((document, 0.8))
                self.assertEqual(prediction.uri, "urn:test:skill")
                self.assertEqual(prediction.title, "Expected title")
                self.assertEqual(prediction.score, 0.8)
                self.assertEqual(prediction.taxonomy, framework)
                self.assertEqual(
                    prediction, SkillPrediction.from_document((document, 0.8))
                )

    def test_embedding_transport_keeps_payload_and_formats_with_finite_timeout(self):
        embedding = HuggingFaceInferenceEmbeddings(
            "test-token", "http://embedding.invalid"
        )
        for payload in ([[1.0, 0.5]], [{"embedding": [1.0, 0.5]}]):
            with self.subTest(payload=payload), patch(
                "src.embeddings.requests.post"
            ) as post:
                post.return_value.json.return_value = payload
                self.assertEqual(
                    embedding.embed_documents(["An outcome"]), [[1.0, 0.5]]
                )
                self.assertEqual(
                    post.call_args.kwargs["json"], {"inputs": ["passage: An outcome"]}
                )
                self.assertEqual(post.call_args.kwargs["timeout"], (5, 30))
                post.return_value.raise_for_status.assert_called_once()

    def test_embedding_prefixes_are_applied_only_by_the_root_calls(self):
        embedding = HuggingFaceInferenceEmbeddings(
            "test-token", "http://embedding.invalid", "query: ", "passage: "
        )
        with patch("src.embeddings.requests.post") as post:
            post.return_value.json.return_value = [[1.0, 0.5]]
            embedding.embed_query("A query")
            self.assertEqual(
                post.call_args.kwargs["json"], {"inputs": ["query: A query"]}
            )
            embedding.embed_queries(["First", "Second"])
            self.assertEqual(
                post.call_args.kwargs["json"],
                {"inputs": ["query: First", "query: Second"]},
            )
            embedding.embed_documents(["A document"])
            self.assertEqual(
                post.call_args.kwargs["json"], {"inputs": ["passage: A document"]}
            )

    def test_embedding_prefixes_are_configurable_with_empty_values(self):
        with patch.dict(
            "os.environ",
            {
                "HF_EMBEDDING_API_KEY": "test-token",
                "HF_EMBEDDING_HOST": "http://embedding.invalid",
                "HF_EMBEDDING_QUERY_PREFIX": "",
                "HF_EMBEDDING_PASSAGE_PREFIX": "document: ",
            },
            clear=True,
        ):
            embedding = load_embedding_function()

        self.assertEqual(embedding.query_prefix, "")
        self.assertEqual(embedding.passage_prefix, "document: ")
