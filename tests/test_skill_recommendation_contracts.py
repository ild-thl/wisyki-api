import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.models.SkillPrediction import SkillPrediction
from src.models.KeywordExtractor import KeywordExtractor
from src.routes import skill_router


class FakeRetriever:
    def __init__(self, embedding, skilldb, domains, request):
        self.used_models = []

    async def predict(self, target="learning_outcomes", get_sources=False):
        return ["An outcome"], [
            SkillPrediction(
                uri="urn:test:skill",
                title="A skill",
                score=0.75,
                taxonomy="ESCO",
            )
        ]


class SkillRecommendationContractTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(skill_router.router)
        app.dependency_overrides.update(
            {
                skill_router.get_db: lambda: object(),
                skill_router.get_embedding_function: lambda: SimpleNamespace(),
                skill_router.get_skilldb: lambda: object(),
                skill_router.get_domains: lambda: set(),
            }
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.retriever = patch.object(skill_router, "SkillRetriever", FakeRetriever)
        self.retriever.start()
        self.addCleanup(self.retriever.stop)
        self.keywords = patch.object(skill_router, "KeywordExtractor")
        self.keywords.start().return_value.used_models = []
        self.addCleanup(self.keywords.stop)

    def test_legacy_preserves_shape_and_inverted_score(self):
        response = self.client.post("/chatsearch", json={"doc": "A course"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "searchterms": ["An outcome"],
                "results": [
                    {
                        "title": "A skill",
                        "uri": "urn:test:skill",
                        "score": 0.25,
                        "metadata": None,
                        "taxonomy": "ESCO",
                        "source": None,
                    }
                ],
                "stats": {"used_models": []},
            },
        )

    def test_v2_preserves_shape_and_similarity_score(self):
        response = self.client.post(
            "/v2/chatsearch",
            json={
                "doc": "A course",
                "los": ["An outcome"],
                "use_llm": False,
                "llm_validation": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "learning_outcomes": {
                    "natural": ["An outcome"],
                    "skills": [
                        {
                            "title": "A skill",
                            "uri": "urn:test:skill",
                            "score": 0.75,
                            "metadata": None,
                            "taxonomy": "ESCO",
                            "source": None,
                        }
                    ],
                    "comp_level": None,
                },
                "prerequisites": None,
                "keywords": None,
                "stats": {"used_models": []},
            },
        )

    def test_embedding_endpoints_hide_prefix_options_and_pass_plain_text_to_wrapper(
        self,
    ):
        schemas = self.client.get("/openapi.json").json()["components"]["schemas"]
        self.assertEqual(
            set(schemas["GetEmbeddingsQueryRequest"]["properties"]),
            {"query", "query_instruction"},
        )
        self.assertEqual(
            set(schemas["GetEmbeddingsDocumentsRequest"]["properties"]),
            {"docs", "embed_instruction"},
        )
        self.assertTrue(
            schemas["GetEmbeddingsQueryRequest"]["properties"]["query_instruction"][
                "deprecated"
            ]
        )
        self.assertTrue(
            schemas["GetEmbeddingsDocumentsRequest"]["properties"]["embed_instruction"][
                "deprecated"
            ]
        )

        embedding = Mock()
        embedding.embed_query.return_value = [0.1]
        embedding.embed_documents.return_value = [[0.2]]
        self.client.app.dependency_overrides[skill_router.get_embedding_function] = (
            lambda: embedding
        )
        response = self.client.post(
            "/embeddings/query",
            json={"query": "plain query", "query_instruction": "caller prefix: "},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [0.1])
        embedding.embed_query.assert_called_once_with("plain query")

        response = self.client.post(
            "/embeddings/documents",
            json={"docs": ["plain document"], "embed_instruction": "caller prefix: "},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [[0.2]])
        embedding.embed_documents.assert_called_once_with(["plain document"])

    def test_v2_empty_predictions_remain_successful(self):
        with patch.object(FakeRetriever, "predict", return_value=([], [])):
            response = self.client.post("/v2/chatsearch", json={"doc": "A course"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["learning_outcomes"],
            {"natural": [], "skills": [], "comp_level": None},
        )

    def test_descriptive_aliases_preserve_request_and_response_contracts(self):
        for original, alias in (
            ("/chatsearch", "/v1/skill-recommendations"),
            ("/v2/chatsearch", "/v2/skill-recommendations"),
        ):
            with self.subTest(alias=alias):
                payload = {"doc": "A course"}
                response = self.client.post(alias, json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json(), self.client.post(original, json=payload).json()
                )


class KeywordExtractorTests(unittest.TestCase):
    @patch("src.models.KeywordExtractor.get_llm")
    def test_extract_passes_document_and_candidates_to_llm(self, get_llm):
        prompt_texts = []

        def respond(prompt_value):
            prompt_texts.append(prompt_value.to_string())
            return AIMessage(
                content="<keywords>data literacy, learning design</keywords>"
            )

        get_llm.return_value = (RunnableLambda(respond), "test-model")
        request = SimpleNamespace(openai_api_key="key", mistral_api_key=None)
        extractor = KeywordExtractor(request)

        result = extractor.extract("Course document", ["data", "design"])

        self.assertEqual(result, ["data literacy", "learning design"])
        self.assertEqual(len(prompt_texts), 1)
        self.assertIn("Course document", prompt_texts[0])
        self.assertIn("data, design", prompt_texts[0])
