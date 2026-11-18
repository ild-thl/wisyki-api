import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.models.SkillPrediction import SkillPrediction
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
