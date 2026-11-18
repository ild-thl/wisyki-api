import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

from src.models.metadata_job_store import (
    JobQueueFull,
    MetadataJobStore,
    ProviderWaitExceeded,
)
from src.models.elm_loq_metadata import (
    ElmLoqMetadataResponse,
)
from src.models.metadata_llm import metadata_provider_bucket
from src.models.ElmLoqMetadataRecommender import retry_delay
from src.metadata_worker import process_job
from src.routes.metadata_router import get_metadata_store, get_recommender, router
from test_elm_loq_metadata import request_data


JOB_ID = "f4102293-3f18-4f66-8b64-207dcb9ab357"
JOB_TOKEN = "private-test-job-token"


class FakeJobStore:
    def __init__(self):
        self.try_acquire_sync_slot = Mock(return_value=True)
        self.release_sync_slot = Mock()
        self.create_job = Mock(return_value=(JOB_ID, JOB_TOKEN, {}))
        self.get_job = Mock(return_value={
            "id": JOB_ID,
            "status": "running",
            "stage": "validating_results",
            "progress": 65,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:01+00:00",
            "completed_at": None,
            "result": None,
            "error": None,
        })
        self.get_events = Mock(return_value=[{
            "id": 7,
            "event_type": "completed",
            "payload": {"status": "succeeded", "stage": "completed", "progress": 100},
        }])


class MetadataJobHttpTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(router)
        self.store = FakeJobStore()
        self.recommender = Mock()
        self.recommender.recommend = AsyncMock()
        self.app.dependency_overrides[get_metadata_store] = lambda: self.store
        self.app.dependency_overrides[get_recommender] = lambda: self.recommender
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def test_async_preference_returns_capability_token_and_does_not_run_inline(self):
        response = self.client.post(
            "/recommend/elm-loq-metadata",
            json=request_data(),
            headers={"Prefer": "respond-async"},
        )
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body["jobId"], JOB_ID)
        self.assertEqual(body["jobToken"], JOB_TOKEN)
        self.assertEqual(response.headers["X-Job-Token"], JOB_TOKEN)
        self.assertEqual(response.headers["Location"], body["statusUrl"])
        self.store.create_job.assert_called_once()
        self.recommender.recommend.assert_not_called()

    def test_status_and_sse_require_the_job_token(self):
        url = f"/recommend/elm-loq-metadata/jobs/{JOB_ID}"
        unauthorized = self.client.get(url)
        self.assertEqual(unauthorized.status_code, 404)
        status = self.client.get(url, headers={"X-Job-Token": JOB_TOKEN})
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["stage"], "validating_results")

        events = self.client.get(
            f"{url}/events", headers={"X-Job-Token": JOB_TOKEN}
        )
        self.assertEqual(events.status_code, 200)
        self.assertIn("event: completed", events.text)
        self.assertIn('"progress":100', events.text)

    def test_full_job_queue_returns_retryable_overload(self):
        self.store.create_job.side_effect = JobQueueFull()
        response = self.client.post(
            "/recommend/elm-loq-metadata",
            json=request_data(),
            headers={"Prefer": "respond-async"},
        )
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "5")

    def test_synchronous_capacity_returns_retryable_overload(self):
        self.store.try_acquire_sync_slot.return_value = False
        response = self.client.post(
            "/recommend/elm-loq-metadata", json=request_data()
        )
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "5")
        self.recommender.recommend.assert_not_called()

    def test_synchronous_request_remains_the_default(self):
        self.recommender.recommend.return_value = JSONResponse({"result": "ok"})
        response = self.client.post(
            "/recommend/elm-loq-metadata", json=request_data()
        )
        self.assertEqual(response.status_code, 200)
        self.recommender.recommend.assert_awaited_once()
        self.store.create_job.assert_not_called()


class SharedQuotaTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_shared_quota_and_reports_wait_time(self):
        store = MetadataJobStore()
        with patch.object(
            store, "try_reserve_provider_call", side_effect=[0.02, 0]
        ), patch("src.models.metadata_job_store.asyncio.sleep", new=AsyncMock()):
            wait_ms = await store.acquire_provider_call("bucket", 2, 10, 1)
        self.assertGreaterEqual(wait_ms, 0)

    async def test_fails_when_quota_wait_exceeds_synchronous_budget(self):
        store = MetadataJobStore()
        with patch.object(store, "try_reserve_provider_call", return_value=10):
            with self.assertRaises(ProviderWaitExceeded):
                await store.acquire_provider_call("bucket", 2, 10, 1)

    def test_provider_bucket_is_secret_safe_and_shared_by_stages(self):
        settings = {
            "LLM_API_URL": "https://provider.invalid/v1",
            "LLM_API_KEY": "do-not-store-this",
        }
        with patch.dict("os.environ", settings, clear=True):
            extraction = metadata_provider_bucket("extraction")
            validation = metadata_provider_bucket("validation")
        self.assertEqual(extraction, validation)
        self.assertNotIn(settings["LLM_API_KEY"], extraction)

    def test_rate_limit_retry_honors_provider_retry_after(self):
        error = SimpleNamespace(
            response=SimpleNamespace(headers={"retry-after": "4.5"})
        )
        self.assertEqual(retry_delay(error), 4.5)
        self.assertEqual(retry_delay(SimpleNamespace()), 1)


class MetadataWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_job_persists_result_and_reports_stage_progress(self):
        response = ElmLoqMetadataResponse(
            learning_outcome=[],
            learning_outcome_summary=None,
            warnings=[],
            stats=None,
        )
        recommender = Mock()
        recommender.recommend = AsyncMock(return_value=response)
        store = Mock()
        job = {
            "id": JOB_ID,
            "request_data": request_data(),
        }
        await process_job(job, recommender, store)
        recommender.recommend.assert_awaited_once()
        progress = recommender.recommend.await_args.args[1]
        await progress("validating_results", 65)
        store.update_progress.assert_called_once_with(
            UUID(JOB_ID), "validating_results", 65
        )
        store.complete_job.assert_called_once()
        self.assertEqual(
            store.complete_job.call_args.args[0], UUID(JOB_ID)
        )
        self.assertNotIn("stats", store.complete_job.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
