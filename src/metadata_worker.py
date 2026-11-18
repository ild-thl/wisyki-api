import asyncio
import logging
import os
import socket
from uuid import UUID

from src.embeddings import load_embedding_function
from src.models.ElmLoqMetadataRecommender import (
    ElmLoqMetadataRecommender,
    RecommendationError,
)
from src.models.elm_loq_metadata import ElmLoqMetadataRequest
from src.models.metadata_job_store import MetadataJobStore
from src.setup import connect_skill_collection

logger = logging.getLogger("metadata-worker")


async def process_job(
    job: dict,
    recommender: ElmLoqMetadataRecommender,
    store: MetadataJobStore,
) -> None:
    job_id = UUID(str(job["id"]))

    async def report_progress(stage: str, progress: int) -> None:
        await asyncio.to_thread(store.update_progress, job_id, stage, progress)

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(max(5, store.lease_seconds // 3))
            await asyncio.to_thread(store.heartbeat, job_id)

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        request = ElmLoqMetadataRequest.model_validate(job["request_data"])
        result = await recommender.recommend(request, report_progress)
        payload = result.model_dump(
            mode="json",
            by_alias=True,
            exclude={"stats"} if result.stats is None else None,
        )
        await asyncio.to_thread(
            store.complete_job, job_id, payload
        )
    except RecommendationError as error:
        payload = {"detail": str(error)}
        if error.stats is not None:
            payload["stats"] = error.stats.model_dump(by_alias=True)
        await asyncio.to_thread(store.fail_job, job_id, payload)
        logger.warning(
            "Metadata job failed: job_id=%s error_type=%s",
            job_id,
            type(error).__name__,
        )
    except Exception as error:
        await asyncio.to_thread(
            store.fail_job, job_id, {"detail": "Recommendation job failed."}
        )
        logger.error(
            "Metadata job crashed: job_id=%s error_type=%s",
            job_id,
            type(error).__name__,
        )
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)


async def run_worker() -> None:
    store = MetadataJobStore.from_env()
    await asyncio.to_thread(store.initialize)
    embedding = load_embedding_function()
    _, collection = connect_skill_collection()
    concurrency = int(os.getenv("ELM_LOQ_METADATA_WORKER_CONCURRENCY", "1"))
    timeout_seconds = int(os.getenv("ELM_LOQ_METADATA_JOB_TIMEOUT_SECONDS", "3600"))
    if not 1 <= concurrency <= 16:
        raise ValueError("ELM_LOQ_METADATA_WORKER_CONCURRENCY must be between 1 and 16.")
    if not 120 <= timeout_seconds <= 86400:
        raise ValueError("ELM_LOQ_METADATA_JOB_TIMEOUT_SECONDS must be between 120 and 86400.")
    recommender = ElmLoqMetadataRecommender(
        embedding,
        collection,
        timeout_seconds=timeout_seconds,
        metadata_store=store,
        rate_wait_seconds=timeout_seconds,
    )
    active: set[asyncio.Task] = set()
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    last_cleanup = 0.0
    while True:
        done = {task for task in active if task.done()}
        active.difference_update(done)
        for task in done:
            try:
                task.result()
            except Exception as error:
                logger.error("Metadata worker task failed: error_type=%s", type(error).__name__)
        if len(active) < concurrency:
            job = await asyncio.to_thread(store.claim_job, worker_id)
            if job is not None:
                active.add(asyncio.create_task(process_job(job, recommender, store)))
                continue
        now = asyncio.get_running_loop().time()
        if now - last_cleanup >= 60:
            await asyncio.to_thread(store.cleanup_expired)
            last_cleanup = now
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    asyncio.run(run_worker())
