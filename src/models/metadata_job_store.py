import asyncio
import hashlib
import hmac
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

import psycopg2
from psycopg2.extras import Json, RealDictCursor


class JobQueueFull(Exception):
    pass


class ProviderWaitExceeded(Exception):
    def __init__(self, wait_seconds: float):
        self.wait_seconds = wait_seconds


class MetadataJobStore:
    def __init__(
        self,
        max_pending_jobs: int = 100,
        retention_seconds: int = 86400,
        lease_seconds: int = 180,
        max_sync_requests: int = 20,
        sync_slot_lease_seconds: int = 180,
    ):
        self.max_pending_jobs = max_pending_jobs
        self.retention_seconds = retention_seconds
        self.lease_seconds = lease_seconds
        self.max_sync_requests = max_sync_requests
        self.sync_slot_lease_seconds = sync_slot_lease_seconds
        self._schema_lock = threading.Lock()
        self._initialized = False

    @contextmanager
    def connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @classmethod
    def from_env(cls) -> "MetadataJobStore":
        max_pending_jobs = int(os.getenv("ELM_LOQ_METADATA_MAX_PENDING_JOBS", "100"))
        retention_seconds = int(
            os.getenv("ELM_LOQ_METADATA_JOB_RETENTION_SECONDS", "86400")
        )
        lease_seconds = int(os.getenv("ELM_LOQ_METADATA_JOB_LEASE_SECONDS", "180"))
        max_sync_requests = int(
            os.getenv("ELM_LOQ_METADATA_MAX_SYNC_REQUESTS", "20")
        )
        if not 1 <= max_pending_jobs <= 10000:
            raise ValueError("ELM_LOQ_METADATA_MAX_PENDING_JOBS must be between 1 and 10000.")
        if not 60 <= retention_seconds <= 2592000:
            raise ValueError(
                "ELM_LOQ_METADATA_JOB_RETENTION_SECONDS must be between 60 and 2592000."
            )
        if not 30 <= lease_seconds <= 3600:
            raise ValueError(
                "ELM_LOQ_METADATA_JOB_LEASE_SECONDS must be between 30 and 3600."
            )
        if not 1 <= max_sync_requests <= 1000:
            raise ValueError(
                "ELM_LOQ_METADATA_MAX_SYNC_REQUESTS must be between 1 and 1000."
            )
        return cls(
            max_pending_jobs,
            retention_seconds,
            lease_seconds,
            max_sync_requests,
            lease_seconds,
        )

    @staticmethod
    def _connect():
        required = ("DB_DATABASE", "DB_USERNAME", "DB_PASSWORD")
        if any(not os.getenv(name) for name in required):
            raise ValueError("Metadata jobs require the configured PostgreSQL database.")
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "postgres"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.environ["DB_DATABASE"],
            user=os.environ["DB_USERNAME"],
            password=os.environ["DB_PASSWORD"],
            sslmode="disable",
            connect_timeout=5,
        )

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._schema_lock:
            if self._initialized:
                return
            self._initialize_schema()
            self._initialized = True

    def _initialize_schema(self) -> None:
        with self.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_recommendation_jobs (
                    id UUID PRIMARY KEY,
                    token_hash CHAR(64) NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'succeeded', 'failed')
                    ),
                    stage TEXT NOT NULL,
                    progress SMALLINT NOT NULL DEFAULT 0 CHECK (
                        progress BETWEEN 0 AND 100
                    ),
                    request_data JSONB,
                    result_data JSONB,
                    error_data JSONB,
                    attempts SMALLINT NOT NULL DEFAULT 0,
                    lease_until TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMPTZ,
                    expires_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS metadata_recommendation_jobs_queue_idx
                ON metadata_recommendation_jobs (created_at)
                WHERE status = 'queued'
                """
            )
            cursor.execute(
                """
                ALTER TABLE metadata_recommendation_jobs
                ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS metadata_recommendation_jobs_expiry_idx
                ON metadata_recommendation_jobs (expires_at)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_recommendation_job_events (
                    id BIGSERIAL PRIMARY KEY,
                    job_id UUID NOT NULL REFERENCES metadata_recommendation_jobs(id)
                        ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS metadata_recommendation_job_events_job_idx
                ON metadata_recommendation_job_events (job_id, id)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_llm_request_events (
                    id BIGSERIAL PRIMARY KEY,
                    bucket_key CHAR(64) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS metadata_llm_request_events_bucket_idx
                ON metadata_llm_request_events (bucket_key, created_at)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_sync_request_slots (
                    request_id UUID PRIMARY KEY,
                    expires_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS metadata_sync_request_slots_expiry_idx
                ON metadata_sync_request_slots (expires_at)
                """
            )

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_job(self, request_data: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        self.initialize()
        job_id = uuid4()
        token = os.urandom(32).hex()
        with self.connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("metadata-job-queue",))
            cursor.execute(
                """
                SELECT COUNT(*) AS count
                FROM metadata_recommendation_jobs
                WHERE status IN ('queued', 'running') AND expires_at > NOW()
                """
            )
            if cursor.fetchone()["count"] >= self.max_pending_jobs:
                raise JobQueueFull()
            cursor.execute(
                """
                INSERT INTO metadata_recommendation_jobs (
                    id, token_hash, status, stage, request_data, expires_at
                )
                VALUES (%s, %s, 'queued', 'queued', %s, NOW() + (%s * INTERVAL '1 second'))
                RETURNING id, status, stage, progress, created_at, updated_at
                """,
                (
                    str(job_id),
                    self.hash_token(token),
                    Json(request_data),
                    self.retention_seconds,
                ),
            )
            job = dict(cursor.fetchone())
            self._insert_event(cursor, job_id, "queued", self._public_job(job))
        return str(job_id), token, self._public_job(job)

    def get_job(
        self, job_id: UUID, token: str, include_result: bool = True
    ) -> Optional[dict[str, Any]]:
        self.initialize()
        with self.connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute(
                """
                SELECT id, token_hash, status, stage, progress, result_data,
                       error_data, created_at, started_at, updated_at, completed_at,
                       expires_at,
                       CASE WHEN started_at IS NULL THEN NULL ELSE
                           FLOOR(EXTRACT(EPOCH FROM (started_at - created_at)) * 1000)::BIGINT
                       END AS queue_wait_ms
                FROM metadata_recommendation_jobs
                WHERE id = %s AND expires_at > NOW()
                """,
                (str(job_id),),
            )
            row = cursor.fetchone()
        if row is None or not hmac.compare_digest(
            row["token_hash"].strip(), self.hash_token(token)
        ):
            return None
        result = self._public_job(dict(row))
        if include_result:
            result["result"] = row["result_data"]
        return result

    def get_events(
        self, job_id: UUID, token: str, after_id: int
    ) -> Optional[list[dict[str, Any]]]:
        self.initialize()
        with self.connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute(
                """
                SELECT token_hash
                FROM metadata_recommendation_jobs
                WHERE id = %s AND expires_at > NOW()
                """,
                (str(job_id),),
            )
            row = cursor.fetchone()
            if row is None or not hmac.compare_digest(
                row["token_hash"].strip(), self.hash_token(token)
            ):
                return None
            cursor.execute(
                """
                SELECT id, event_type, payload
                FROM metadata_recommendation_job_events
                WHERE job_id = %s AND id > %s
                ORDER BY id
                """,
                (str(job_id), after_id),
            )
            return [dict(event) for event in cursor.fetchall()]

    @staticmethod
    def _insert_event(cursor, job_id: UUID, event_type: str, payload: dict[str, Any]):
        cursor.execute(
            """
            INSERT INTO metadata_recommendation_job_events (job_id, event_type, payload)
            VALUES (%s, %s, %s)
            """,
            (str(job_id), event_type, Json(payload)),
        )

    @staticmethod
    def _public_job(job: dict[str, Any]) -> dict[str, Any]:
        public = {
            key: job[key]
            for key in (
                "id", "status", "stage", "progress", "created_at", "started_at",
                "updated_at", "queue_wait_ms",
            )
            if key in job
        }
        if "id" in public:
            public["id"] = str(public["id"])
        for key in ("created_at", "started_at", "updated_at", "completed_at"):
            if isinstance(public.get(key), datetime):
                public[key] = public[key].isoformat()
        if "completed_at" in job:
            public["completed_at"] = (
                job["completed_at"].isoformat()
                if isinstance(job["completed_at"], datetime)
                else job["completed_at"]
            )
        if "queue_wait_ms" in job:
            public["queue_wait_ms"] = job["queue_wait_ms"]
        if job.get("error_data") is not None:
            public["error"] = job["error_data"]
        return public

    def claim_job(self, worker_id: str, max_attempts: int = 3) -> Optional[dict[str, Any]]:
        self.initialize()
        with self.connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute(
                """
                UPDATE metadata_recommendation_jobs
                SET status = 'failed',
                    stage = 'failed',
                    progress = 100,
                    request_data = NULL,
                    error_data = %s,
                    completed_at = NOW(),
                    updated_at = NOW(),
                    lease_until = NULL
                WHERE status = 'running' AND lease_until < NOW()
                  AND attempts >= %s
                RETURNING id
                """,
                (Json({"detail": "The job worker stopped unexpectedly."}), max_attempts),
            )
            for expired in cursor.fetchall():
                self._insert_event(
                    cursor, str(expired["id"]), "failed",
                    {"status": "failed", "stage": "failed", "progress": 100},
                )
            cursor.execute(
                """
                SELECT id, request_data
                FROM metadata_recommendation_jobs
                WHERE expires_at > NOW() AND attempts < %s
                  AND (
                    status = 'queued'
                    OR (status = 'running' AND lease_until < NOW())
                  )
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (max_attempts,),
            )
            job = cursor.fetchone()
            if job is None:
                return None
            cursor.execute(
                """
                UPDATE metadata_recommendation_jobs
                SET status = 'running', stage = 'reading_document',
                    progress = GREATEST(progress, 5), attempts = attempts + 1,
                    lease_until = NOW() + (%s * INTERVAL '1 second'),
                    started_at = COALESCE(started_at, NOW()), updated_at = NOW()
                WHERE id = %s
                RETURNING id, request_data, attempts, stage, progress, created_at,
                          started_at, updated_at
                """,
                (self.lease_seconds, str(job["id"])),
            )
            claimed = dict(cursor.fetchone())
            self._insert_event(
                cursor, claimed["id"], "progress", self._public_job(claimed)
            )
            claimed["worker_id"] = worker_id
            return claimed

    def update_progress(self, job_id: UUID, stage: str, progress: int) -> None:
        self.initialize()
        with self.connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute(
                """
                UPDATE metadata_recommendation_jobs
                SET stage = %s, progress = GREATEST(progress, %s), updated_at = NOW(),
                    lease_until = NOW() + (%s * INTERVAL '1 second')
                WHERE id = %s AND status = 'running'
                RETURNING id, status, stage, progress, updated_at
                """,
                (stage, progress, self.lease_seconds, str(job_id)),
            )
            job = cursor.fetchone()
            if job is not None:
                self._insert_event(cursor, job_id, "progress", self._public_job(dict(job)))

    def heartbeat(self, job_id: UUID) -> None:
        self.initialize()
        with self.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE metadata_recommendation_jobs
                SET lease_until = NOW() + (%s * INTERVAL '1 second'), updated_at = NOW()
                WHERE id = %s AND status = 'running'
                """,
                (self.lease_seconds, str(job_id)),
            )

    def complete_job(self, job_id: UUID, result: dict[str, Any]) -> None:
        self.initialize()
        with self.connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute(
                """
                UPDATE metadata_recommendation_jobs
                SET status = 'succeeded', stage = 'completed', progress = 100,
                    request_data = NULL, result_data = %s, error_data = NULL,
                    completed_at = NOW(), updated_at = NOW(), lease_until = NULL
                WHERE id = %s AND status = 'running'
                RETURNING id, status, stage, progress, updated_at, completed_at
                """,
                (Json(result), str(job_id)),
            )
            job = cursor.fetchone()
            if job is not None:
                self._insert_event(
                    cursor, job_id, "completed", self._public_job(dict(job))
                )

    def fail_job(self, job_id: UUID, error: dict[str, Any]) -> None:
        self.initialize()
        with self.connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute(
                """
                UPDATE metadata_recommendation_jobs
                SET status = 'failed', stage = 'failed', progress = 100,
                    request_data = NULL, error_data = %s, result_data = NULL,
                    completed_at = NOW(), updated_at = NOW(), lease_until = NULL
                WHERE id = %s AND status = 'running'
                RETURNING id, status, stage, progress, updated_at, completed_at
                """,
                (Json(error), str(job_id)),
            )
            job = cursor.fetchone()
            if job is not None:
                payload = self._public_job(dict(job))
                payload["error"] = error
                self._insert_event(cursor, job_id, "failed", payload)

    def try_reserve_provider_call(
        self, bucket_key: str, limit: int, window_seconds: int
    ) -> float:
        self.initialize()
        with self.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))", (bucket_key,)
            )
            cursor.execute(
                """
                DELETE FROM metadata_llm_request_events
                WHERE bucket_key = %s
                  AND created_at <= NOW() - (%s * INTERVAL '1 second')
                """,
                (bucket_key, window_seconds),
            )
            cursor.execute(
                """
                SELECT COUNT(*), MIN(created_at)
                FROM metadata_llm_request_events
                WHERE bucket_key = %s
                """,
                (bucket_key,),
            )
            count, oldest = cursor.fetchone()
            if count < limit:
                cursor.execute(
                    """
                    INSERT INTO metadata_llm_request_events (bucket_key)
                    VALUES (%s)
                    """,
                    (bucket_key,),
                )
                return 0
            cursor.execute(
                """
                SELECT GREATEST(
                    0.05,
                    EXTRACT(EPOCH FROM (%s + (%s * INTERVAL '1 second') - NOW()))
                )
                """,
                (oldest, window_seconds),
            )
            return float(cursor.fetchone()[0])

    async def acquire_provider_call(
        self,
        bucket_key: str,
        limit: int,
        window_seconds: int,
        max_wait_seconds: float,
    ) -> int:
        started = time.monotonic()
        while True:
            delay = await asyncio.to_thread(
                self.try_reserve_provider_call, bucket_key, limit, window_seconds
            )
            if delay <= 0:
                return round((time.monotonic() - started) * 1000)
            elapsed = time.monotonic() - started
            if elapsed + delay > max_wait_seconds:
                raise ProviderWaitExceeded(delay)
            await asyncio.sleep(delay)

    def cleanup_expired(self) -> None:
        self.initialize()
        with self.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM metadata_recommendation_jobs WHERE expires_at <= NOW()"
            )
            cursor.execute(
                """
                DELETE FROM metadata_llm_request_events
                WHERE created_at < NOW() - INTERVAL '1 hour'
                """
            )
            cursor.execute(
                "DELETE FROM metadata_sync_request_slots WHERE expires_at <= NOW()"
            )

    def try_acquire_sync_slot(self, request_id: UUID) -> bool:
        self.initialize()
        with self.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                ("metadata-sync-request-slots",),
            )
            cursor.execute(
                "DELETE FROM metadata_sync_request_slots WHERE expires_at <= NOW()"
            )
            cursor.execute("SELECT COUNT(*) FROM metadata_sync_request_slots")
            if cursor.fetchone()[0] >= self.max_sync_requests:
                return False
            cursor.execute(
                """
                INSERT INTO metadata_sync_request_slots (request_id, expires_at)
                VALUES (%s, NOW() + (%s * INTERVAL '1 second'))
                """,
                (str(request_id), self.sync_slot_lease_seconds),
            )
            return True

    def release_sync_slot(self, request_id: UUID) -> None:
        self.initialize()
        with self.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM metadata_sync_request_slots WHERE request_id = %s",
                (str(request_id),),
            )
def valid_job_id(value: str) -> Optional[UUID]:
    try:
        return UUID(value)
    except (ValueError, TypeError, AttributeError):
        return None
