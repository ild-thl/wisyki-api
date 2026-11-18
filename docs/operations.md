# Operations

## Dependencies and configuration

The development Compose file starts PostgreSQL, Chroma, and Hugging Face Text Embeddings Inference. An OpenAI-compatible LLM provider is configured separately for extraction, LLM validation, summaries, and localization. PostgreSQL coordinates request admission and shared LLM rate limits, and stores background jobs. `.env.example` lists the available environment variables. This Compose configuration is for local development, not production deployment.

`LLM_MODEL`, `LLM_API_URL`, and `LLM_API_KEY` provide the defaults. Optional stage-specific model and provider overrides are:

| Variables | Purpose |
| --- | --- |
| `LLM_MODEL_COMPLEX` | Stronger default model for validation and calls requesting the most competent model |
| `LLM_EXTRACTION_MODEL`, `LLM_VALIDATION_MODEL`, `LLM_SUMMARY_MODEL`, `LLM_TRANSLATION_MODEL` | Select a model for an individual stage |
| `LLM_{EXTRACTION,VALIDATION,SUMMARY,TRANSLATION}_API_URL` | Select a provider base URL per stage |
| `LLM_{EXTRACTION,VALIDATION,SUMMARY,TRANSLATION}_API_KEY` | Select a provider credential per stage |
| `ELM_LOQ_METADATA_RESPONSE_FORMAT` | Choose portable prompt formatting or provider-enforced JSON mode |

Empty stage-specific values inherit the defaults. Model clients initialize when needed, so requests that do not use an LLM do not initialize one. Keep provider credentials in deployment configuration, not request payloads or logs.

## Capacity and provider limits

LLM request starts are coordinated across API and worker processes through PostgreSQL. The default shared quota is two requests per ten seconds per provider credential. Configure the quota with `ELM_LOQ_METADATA_LLM_REQUESTS_PER_WINDOW` and `ELM_LOQ_METADATA_LLM_RATE_WINDOW_SECONDS`. If different credentials share a provider quota, set the same arbitrary `ELM_LOQ_METADATA_LLM_RATE_LIMIT_BUCKET` value for those deployments; only a digest of this value is stored.

Synchronous processing is capped at 20 active requests by default across API processes. Excess requests are rejected with HTTP 429 rather than queued indefinitely. The `ELM_LOQ_METADATA_MAX_SYNC_REQUESTS` setting changes this limit. Per-request LLM concurrency defaults to two and is bounded from one to four by `ELM_LOQ_METADATA_LLM_CONCURRENCY`. Increasing concurrency does not increase the shared provider quota and may not reduce latency when a provider is saturated.

## Statistics and latency

Statistics are omitted by default. Callers can request aggregate pipeline timings and candidate counts, or full per-call and token-usage data. Token counts are reported only when supplied by provider metadata; missing values remain unknown. The API does not estimate monetary costs. Apply the relevant provider pricing to reported usage, and treat failed or cancelled calls without usage metadata as unknown cost.

Pipeline and LLM wall durations are separate from HTTP ingress and network queueing. Stage and individual call durations can overlap, so they should not be summed to estimate total request latency. The OpenAPI response schema documents the precise fields and levels.

## Background jobs

Synchronous processing is the default. Background mode requires both the API and metadata worker to be running with access to the shared PostgreSQL database. The local development Compose setup starts both services. Rebuild and recreate the services after changing their image or environment configuration:

```bash
docker compose up --build -d
```

The default queue limit is 100 pending plus running jobs; queue-full requests receive HTTP 429. Completed job results and events are retained for 24 hours by default. A worker crash can lead to a retry (up to three claims), so provider requests may be repeated and billed again. Configure queue size, worker concurrency, retention, lease, and timeout with the `ELM_LOQ_METADATA_*` variables listed in `.env.example`.

Job tokens are secret capabilities. Send them in the designated request header, never in URLs or logs. Progress events indicate pipeline stages; clients should retrieve the final job status/result after receiving a terminal event.

## Tests

The unit and HTTP contract suite uses fake providers and an isolated in-memory Chroma collection. It does not need live LLM, embedding, or database services. From the repository root, run:

```bash
docker compose run --rm --no-deps \
  wisyki-api python -m unittest discover -s /code/tests -v
```
