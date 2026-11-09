# WISYKI-API

The WISYKI-API is a tool developed as part of the [WISY@KI project](https://www.wisyki.de/) that provides capabilities to predict [ESCOv1.2.0](https://esco.ec.europa.eu/), [GRETAv2](https://www.greta-die.de/webpages/greta-interaktiv), DKZ, DigComp 2.2 (ESCO) and DigCompEdu skills based on given course descriptions or learning outcomes. Predictions include learning outcomes, prerequisites, keywords and competency levels.

The application relies on a set of external services rather than providing them itself, which keeps the memory footprint small and improves maintainability:

* **A running Chroma vector database** that stores the skill vectors.
* **An embedding server** (e.g. a HuggingFace Inference endpoint) that provides the embedding model used to vectorize course descriptions and queries.
* **An LLM API endpoint** for learning outcome extraction, keyword extraction and LLM validation. The endpoint and model are fully configurable via environment variables, so you can point the API at any OpenAI-compatible LLM service of your choice.

The skill data used to populate the Chroma database is provided as JSONL files in the `data/import` directory. These files contain the documents, their metadata and (optionally) precomputed embeddings. On first startup, the API reads these files and populates the configured Chroma collection automatically.

## Features

* Predict ESCOv1.2.0, GRETAv2, DKZ, DigComp 2.2 (ESCO) and DigCompEdu Skills based on course descriptions or learning outcomes.
* Predict competency levels.
* Create embeddings.
* Provide validated training data.

## Installation

To get the API up and running, follow these steps:

### Prerequisites

You will need to have Docker installed on your machine. If you don't have Docker installed, you can download it from [the official docker website](https://www.docker.com/products/docker-desktop).

You will also need access to the following external services:

* A running **Chroma** vector database.
* An **embedding server** (e.g. a HuggingFace Inference endpoint) providing the embedding model.
* An **LLM API endpoint** (OpenAI-compatible) for learning outcome extraction, keyword extraction and validation.

### Environment Variables

The following environment variables are required:

* `POSTGRES_PASSWORD`: The password for the PostgreSQL database.
* `PGADMIN_EMAIL`: The email for PgAdmin.
* `PGADMIN_PASSWORD`: The password for PgAdmin.
* `HOST_PORT`: The host port for the API.
* `POSTGRES_PORT`: The port for the PostgreSQL database.
* `PGADMIN_PORT`: The port for PgAdmin.
* `CHROMA_PORT`: The port for the Chroma server.
* `HF_TEXT_INF_PORT`: The port for the text-embeddings-inference service.
* `ROOT_PATH`: This variable is used when the API is running behind a proxy. If the API is served under a subpath, such as `/api/v1`, then `ROOT_PATH` should be set to that subpath (`/api/v1` in this case). If the API is not behind a proxy or is served from the root path, this variable can be left empty.

The following environment variables configure the external services and can be set freely to match your setup:

* `LLM_API_URL`: The base URL of your OpenAI-compatible LLM API endpoint.
* `LLM_MODEL`: The name of the LLM model to use.
* `LLM_API_KEY`: The API key for the LLM endpoint.
* `CHROMA_HOST`: The host of the Chroma server.
* `CHROMA_PORT`: The port of the Chroma server.
* `CHROMA_COLLECTION`: The name of the Chroma collection to use.
* `CHROMA_DATABASE`: The Chroma database to use (defaults to `wisyki`).
* `CHROMA_TENANT`: The Chroma tenant (defaults to `default_tenant`).
* `CHROMA_SERVER_AUTHN_CREDENTIALS`: The authentication token for the Chroma server.
* `CHROMA_SERVER_AUTHN_PROVIDER`: The authentication provider for the Chroma server.
* `HF_EMBEDDING_HOST`: The URL of the embedding server (e.g. a HuggingFace Inference endpoint).
* `HF_EMBEDDING_API_KEY`: The API key for the embedding server.
* `HF_EMBEDDING_MODEL`: The model served by the local text-embeddings-inference service.

The provided `docker-compose.yml` starts all required services itself, including the Chroma vector database and the text-embeddings-inference embedding server. If you already run these services elsewhere, you can point the API at them via the environment variables above instead.

### Steps

1. Clone the repository:

    ```bash
    git clone <repository-url>
    ```

2. Navigate to the project directory:

    ```bash
    cd wisyki-api
    ```

3. Download the skill import data:

   Download the skill import data archive from the [release page](https://github.com/ild-thl/wisyki-api/releases).

4. Extract the import data to the correct location:

   Unzip the downloaded file into a directory at `data/import`. The JSONL files in this directory are read on first startup to populate the configured Chroma collection.

5. Copy and setup environment variables:

    ```bash
    cp .env.example .env
    ```

   Then edit `.env` and set the values for your external services (LLM endpoint, Chroma server and embedding server).

6. Run the Docker compose command:

    ```bash
    docker-compose up -d
    ```

    or run for local development

    ```bash
    docker compose -f docker-compose.dev.yml up -d
    ```

On first startup, the API connects to the configured Chroma server. If the collection is empty, it populates it from the JSONL files in `data/import`.

## Docker Image Build

The official Docker image for the WISYKI-API is available for amd64 and arm64 architectures. To build the Docker image for both architectures, Docker Buildx was used.

Creating a new builder instance and setting it as the active one is necessary for building multi-platform images.

```bash
docker buildx create --name mybuilder --use
docker buildx inspect --bootstrap
```

After setting up the builder, the Docker image can be built and pushed to the Docker Hub repository using the following command:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t pascalhuerten/comp-ai-api:latest --push .
```

This command builds the Docker image for both `linux/amd64` and `linux/arm64` platforms and pushes it to the Docker Hub repository `pascalhuerten/comp-ai-api`.
Make sure to replace `pascalhuerten/comp-ai-api` with your own Docker Hub repository name if you want to push it to your own repository.

## API Documentation

For more details on how to use the API, please refer to the [API Documentation](https://lab.dlc.sh/competence-analyser/docs).

## Acknowledgements

This service uses the ESCO classification of the European Commission.

This API was developed as part of the project WISY@KI by the Institut für interakive Systeme, Technical University of Applied Sciences Lübeck and was funded by the Federal Ministry of Education and Research.

<div style="display: flex; align-items: center; gap: 35px;">
    <img src="static/bmbf_logo.jpg" alt="BMBF Logo" style="width: 150px; height: auto;">
    <img src="static/thl_logo.jpg" alt="THL Logo" style="width: 150px; height: auto;">
</div>

## Support

If you encounter any issues or require further assistance, feel free to raise an issue in this repository.

## Contributing

We welcome contributions from the community.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
