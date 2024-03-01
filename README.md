# WISYKI-API

The WISYKI-API is a tool developed as part of the [WISY@KI project](https://www.wisyki.de/) that provides capabilities to predict ESCO, GRETA, and DKZ Skills based on given course descriptions or learning outcomes. It also has the ability to predict competency levels.

The application uses two fine-tuned models:

1. [Instructor SkillFit](https://huggingface.co/pascalhuerten/instructor-skillfit): A fine-tuning of teh embedding model [hkunlp/instructor-base](https://huggingface.co/hkunlp/instructor-base) used for retrieving relevant skills from the vector database.
2. [BGE Reranker SkillFit](https://huggingface.co/pascalhuerten/bge_reranker_skillfit): A fine-tuning of the cross-encoder model [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) used for reranking/validating skill predictions.

In addition to the above models, the WISYKI-API also uses other models via their APIs for learning outcome extraction and LLM validation. Depending on the request, the following models are used:

For learning outcome extraction, one of the following lightweight models is used:

1. [zephyr-7b-alpha](https://wiki.mylab.th-luebeck.dev/de/mylab-llms)
2. mistral-small (API key required)
3. gpt-3.5-turbo-1106 (API key required)

For more powerful LLM validation, one of the following lightweight models is used:

1. [em-german-70b](https://wiki.mylab.th-luebeck.dev/de/mylab-llms)
2. mistral-medium (API key required)
3. gpt-4-0125-preview (API key required)

Please note that some of these models require an API key for access.

## Features

* Predict ESCO, GRETA, and DKZ Skills based on course descriptions or learning outcomes.
* Predict competency levels.
* Create embeddings.
* Provide validated training data.

## Installation

To get the API up and running, follow these steps:

### Prerequisites

You will need to have Docker installed on your machine. If you don't have Docker installed, you can download it from [the official docker website](https://www.docker.com/products/docker-desktop).

### Environment Variables

The following environment variables are required:

* `POSTGRES_PASSWORD`: The password for the PostgreSQL database.
* `PGADMIN_EMAIL`: The email for PgAdmin.
* `PGADMIN_PASSWORD`: The password for PgAdmin.
* `HOST_PORT`: The host port.
* `POSTGRES_PORT`: The port for the PostgreSQL database.
* `PGADMIN_PORT`: The port for PgAdmin.

### Steps

1. Clone the repository:

    ```bash
    git clone <repository-url>
    ```

2. Navigate to the project directory:

    ```bash
    cd wisyki-api
    ```

3. Run the Docker compose command:

    ```bash
    docker-compose up
    ```

Alternatively, you can pull the Docker image from Docker Hub:

```bash
docker pull pascalhuerten/comp-ai-api:latest
```

## API Documentation

For more details on how to use the API, please refer to the [API Documentation](http://141.144.239.168/redoc).

## Support

If you encounter any issues or require further assistance, feel free to raise an issue in this repository.

## Contributing

We welcome contributions from the community.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
