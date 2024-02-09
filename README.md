# WISYKI-API

The WISYKI-API is a tool developed as part of the [WISY@KI project](https://www.wisyki.de/) that provides capabilities to predict ESCO, GRETA, and DKZ Skills based on given course descriptions or learning outcomes. It also has the ability to predict competency levels.

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
