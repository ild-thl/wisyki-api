# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.10.13
FROM python:$PYTHON_VERSION

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

VOLUME ["/data"]

# Make the host_port available to the world outside this container
EXPOSE 7680

# Run the FastAPI app when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7680"]