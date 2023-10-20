# Use an official Python runtime as a parent image
FROM python:3.10.13

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Make port 7680 available to the world outside this container
EXPOSE 7680

# Run the Flask app when the container launches
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "7680"]