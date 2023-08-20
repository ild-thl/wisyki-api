# Use an official Python runtime as a parent image
FROM python:3

# Set the working directory within the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install --upgrade pip
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the Flask app when the container launches
CMD ["flask", "run", "--port", "80"]
