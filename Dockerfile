# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the model file is included (assuming it's already in your directory)
COPY titanic_model.pickle /app/

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run server1.py when the container launches
CMD ["python", "server1.py"]

