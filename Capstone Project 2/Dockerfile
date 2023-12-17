# Use Python 3.10.13 slim image based on Debian Bullseye as the parent image
FROM python:3.10.13-slim-bullseye

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set the working directory to /app
WORKDIR /app

# Copy requirements.txt and other necessary files to /app
COPY requirements.txt train.py predict.py model.ipynb Dockerfile salaries.csv stacked_model.h5 stacked_model.pkl predict_for_Docker.py ping.py salaries.csv /app/

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The CMD instruction should specify the script to run.
# Replace "predict.py" with the name of the script you want to execute.
CMD ["python", "predict_for_Docker.py"]
