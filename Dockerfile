# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies and git
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set permissions
RUN useradd -ms /bin/bash appuser
# Set the working directory in the container
WORKDIR /home/appuser/app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Switch to non-root user
USER appuser

# Copy the rest of the application code into the container
COPY --chown=appuser . /home/appuser/app

# Expose the port that the app runs on (documentation only — Render still uses $PORT)
EXPOSE 8501

# Command to run the application — bind to Render's $PORT, fall back to 8501 for local docker run
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0