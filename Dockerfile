# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - build-essential: for compiling python packages
# - libopenblas-dev: a dependency for numpy/scipy/faiss
# - git: for pulling any git-based dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: reduces image size
# --prefer-binary: prefers pre-compiled packages, which can be faster
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Add a healthcheck to verify the app is running
# This tells the container orchestrator (like Docker Swarm or Kubernetes)
# if the application is healthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Define the entrypoint command to run the application
# Use the "streamlit run" command and set server.runOnSave to false
# for better performance in a containerized environment.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=false"]
