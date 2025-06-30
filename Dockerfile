# Use CUDA-enabled base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Install software-properties-common to manage PPAs, and other system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for newer Python versions
RUN add-apt-repository ppa:deadsnakes/ppa -y

# Install Python 3.13 and related packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-pip \
    python3.13-venv \
    python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives to make python3.13 the default python3 and pip3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.13 1
    # Note: pip from deadsnakes PPA might install to /usr/local/bin for the specific version

# Install Ollama (can be done before or after Python setup, as long as curl is available)
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies using python3.13 explicitly to be sure
RUN python3.13 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories (if not already created by COPY . .)
# Best practice is to ensure they exist if app expects them.
RUN mkdir -p /app/cache /app/logs

# Pull the Ollama model
# Ensure Ollama service is running or accessible if `ollama pull` requires it during build.
# Often, models are pulled at runtime or in an entrypoint script.
# For build-time pull, this is okay if `ollama` command works standalone after install.
RUN ollama pull snowflake-arctic-embed2:latest

# Set up environment for GPU usage
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for API
EXPOSE 8000

# Start command
CMD ["python3", "-m", "Scripts.main"]
