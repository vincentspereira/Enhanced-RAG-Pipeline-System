# Use CUDA-enabled base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Install system dependencies, including an attempt for Python 3.13
# Note: python3.13 might not be available in default ubuntu22.04 repos for nvidia/cuda base.
# If this step fails, may need to use a PPA, compile from source, or use a newer base / different strategy.
# For now, trying with python3.13 directly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.13 python3.13-pip python3.13-venv \
    curl git tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    || (apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-pip python3.12-venv \
        curl git tesseract-ocr poppler-utils \
        && rm -rf /var/lib/apt/lists/* \
        && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
        && update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.12 1) \
    || (apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-pip python3.11-venv \
        curl git tesseract-ocr poppler-utils \
        && rm -rf /var/lib/apt/lists/* \
        && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
        && update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.11 1)

# Ensure python3 and pip3 point to the installed version if not python3.13
# This step might need adjustment based on which Python version was successfully installed.
# If python3.13 was installed, these alternatives might not be strictly necessary if it becomes default.
# Assuming one of the python3.11, 3.12 or 3.13 installs succeeded and became 'python3'
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 || \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 || \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
# RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.13 1 || \
#     update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.12 1 || \
#     update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.11 1
# The above alternative setup is complex and error-prone. A simpler way if a specific version is needed
# and installed (e.g. python3.11 becomes the fallback) is to use `python3.11 -m pip ...` directly.
# For now, the OR chain for installation attempts to get the highest version.

# Install Ollama (better to do this before copying app code if it doesn't depend on it)
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies using the available python3
# This will use whatever 'python3' points to after apt-get and alternatives.
RUN python3 -m pip install --no-cache-dir -r requirements.txt

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
