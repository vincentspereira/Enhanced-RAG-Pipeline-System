# Use CUDA-enabled base image supporting Python 3.11
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="Jules <jules@example.com>"
LABEL description="RAG System Application with multiple services."

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies including Python 3.11
# Add git for potential VCS operations by tools, curl for downloads
# tesseract-ocr and poppler-utils are from original Dockerfile, kept for now.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    curl \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies using pip for python3.11
# Using a virtual environment is good practice even in Docker,
# but for simplicity here, installing to system Python 3.11 site-packages.
# If venv is desired:
# RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Ollama
# Note: This installs Ollama system-wide. Consider user permissions if needed.
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the rest of the application code
COPY . .

# Copy the entrypoint script and make it executable (already done via tool, but good for Dockerfile context)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create necessary directories if they are not created by the application itself
# Ensure these paths align with application's expectations (e.g., config.yaml, service code)
RUN mkdir -p /app/cache /app/logs /app/data/model_versions /app/data/benchmarks /app/data/knowledge_graphs /app/data/api_analytics /app/data/api_tokens

# Set Python path to include the /app directory where Scripts and other modules are
ENV PYTHONPATH=/app

# Expose ports used by different services (gateway, individual services)
# The actual port used inside the container will be determined by the service's config.
# This just informs Docker that the container *might* listen on these ports.
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002
EXPOSE 11434 # Default Ollama port

# Entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (service to run).
# This can be overridden when running `docker run` e.g., `docker run myimage rag_query_service`
CMD ["internal_api_gateway"]

# Optional: Pull a default Ollama model during build time.
# This increases image size but ensures the model is available.
# Consider if this is best done at build or runtime (e.g., in entrypoint or by the app itself).
# RUN ollama pull llama2 # Example, replace with actual desired model like snowflake-arctic-embed
# The original Dockerfile had: RUN ollama pull snowflake-arctic-embed2:latest
# Keeping it for consistency, but be mindful of build times and image size.
# It might be better to have a separate step or script to initialize models if they are large or change often.
# Ensure ollama service is running if you try to pull here, which it isn't during typical RUN.
# This pull might be better suited for a script run after ollama serve starts, or manually.
# For now, commenting out the build-time pull to avoid potential issues with ollama server not being ready.
# If you need it, you'd typically do:
# CMD ollama serve & ollama pull your_model && wait
# But that's more for a container that *only* serves ollama and pre-pulls.
# With the current entrypoint.sh, one could run `docker run myimage ollama_serve`
# and then `docker exec <container_id> ollama pull modelname`.

# Or, if a specific model is always needed by one of the python services,
# that service's startup sequence could check and pull the model.
# For now, leaving model pulling as a runtime concern or manual step after starting ollama_serve.
