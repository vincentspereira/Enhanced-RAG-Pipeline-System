#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Default service if not specified
DEFAULT_SERVICE="internal_api_gateway"
SERVICE_TO_RUN="${1:-$DEFAULT_SERVICE}" # Use first argument or default

echo "Attempting to start service: $SERVICE_TO_RUN"

# Add Scripts directory to PYTHONPATH if not already set by Dockerfile (belt and suspenders)
export PYTHONPATH=/app:$PYTHONPATH

if [ "$SERVICE_TO_RUN" = "internal_api_gateway" ]; then
    echo "Starting Internal API Gateway..."
    exec python3 -u /app/Scripts/services/internal_api_gateway.py
elif [ "$SERVICE_TO_RUN" = "rag_query_service" ]; then
    echo "Starting RAG Query Service..."
    exec python3 -u /app/Scripts/services/rag_query_service.py
elif [ "$SERVICE_TO_RUN" = "doc_processing_service" ]; then
    echo "Starting Document Processing Service..."
    exec python3 -u /app/Scripts/services/doc_processing_service.py
elif [ "$SERVICE_TO_RUN" = "notification_service" ]; then
    echo "Starting Notification Service..."
    exec python3 -u /app/Scripts/services/notification_service.py
elif [ "$SERVICE_TO_RUN" = "main_script" ]; then # Option to run the original main script
    echo "Starting main script (Scripts.main)..."
    exec python3 -u -m Scripts.main
elif [ "$SERVICE_TO_RUN" = "ollama_serve" ]; then # Option to just run ollama serve
    echo "Starting Ollama serve..."
    # This might need to run in background if other services depend on it within the same container,
    # or this container is dedicated to Ollama. For now, it takes over.
    exec ollama serve
else
    echo "Error: Unknown service '$SERVICE_TO_RUN'"
    echo "Available services: internal_api_gateway, rag_query_service, doc_processing_service, notification_service, main_script, ollama_serve"
    exit 1
fi
