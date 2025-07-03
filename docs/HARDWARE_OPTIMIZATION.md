# Hardware Optimization and Resource Management

This document outlines the current strategies and considerations for hardware optimization and resource management within the RAG system.

## 1. Overview

The system is designed to leverage available hardware resources, particularly GPUs for machine learning model inference (e.g., embeddings, language models) and CPUs for general processing and data handling. Resource management is primarily handled through configuration and orchestration layer settings (Kubernetes).

## 2. GPU Utilization

*   **Detection and Usage:**
    *   Services like `RAG Query Service` and `Document Processing Service` utilize PyTorch.
    *   They check for CUDA availability (`torch.cuda.is_available()`) during startup.
    *   The device for ML models (e.g., sentence transformers) can be configured via `config.yaml` (`model.device: "cuda"` or `model.device: "cpu"`) or corresponding environment variables.
    *   If "cuda" is specified and a GPU is available, models are loaded onto the GPU.
*   **Fallback:**
    *   If "cuda" is configured but no compatible GPU is found at runtime, the services will log a warning and fall back to using the CPU for model operations.
*   **Containerization:**
    *   The Dockerfile (`Dockerfile`) is based on an `nvidia/cuda` base image, providing the necessary CUDA runtime libraries.
    *   When deploying with Docker Compose (`docker-compose.yml`), GPU access can be requested for services.
    *   In Kubernetes:
        *   Manual manifests (`deployment/kubernetes/deployment.yaml`) for `rag-pipeline-api` request `nvidia.com/gpu: 1`.
        *   The Helm chart (`charts/rag-system/`) has been updated to allow specifying GPU resources (e.g., `nvidia.com/gpu`) in `values.yaml` for `ragQueryService` and `docProcessingService` deployments. This requires the Kubernetes cluster to have the NVIDIA device plugin installed and GPUs available on nodes.

## 3. CPU Utilization

*   **General Processing:** Standard Python services and FastAPI utilize CPU for I/O, request handling, and application logic.
*   **Parallelism:**
    *   Underlying libraries like PyTorch and SentenceTransformers may utilize multiple CPU cores for their operations when running in CPU mode.
    *   The system configuration (`config.yaml`) includes `processing.max_workers`, which suggests that some tasks (like document processing in older scripts or certain parallelizable operations) might use thread pools or process pools to leverage multiple CPU cores.
    *   FastAPI applications run by Uvicorn can be configured with multiple worker processes (e.g., `api.workers` in `config.yaml`, though this is typically for Gunicorn with Uvicorn workers).

## 4. Resource Management (Kubernetes)

*   **Requests and Limits:**
    *   CPU and memory requests and limits are defined for services in Kubernetes deployments (both manual manifests and Helm chart templates). This helps Kubernetes schedule pods appropriately and manage resource allocation.
*   **Horizontal Pod Autoscaler (HPA):**
    *   The `deployment/kubernetes/deployment.yaml` for `rag-pipeline-api` includes an HPA that scales the number of pod replicas based on CPU and memory utilization. This provides dynamic resource allocation at the pod replica level in response to workload changes.

## 5. Current Limitations and Future Considerations

*   **GPU Insufficiency Handling:**
    *   The current fallback mechanism is binary: if a GPU is configured and available, it's used; otherwise, CPU is used.
    *   The system does not currently implement dynamic load balancing for a single type of task between GPU and CPU if the GPU becomes a bottleneck (e.g., tasks queuing for GPU while CPU is idle). This would require a more sophisticated internal queueing and dispatching mechanism.
*   **Custom Resource Management System:**
    *   Beyond the configuration-driven device selection and basic PyTorch CUDA checks, there is no advanced custom resource management system within the application (e.g., for managing a pool of GPUs, prioritizing tasks for specific hardware, or dynamically adjusting resources within a pod).
*   **ML-based Prediction of Resource Requirements:**
    *   This is a future enhancement and not currently implemented. It would involve analyzing historical usage patterns to predict future needs.
*   **Detailed Hardware Profiling (In-App):**
    *   The system relies on external monitoring tools like Prometheus and Grafana for performance monitoring and bottleneck identification. There are no built-in tools for fine-grained hardware profiling (e.g., GPU kernel analysis, CPU cache efficiency).
*   **Energy Optimization:**
    *   As a server-side application, specific energy optimization modes for mobile devices are not applicable. General server energy efficiency relies on efficient code and appropriate hardware selection.

## 6. Recommendations for Users/Deployers

*   **GPU Nodes:** If GPUs are intended for use, ensure your Kubernetes cluster has nodes with compatible NVIDIA GPUs and the NVIDIA device plugin for Kubernetes is installed.
*   **Helm Chart Configuration:** When deploying via the Helm chart, customize `values.yaml` to set `MODEL_DEVICE: "cuda"` for `ragQueryService` and `docProcessingService` and specify GPU resource requests/limits under their respective `gpus` sections if those services are intended to run on GPU nodes. Example:
    ```yaml
    ragQueryService:
      # ...
      config:
        MODEL_DEVICE: "cuda"
      resources:
        requests:
          cpu: "500m"
          memory: "2Gi"
        limits:
          cpu: "1"
          memory: "4Gi"
      gpus:
        limits:
          nvidia.com/gpu: "1"
    ```
*   **Monitoring:** Actively monitor CPU, memory, and GPU (if used) utilization via Prometheus/Grafana to understand resource consumption patterns and identify potential bottlenecks or needs for scaling.
*   **Load Testing:** Conduct load testing to determine appropriate resource requests/limits for your expected workload and to observe how the HPA behaves.

This document will be updated as hardware optimization and resource management capabilities evolve.
