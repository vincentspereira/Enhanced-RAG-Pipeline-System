# Intelligent Hardware Optimization Strategy (Conceptual)

This document outlines a conceptual strategy for intelligent hardware optimization within the RAG system. The goal is to enable services, particularly those performing computationally intensive tasks like model inference (embeddings, LLM responses), to efficiently utilize available hardware resources (GPU, CPU).

**Phase 1 Focus**: This document primarily serves as a placeholder for the *intended strategy* as per original Phase 1 requirements. Full implementation of these advanced features is complex and would likely occur in later development phases.

## 1. Objectives

*   **Performance**: Maximize throughput and minimize latency for ML tasks by leveraging specialized hardware (GPUs) when available.
*   **Efficiency**: Utilize CPU resources effectively when GPUs are not available or are insufficient.
*   **Flexibility**: Adapt to different deployment environments with varying hardware configurations.
*   **Resource Awareness**: Enable services to make informed decisions based on detected hardware.

## 2. Core Components of the Strategy

### 2.1. Hardware Detection

*   **Mechanism**: At startup, services that perform ML inference (e.g., RAG Query Service, Document Processing Service) should attempt to detect available hardware.
    *   **GPU Detection**:
        *   Primarily using libraries like PyTorch (`torch.cuda.is_available()`, `torch.cuda.device_count()`) or TensorFlow (`tf.config.list_physical_devices('GPU')`).
        *   The selected ML framework (e.g., SentenceTransformers, Hugging Face Transformers) usually handles device placement (e.g., `.to('cuda')`).
    *   **CPU Detection**:
        *   Standard Python libraries like `os.cpu_count()` to determine the number of available CPU cores.
*   **Configuration**:
    *   A configuration parameter (e.g., `MODEL_DEVICE` set to `auto`, `cuda`, or `cpu`) will guide the service.
        *   `auto`: Service attempts to detect and use GPU if available, otherwise falls back to CPU.
        *   `cuda`: Service attempts to use GPU; fails or logs a warning if not available.
        *   `cpu`: Service forces CPU usage.

### 2.2. Prioritization Logic (Service-Level)

1.  **GPU First**: If `MODEL_DEVICE` is `auto` or `cuda`, and a compatible GPU is detected and available (e.g., not fully utilized by other processes if sharing is a concern), the service will load models onto the GPU and perform inference there.
2.  **CPU Fallback**:
    *   If `MODEL_DEVICE` is `auto` and no GPU is found/usable, or if `MODEL_DEVICE` is `cpu`.
    *   The service will use CPU for model inference.
    *   **Multi-core CPU Utilization**:
        *   For libraries like SentenceTransformers or Hugging Face Transformers, batch processing during `encode()` or inference pipelines often utilizes multiple CPU cores by default or can be configured to do so (e.g., via underlying PyTorch/TensorFlow settings for inter/intra-op parallelism).
        *   For custom parallelizable tasks (e.g., processing multiple documents or chunks independently before model inference), Dask or Python's `multiprocessing` could be leveraged.

### 2.3. Dynamic Resource Allocation in Kubernetes (Conceptual - Future Enhancement)

While services will make local decisions based on detected hardware within their pod, Kubernetes plays a crucial role in allocating those pods to appropriate nodes and managing resources.

*   **GPU Nodes**:
    *   Kubernetes clusters need to be configured with GPU-enabled nodes and the necessary device plugins (e.g., NVIDIA device plugin).
    *   Pods requiring GPUs will specify them in their resource requests/limits:
      ```yaml
      resources:
        limits:
          nvidia.com/gpu: 1 # Request 1 GPU
      ```
    *   Node selectors (`nodeSelector`) or taints and tolerations would be used to ensure GPU-requiring pods are scheduled onto GPU-enabled nodes.
*   **CPU Resource Profiles**:
    *   Pods running CPU-only inference would have appropriate CPU and memory requests/limits.
    *   Different deployment profiles (e.g., via Helm values) could exist for CPU-only vs. GPU-enabled deployments of a service, with different resource settings.
*   **Horizontal Pod Autoscaler (HPA)**: Can scale service replicas based on CPU/memory utilization or custom metrics. This helps handle varying load but doesn't change the hardware type a single pod uses.
*   **Vertical Pod Autoscaler (VPA) (Future Consideration)**: Could potentially adjust CPU/memory requests for pods over time, but typically doesn't handle switching between CPU/GPU or managing GPU types.
*   **Custom Schedulers / Admission Controllers (Advanced - Future)**: For very sophisticated scenarios, custom Kubernetes scheduling logic could make decisions based on fine-grained hardware capabilities and current utilization across the cluster.

## 3. Specific Service Considerations

*   **RAG Query Service & Document Processing Service**:
    *   These are the primary candidates for hardware optimization due to embedding model usage.
    *   The `MODEL_DEVICE` configuration (already implemented for these services) is the first step.
    *   Future work could involve more dynamic batch sizing for encoding based on available GPU memory.

## 4. ML-based Prediction of Resource Requirements (Advanced - Future Phase)

*   Collect telemetry on actual resource usage (CPU, GPU, memory) under various loads and for different models/tasks.
*   Use this data to train ML models that can predict resource needs for upcoming jobs or to dynamically adjust K8s resource requests.
*   This is a highly advanced topic requiring significant MLOps infrastructure and expertise.

## 5. Energy Optimization (Mobile Devices - Future Phase, if applicable)

*   If the system or parts of it are ever intended to run on edge/mobile devices with power constraints:
    *   Utilize model quantization and pruning techniques.
    *   Employ lighter model architectures.
    *   Implement specific "low power" or "energy saving" modes that might reduce inference frequency, use smaller models, or batch requests differently.
*   This is generally outside the scope of the current server-side RAG system focus.

## Current Implementation Status (as of Iteration 12/13)

*   Services (RAG Query, Doc Processing) have a `MODEL_DEVICE` configuration that allows specifying `cpu` or `cuda`.
*   The services attempt to use the configured device, with a fallback to CPU if CUDA is specified but unavailable.
*   Kubernetes deployment files (raw YAMLs and Helm templates) include placeholders or examples for GPU resource requests (`nvidia.com/gpu: 1`) but are typically defaulted to CPU.

This document provides a high-level strategic direction. Detailed design and implementation of each advanced optimization feature would require dedicated planning.
