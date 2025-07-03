# Kubernetes Manifests / Helm Charts

This document outlines the Kubernetes manifests and Helm charts used for deploying and managing applications on Kubernetes.

**Purpose:** To define, deploy, and manage containerized applications and their dependencies within a Kubernetes cluster.

**Details:**

The primary method for deploying this application to Kubernetes is via the Helm chart located in the `/charts/rag-system` directory. Manual Kubernetes manifests are also available in `/deployment/kubernetes/` for specific or alternative deployment scenarios.

### 1. Helm Chart (`/charts/rag-system`)

*   **Purpose:** Provides a configurable and repeatable way to deploy the RAG system microservices (Internal API Gateway, RAG Query Service, Document Processing Service, Notification Service) and their dependencies (optionally Qdrant, Redis, RabbitMQ as subcharts).
*   **Key Components:**
    *   `Chart.yaml`: Defines the chart metadata and dependencies.
    *   `values.yaml`: Contains default configuration values that can be overridden by users. This includes image settings, replica counts, service types, resource requests/limits (CPU, memory, GPU), and application-specific configurations for each service.
    *   `templates/`: Contains Kubernetes manifest templates for Deployments, Services, ConfigMaps, and Secrets for each application microservice. It uses Go templating to render manifests based on values from `values.yaml`.
    *   `templates/_helpers.tpl`: Contains common Helm templating helper functions.
*   **Deployment:**
    1.  Ensure Helm is installed and your Kubernetes cluster is configured (`kubectl config current-context`).
    2.  Navigate to the repository root.
    3.  Customize deployment by creating a `my-values.yaml` file or by using `--set` flags. For example, to enable GPU for the RAG Query Service:
        ```yaml
        # my-values.yaml
        ragQueryService:
          config:
            MODEL_DEVICE: "cuda"
          gpus:
            limits:
              nvidia.com/gpu: "1"
        ```
    4.  Install or upgrade the chart:
        ```bash
        # Install
        helm install my-rag-release ./charts/rag-system -f my-values.yaml --namespace my-namespace --create-namespace

        # Upgrade
        helm upgrade my-rag-release ./charts/rag-system -f my-values.yaml --namespace my-namespace
        ```
*   **Key Configurable Values (see `values.yaml` for full list):**
    *   Image repository and tag.
    *   Replica counts.
    *   Service-specific configurations (ports, URLs, model names, logging levels).
    *   Resource requests/limits for CPU, memory, and GPUs.
    *   Enable/disable and configure subcharts for Qdrant, Redis, RabbitMQ.
    *   API keys and other secrets (preferably by referencing existing Kubernetes Secrets).

### 2. Manual Kubernetes Manifests (`/deployment/kubernetes/`)

*   **Purpose:** Provides example raw Kubernetes YAML manifests for deploying a version of the RAG pipeline and its dependencies (Qdrant, Prometheus, Grafana). These can be used for direct `kubectl apply` deployments or as a reference.
*   **Key Files:**
    *   `deployment.yaml`: Defines Deployments, Services, ConfigMaps, StatefulSets (for Qdrant), HPA for the main RAG API, and a ServiceMonitor for Prometheus.
    *   `monitoring.yaml`: Defines Deployments and Services for Prometheus and Grafana, along with their configurations.
*   **Deployment:**
    ```bash
    kubectl apply -f deployment/kubernetes/deployment.yaml
    kubectl apply -f deployment/kubernetes/monitoring.yaml
    ```
*   **Note:** These manifests might not be as frequently updated or as configurable as the Helm chart. They are good for understanding the basic K8s resource structure.

### Environment Parity

Using Helm charts with version-controlled `values.yaml` files for different environments (dev, staging, prod) is the recommended approach to ensure environment parity and manage configurations effectively. Manual manifests should be kept in sync or used primarily for development/testing setups if not managed by a more sophisticated deployment tool.
