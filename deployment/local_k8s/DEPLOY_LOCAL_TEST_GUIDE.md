# Local Kubernetes Test Deployment Guide

This guide provides instructions to deploy the RAG system and its dependencies to a local Kubernetes cluster (e.g., Minikube, Kind, Docker Desktop Kubernetes).

## 1. Prerequisites
*   Docker, Local Kubernetes Cluster (`minikube`, `kind`, or Docker Desktop K8s)
*   `kubectl` command-line tool
*   `git` for cloning the repository
*   `helm` package manager for Kubernetes
*   (Optional) Web Browser / `curl` / Postman for testing
*   (For XLSX testing) A utility to create a simple `.xlsx` file (e.g., LibreOffice Calc, Microsoft Excel).

## 2. Setup
### 2.1. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-directory>
```
### 2.2. Start Your Local Kubernetes Cluster
*   Minikube: `minikube start --cpus 4 --memory 6144; eval $(minikube -p minikube docker-env)`
*   Kind: `kind create cluster --name rag-dev-cluster`
*   Docker Desktop: Enable Kubernetes in settings.

### 2.3. (Optional) Setup Local Docker Registry for Kind
(Assuming image loaded directly or pushed to accessible registry for brevity)

## 3. Build and Push Docker Image
1.  Navigate to repository root.
2.  Build:
    ```bash
    export DOCKER_IMAGE_NAME="rag-system"
    export DOCKER_IMAGE_TAG="iter14-local" # Use current iteration tag
    docker build -t "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" .
    # For Kind, load if not using a registry:
    # kind load docker-image "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" --name rag-dev-cluster
    ```
    Ensure `charts/rag-system/values.yaml` reflects this image name, tag, and `pullPolicy: IfNotPresent` or `Never`.

## 4. Prepare Kubernetes Secrets

### 4.1. API Gateway API Keys
1.  Create Secret (replace keys as needed):
    ```bash
    kubectl create secret generic rag-gateway-apikeys-secret \
      --from-literal=api-keys-csv="secretkey1,supersecretkey2" --namespace default
    ```
2.  Configure `charts/rag-system/values.yaml` to use this existing secret:
    ```yaml
    internalApiGateway:
      apiKeys:
        existingSecret:
          name: "rag-gateway-apikeys-secret"
          keyName: "api-keys-csv"
        createSecret: false # Ensure this is false if using an existing secret
    ```

### 4.2. (Optional) RabbitMQ Password Secret
If you enable RabbitMQ as a subchart (`rabbitmq.enabled: true` in `values.yaml`) AND enable its authentication (`rabbitmq.auth.enabled: true` - note: default in our values is `username: "user"`, `password: "password"` which Bitnami chart might use to create its own secret), you might want to use an existing secret for the password:
1.  Create the secret:
    ```bash
    kubectl create secret generic my-rabbitmq-password-secret \
      --from-literal=rabbitmq-password="yourStrongRabbitPassword" --namespace default
    ```
2.  Configure `charts/rag-system/values.yaml`:
    ```yaml
    rabbitmq:
      enabled: true # If deploying RabbitMQ via this chart
      auth:
        username: "user" # Or your desired username
        # password: "" # Comment out or leave empty if using existingPasswordSecret
        existingPasswordSecret: "my-rabbitmq-password-secret"
        # existingPasswordSecretKey: "rabbitmq-password" # Default key by Bitnami chart is often 'rabbitmq-password'
    ```

### 4.3. (Optional) Redis Password Secret
If you enable Redis as a subchart (`redis.enabled: true`) AND enable its authentication (`redis.auth.enabled: true` in `values.yaml`), you can use an existing secret:
1.  Create the secret:
    ```bash
    kubectl create secret generic my-redis-password-secret \
      --from-literal=redis-password="yourStrongRedisPassword" --namespace default
    ```
2.  Configure `charts/rag-system/values.yaml`:
    ```yaml
    redis:
      enabled: true # If deploying Redis via this chart
      auth:
        enabled: true
        # password: "" # Comment out or leave empty
        existingSecret: "my-redis-password-secret"
        existingSecretPasswordKey: "redis-password" # Common key for Bitnami Redis chart
    ```

## 5. Deploy External Dependencies
(As described in previous versions - deploy Qdrant, Ollama manually if their subcharts are not enabled. RabbitMQ & Redis covered by Helm options above if enabled.)
Ensure `helm dependency update ./charts/rag-system` or `helm dependency build ./charts/rag-system` is run if you enable subcharts.

## 6. Deploy Application Services using Helm Chart
(As before - `helm install ...`)

## 7. Verify Deployment
(As before - `kubectl get all ...`, check logs)

## 8. Access the API Gateway
(As before - determine `$GATEWAY_URL`)

## 9. Smoke Test Plan & Execution
(As before - using `$MY_API_KEY` and `$GATEWAY_URL`)

### 9.1 - 9.7 (Smoke tests for Gateway, RAG Query, Doc Proc, Metrics, RabbitMQ examples remain the same as Iteration 12/13)

## 10. Troubleshooting
(As before)
## 11. Cleanup
(As before - remember to delete manually created secrets if not managed by Helm: `kubectl delete secret my-rabbitmq-password-secret my-redis-password-secret`)
## 12. Running Automated Tests Locally
(As before)
