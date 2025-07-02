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
    export DOCKER_IMAGE_TAG="iter12-local" # Use current iteration tag
    docker build -t "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" .
    # For Kind, load if not using a registry:
    # kind load docker-image "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" --name rag-dev-cluster
    ```
    Ensure `charts/rag-system/values.yaml` reflects this image name, tag, and `pullPolicy: IfNotPresent` or `Never`.

## 4. Prepare Kubernetes Secrets (Example for API Keys)
1.  Create Secret:
    ```bash
    kubectl create secret generic rag-gateway-apikeys-secret \
      --from-literal=api-keys-csv="secretkey1,supersecretkey2" --namespace default
    ```
2.  Configure `charts/rag-system/values.yaml`:
    ```yaml
    internalApiGateway:
      apiKeys:
        existingSecret:
          name: "rag-gateway-apikeys-secret"
          keyName: "api-keys-csv"
        createSecret: false
    ```

## 5. Deploy External Dependencies

This section outlines deploying dependencies. You can choose to deploy them manually using `kubectl apply` or, where available, enable them as subcharts in the `rag-system` Helm chart.

### 5.1. Qdrant
*   **Option A (Manual `kubectl apply`)**: If `qdrant.enabled: false` in Helm values.
    ```bash
    kubectl apply -f deployment/local_k8s/dependencies/qdrant-deployment.yaml
    kubectl apply -f deployment/local_k8s/dependencies/qdrant-service.yaml
    ```
*   **Option B (Helm Subchart)**: If `qdrant.enabled: true` in Helm values.
    Run `helm dependency update ./charts/rag-system` first. Helm will deploy Qdrant.

### 5.2. Ollama
**IMPORTANT**: The `rag-system` Helm chart **does not deploy Ollama itself**. You must deploy Ollama separately.
1.  **Deploy Ollama Manually**:
    ```bash
    kubectl apply -f deployment/local_k8s/dependencies/ollama-statefulset.yaml
    kubectl apply -f deployment/local_k8s/dependencies/ollama-service.yaml
    ```
2.  **Wait for Ollama to be ready**: `kubectl get statefulset ollama -w && kubectl get pods -l app=ollama -w`
3.  **Pull a model into Ollama**:
    ```bash
    OLLAMA_POD=$(kubectl get pods -l app=ollama -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -it $OLLAMA_POD -- ollama pull llama2 # Or your desired model
    ```
4.  **Configure `charts/rag-system/values.yaml` for your Ollama service**:
    Ensure the `ollama.service.name` (e.g., "ollama-service") and `ollama.service.port` (e.g., 11434) in `values.yaml` match your Ollama Kubernetes service details. The RAG Query Service uses these values to connect.

### 5.3. RabbitMQ
*   **Option A (Manual `kubectl apply`)**: If `rabbitmq.enabled: false` in Helm values.
    ```bash
    kubectl apply -f deployment/local_k8s/dependencies/rabbitmq-deployment.yaml
    kubectl apply -f deployment/local_k8s/dependencies/rabbitmq-service.yaml
    ```
*   **Option B (Helm Subchart)**: If `rabbitmq.enabled: true` in Helm values.
    Run `helm dependency update ./charts/rag-system` first. Helm will deploy RabbitMQ. Configure credentials in `values.yaml` under `rabbitmq.auth`.

### 5.4. Redis
*   **Option A (Manual `kubectl apply`)**: If `redis.enabled: false` in Helm values.
    ```bash
    kubectl apply -f deployment/local_k8s/dependencies/redis-deployment.yaml
    kubectl apply -f deployment/local_k8s/dependencies/redis-service.yaml
    ```
*   **Option B (Helm Subchart)**: If `redis.enabled: true` in Helm values.
    Run `helm dependency update ./charts/rag-system` first. Helm will deploy Redis.

Wait for all chosen dependencies to be ready before proceeding.

## 6. Deploy Application Services using Helm Chart
1.  Navigate to repository root.
2.  Ensure `charts/rag-system/values.yaml` is configured (image, secrets, dependency URLs if external and subcharts disabled).
3.  Install/Upgrade:
    ```bash
    helm install my-rag-instance ./charts/rag-system -f ./charts/rag-system/values.yaml --namespace default
    # Or: helm upgrade ...
    ```

## 7. Verify Deployment
```bash
kubectl get all -l app.kubernetes.io/instance=my-rag-instance -n default -w
# Check logs...
```

## 8. Access the API Gateway
(Determine `$GATEWAY_URL` via NodePort and Minikube IP/localhost)
```bash
# Example for Docker Desktop/Kind:
# NODE_PORT_GATEWAY=$(kubectl get service my-rag-instance-internal-api-gateway -n default -o jsonpath='{.spec.ports[0].nodePort}')
# export GATEWAY_URL="http://localhost:$NODE_PORT_GATEWAY"
# echo "Gateway URL: $GATEWAY_URL"
```

## 9. Smoke Test Plan & Execution
```bash
export MY_API_KEY="secretkey1"
# Ensure GATEWAY_URL is set
```
### 9.1. Gateway Health Check
`curl $GATEWAY_URL/gateway_health`
### 9.2. RAG Query Service Health
`curl -H "X-API-Key: $MY_API_KEY" $GATEWAY_URL/rag/health`
### 9.3. Document Processing Service Health
`curl -H "X-API-Key: $MY_API_KEY" $GATEWAY_URL/document/health`

### 9.4. Test Document Processing (TXT, PDF, DOCX, PPTX, XLSX)
1.  Create `testdoc.txt`: `echo "Text processing by Jules is fine." > testdoc.txt`
2.  Create `testdoc.pdf` (content: "PDF processing by Jules is good.")
3.  Create `testdoc.docx` (content: "DOCX processing by Jules is also good.")
4.  Create `testdoc.pptx` (content on a slide: "PPTX processing by Jules works.")
5.  Create `testdoc.xlsx` (cell A1: "XLSX processing by Jules is okay.", cell B1: "Excel figures are important.")

    ```bash
    curl -X POST "$GATEWAY_URL/document/process_document" -H "X-API-Key: $MY_API_KEY" -F "file=@testdoc.txt" -F "metadata_json={\"source\":\"e2e_txt_v12\"}"
    curl -X POST "$GATEWAY_URL/document/process_document" -H "X-API-Key: $MY_API_KEY" -F "file=@testdoc.pdf" -F "metadata_json={\"source\":\"e2e_pdf_v12\"}"
    curl -X POST "$GATEWAY_URL/document/process_document" -H "X-API-Key: $MY_API_KEY" -F "file=@testdoc.docx" -F "metadata_json={\"source\":\"e2e_docx_v12\"}"
    curl -X POST "$GATEWAY_URL/document/process_document" -H "X-API-Key: $MY_API_KEY" -F "file=@testdoc.pptx" -F "metadata_json={\"source\":\"e2e_pptx_v12\"}"
    curl -X POST "$GATEWAY_URL/document/process_document" -H "X-API-Key: $MY_API_KEY" -F "file=@testdoc.xlsx" -F "metadata_json={\"source\":\"e2e_xlsx_v12\"}"
    ```
    Check `doc-processing-service` logs.

### 9.5. Test RAG Query Service (Querying Processed Documents)
Wait a few seconds for indexing.
```bash
curl -X POST "$GATEWAY_URL/rag/query" -H "Content-Type: application/json" -H "X-API-Key: $MY_API_KEY" \
  -d '{"query": "status of text processing by Jules", "top_k": 1, "generate_answer": true}'
curl -X POST "$GATEWAY_URL/rag/query" -H "Content-Type: application/json" -H "X-API-Key: $MY_API_KEY" \
  -d '{"query": "status of PDF processing by Jules", "top_k": 1, "generate_answer": true}'
curl -X POST "$GATEWAY_URL/rag/query" -H "Content-Type: application/json" -H "X-API-Key: $MY_API_KEY" \
  -d '{"query": "status of DOCX processing by Jules", "top_k": 1, "generate_answer": true}'
curl -X POST "$GATEWAY_URL/rag/query" -H "Content-Type: application/json" -H "X-API-Key: $MY_API_KEY" \
  -d '{"query": "status of PPTX processing by Jules", "top_k": 1, "generate_answer": true}'
curl -X POST "$GATEWAY_URL/rag/query" -H "Content-Type: application/json" -H "X-API-Key: $MY_API_KEY" \
  -d '{"query": "status of XLSX processing by Jules", "top_k": 1, "generate_answer": true}'
```
Observe responses.

### 9.6. Check RAG Query Service Metrics (Conceptual)
(Port-forward RAG service and curl `/metrics`)

### 9.7. Test RabbitMQ Producer/Consumer Examples (Manual - if deployed)
(As before)

## 10. Troubleshooting
(As before)
## 11. Cleanup
(As before)
## 12. Running Automated Tests Locally
(As before)
