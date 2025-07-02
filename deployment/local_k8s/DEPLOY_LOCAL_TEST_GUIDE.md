# Local Kubernetes Test Deployment Guide

This guide provides instructions to deploy the RAG system and its dependencies to a local Kubernetes cluster (e.g., Minikube, Kind, Docker Desktop Kubernetes).

## 1. Prerequisites
*   Docker, Local Kubernetes Cluster (`minikube`, `kind`, or Docker Desktop K8s)
*   `kubectl` command-line tool
*   `git` for cloning the repository
*   `helm` package manager for Kubernetes
*   (Optional) Web Browser / `curl` / Postman for testing

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
(See previous guide versions if needed; for brevity, assuming image loaded directly or pushed to accessible registry)

## 3. Build and Push Docker Image
1.  Navigate to repository root.
2.  Build:
    ```bash
    # For Minikube with `eval $(minikube docker-env)` or direct load to Kind:
    export DOCKER_IMAGE_NAME="rag-system"
    export DOCKER_IMAGE_TAG="iter11-local" # Use current iteration tag
    docker build -t "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" .
    # For Kind, load if not using a registry:
    # kind load docker-image "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" --name rag-dev-cluster
    ```
    Ensure `charts/rag-system/values.yaml` reflects this image name, tag, and `pullPolicy: IfNotPresent` or `Never`.
    (For other registries, build, tag fully, push, and update `values.yaml`.)

## 4. Prepare Kubernetes Secrets (Example for API Keys)
1.  Create Secret (replace keys as needed):
    ```bash
    kubectl create secret generic rag-gateway-apikeys-secret \
      --from-literal=api-keys-csv="secretkey1,supersecretkey2" \
      --namespace default
    ```
2.  Configure `charts/rag-system/values.yaml` to use it:
    ```yaml
    internalApiGateway:
      apiKeys:
        existingSecret:
          name: "rag-gateway-apikeys-secret"
          keyName: "api-keys-csv"
        createSecret: false
    ```

## 5. Deploy External Dependencies (if not managed by Helm chart)
Refer to `charts/rag-system/values.yaml` for `<dependency>.enabled` flags.
*   If enabled, run `helm dependency update ./charts/rag-system` or `helm dependency build ./charts/rag-system` before installing the main chart. Helm will deploy them.
*   If disabled, deploy manually, e.g.: `kubectl apply -f deployment/local_k8s/dependencies/`
    (Ensure Qdrant, Ollama, RabbitMQ, Redis are up. For Ollama, pull a model like `llama2`).

## 6. Deploy Application Services using Helm Chart
1.  Navigate to repository root.
2.  Ensure `charts/rag-system/values.yaml` is configured (image, secrets, dependency URLs if external).
3.  Install/Upgrade:
    ```bash
    helm install my-rag-instance ./charts/rag-system -f ./charts/rag-system/values.yaml --namespace default
    # Or for upgrades:
    # helm upgrade my-rag-instance ./charts/rag-system -f ./charts/rag-system/values.yaml --namespace default
    ```

## 7. Verify Deployment
```bash
kubectl get all -l app.kubernetes.io/instance=my-rag-instance -n default -w
# Check logs: kubectl logs deployment/my-rag-instance-internal-api-gateway -c internal-api-gateway -n default -f (etc.)
```

## 8. Access the API Gateway
(As per Helm chart `NOTES.txt` or previous guide versions - get NodePort and Minikube IP / localhost)
```bash
# Example for Minikube:
# MINIKUBE_IP=$(minikube ip)
# NODE_PORT_GATEWAY=$(kubectl get service my-rag-instance-internal-api-gateway -n default -o jsonpath='{.spec.ports[0].nodePort}')
# export GATEWAY_URL="http://$MINIKUBE_IP:$NODE_PORT_GATEWAY"
# echo "Gateway URL: $GATEWAY_URL"
# For Docker Desktop/Kind, GATEWAY_URL is likely http://localhost:<NODE_PORT_GATEWAY>
```

## 9. Smoke Test Plan & Execution
Use a valid API key from your secret (e.g., `secretkey1`).
```bash
export MY_API_KEY="secretkey1"
# Ensure GATEWAY_URL is set from step 8
```

### 9.1. Gateway Health Check (Unprotected)
`curl $GATEWAY_URL/gateway_health` (Expected: `{"status":"healthy",...}`)

### 9.2. RAG Query Service Health (Protected)
`curl -H "X-API-Key: $MY_API_KEY" $GATEWAY_URL/rag/health` (Expected: `{"status":"healthy", ...}` or `"degraded"`)

### 9.3. Document Processing Service Health (Protected)
`curl -H "X-API-Key: $MY_API_KEY" $GATEWAY_URL/document/health` (Expected: `{"status":"healthy", ...}`)

### 9.4. Test Document Processing (TXT, PDF, DOCX, PPTX - Protected)
1.  Create `testdoc.txt`: `echo "Text documents are processed well by Jules." > testdoc.txt`
2.  Create `testdoc.pdf` (e.g., save "PDF documents provide portable views for Jules." as PDF).
3.  Create `testdoc.docx` (e.g., save "DOCX files are also handled by Jules." as DOCX).
4.  Create `testdoc.pptx` (e.g., save "PPTX presentations are parsed for text by Jules." on a slide).

    ```bash
    curl -X POST "$GATEWAY_URL/document/process_document" -H "X-API-Key: $MY_API_KEY" -F "file=@testdoc.txt" -F "metadata_json={\"source\":\"e2e_txt_v11\"}"
    curl -X POST "$GATEWAY_URL/document/process_document" -H "X-API-Key: $MY_API_KEY" -F "file=@testdoc.pdf" -F "metadata_json={\"source\":\"e2e_pdf_v11\"}"
    curl -X POST "$GATEWAY_URL/document/process_document" -H "X-API-Key: $MY_API_KEY" -F "file=@testdoc.docx" -F "metadata_json={\"source\":\"e2e_docx_v11\"}"
    curl -X POST "$GATEWAY_URL/document/process_document" -H "X-API-Key: $MY_API_KEY" -F "file=@testdoc.pptx" -F "metadata_json={\"source\":\"e2e_pptx_v11\"}"
    ```
    Check `doc-processing-service` logs for successful indexing messages for all types.

### 9.5. Test RAG Query Service (Hybrid Search, LLM, Cache - Protected)
Wait a few seconds for indexing.
Query for content from each document type:
```bash
# Query for TXT content
curl -X POST "$GATEWAY_URL/rag/query" -H "Content-Type: application/json" -H "X-API-Key: $MY_API_KEY" \
  -d '{"query": "How are text documents processed?", "top_k": 1, "generate_answer": true}'
# Query for PDF content
curl -X POST "$GATEWAY_URL/rag/query" -H "Content-Type: application/json" -H "X-API-Key: $MY_API_KEY" \
  -d '{"query": "What do PDF documents provide?", "top_k": 1, "generate_answer": true}'
# Query for DOCX content
curl -X POST "$GATEWAY_URL/rag/query" -H "Content-Type: application/json" -H "X-API-Key: $MY_API_KEY" \
  -d '{"query": "Which files are also handled by Jules?", "top_k": 1, "generate_answer": true}'
# Query for PPTX content
curl -X POST "$GATEWAY_URL/rag/query" -H "Content-Type: application/json" -H "X-API-Key: $MY_API_KEY" \
  -d '{"query": "How are PPTX presentations parsed?", "top_k": 1, "generate_answer": true}'
```
Observe responses. Check RAG service logs for cache behavior (run a query twice).

### 9.6. Check RAG Query Service Metrics (Conceptual)
(Port-forward to RAG service pod on port 8001 and curl `http://localhost:8001/metrics` as described previously).

### 9.7. Test RabbitMQ Producer/Consumer Examples (Manual - if deployed)
(As described previously).

## 10. Troubleshooting
(As before)

## 11. Cleanup
(As before, including `helm uninstall` and manual secret deletion if created).

## 12. Running Automated Tests Locally
(As before)
