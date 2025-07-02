# Local Kubernetes Test Deployment Guide (Iteration 1)

This guide provides instructions to deploy the RAG system (Iteration 1: API Gateway, RAG Query Service, Document Processing Stub) to a local Kubernetes cluster (e.g., Minikube, Kind, Docker Desktop Kubernetes).

## 1. Prerequisites

*   **Docker**: Installed and running. (Ensure it has enough resources allocated, e.g., 4GB+ RAM, 2+ CPUs).
*   **Local Kubernetes Cluster**:
    *   **Minikube**: [Install Minikube](https://minikube.sigs.k8s.io/docs/start/)
    *   **Kind**: [Install Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
    *   **Docker Desktop Kubernetes**: Enable Kubernetes in Docker Desktop settings.
*   **`kubectl`**: Kubernetes command-line tool, configured to point to your local cluster. [Install kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/).
*   **`git`**: For cloning the repository.
*   **(Optional) `helm`**: If you choose to deploy dependencies using Helm. [Install Helm](https://helm.sh/docs/intro/install/).
*   **(Optional) Web Browser / `curl` / Postman**: For testing API endpoints.

## 2. Setup

### 2.1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 2.2. Start Your Local Kubernetes Cluster

*   **Minikube**:
    ```bash
    minikube start --cpus 4 --memory 4096 # Adjust resources as needed
    minikube addons enable ingress # Optional, for more advanced routing later
    eval $(minikube -p minikube docker-env) # To use Minikube's Docker daemon (important!)
    ```
*   **Kind**:
    ```bash
    kind create cluster --name rag-dev-cluster
    # For Kind, you'll need to load images into the cluster nodes (see image building step).
    ```
*   **Docker Desktop**: Ensure Kubernetes is enabled and running.

### 2.3. (Optional) Setup Local Docker Registry
For Kind or other clusters that don't share Docker Desktop's image cache, a local registry can be helpful:
```bash
# Run a local Docker registry
docker run -d -p 5000:5000 --restart=always --name kind-registry registry:2

# For Kind, connect the registry to the Kind network (if not already configured)
# See Kind documentation for local registry setup: https://kind.sigs.k8s.io/docs/user/local-registry/
```
If using a local registry like `localhost:5000`, you'll need to tag your images with `localhost:5000/your-image-name:tag` and push to it. Update the `image:` fields in Kubernetes deployments accordingly.

## 3. Build and Push Docker Image

1.  **Navigate to the repository root directory.**
2.  **Build the Docker image**:
    ```bash
    # Replace 'your-dockerhub-username/rag-system' with your Docker Hub username or other registry path.
    # If using Minikube's Docker daemon (after `eval $(minikube docker-env)`),
    # you can use a simple name like 'rag-system' as it builds directly into Minikube's context.
    export DOCKER_IMAGE_NAME="your-dockerhub-username/rag-system" # Or "rag-system" for Minikube direct build
    export DOCKER_IMAGE_TAG="iter1-local"

    docker build -t "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" .
    ```
3.  **Push the image (if not using Minikube's Docker daemon directly)**:
    *   If using Docker Hub:
        ```bash
        docker login
        docker push "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
        ```
    *   If using Kind with a local registry (e.g., `localhost:5000`):
        ```bash
        export DOCKER_IMAGE_NAME="localhost:5000/rag-system" # Example for local registry
        docker build -t "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" .
        docker push "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
        ```
    *   If using Kind without a local registry, load the image directly:
        ```bash
        # Build with a simple name first
        docker build -t "rag-system:iter1-local" .
        kind load docker-image "rag-system:iter1-local" --name rag-dev-cluster
        # Then ensure your Kubernetes manifests use "rag-system:iter1-local"
        ```
4.  **Update Kubernetes Manifests**:
    Open the files in `deployment/local_k8s/`:
    *   `internal-api-gateway-deployment.yaml`
    *   `rag-query-service-deployment.yaml`
    *   `doc-processing-service-deployment.yaml`
    And change `image: your-repo/rag-system:latest` to the image name and tag you just built and pushed (e.g., `your-dockerhub-username/rag-system:iter1-local` or `localhost:5000/rag-system:iter1-local` or `rag-system:iter1-local` if loaded directly to Kind/Minikube).

## 4. Deploy External Dependencies

For Iteration 1, the RAG Query Service requires Qdrant. PostgreSQL and MongoDB are planned for future use by the connectors but not strictly required by the current stub services.

### 4.1. Deploy Qdrant

You can deploy Qdrant using Docker directly (if your K8s can reach it, less common for local clusters) or via Kubernetes manifests/Helm.

**Option A: Using Kubernetes Manifest (Simple)**
Create `deployment/local_k8s/dependencies/qdrant-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant-db
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qdrant-db
  template:
    metadata:
      labels:
        app: qdrant-db
    spec:
      containers:
      - name: qdrant-container
        image: qdrant/qdrant:latest # Use a specific version for stability if preferred
        ports:
        - containerPort: 6333 # HTTP
          name: http
        - containerPort: 6334 # gRPC
          name: grpc
        # Add volume for persistence if needed for testing data continuity
        # volumeMounts:
        # - name: qdrant-storage
        #   mountPath: /qdrant/storage
      # volumes:
      #   - name: qdrant-storage
      #     emptyDir: {} # For local testing, data lost if pod dies. Use PersistentVolume for real persistence.
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant-service # This is the name used in rag-query-service-configmap.yaml (QDRANT_HOST)
spec:
  selector:
    app: qdrant-db
  ports:
  - name: http
    port: 6333
    targetPort: 6333
  - name: grpc
    port: 6334
    targetPort: 6334
  type: ClusterIP # Only needs to be accessible within the cluster
```
Apply it:
```bash
kubectl apply -f deployment/local_k8s/dependencies/qdrant-deployment.yaml # Ensure service name is qdrant-service
kubectl apply -f deployment/local_k8s/dependencies/qdrant-service.yaml # If you created a separate service file
```

**Option B: Helm Subchart Deployment (if `qdrant.enabled` is `true` in Helm `values.yaml`)**
   If you set `qdrant.enabled: true` in your `charts/rag-system/values.yaml` (or via `--set qdrant.enabled=true`), Helm will deploy Qdrant when you install the `rag-system` chart.
   The RAG Query Service and Document Processing Service configurations in `values.yaml` are set up to point to the Helm-deployed Qdrant service name (`{{ .Release.Name }}-qdrant`).
   No separate `kubectl apply` or `helm install qdrant/...` is needed for Qdrant in this case.
   You might need to run `helm dependency update ./charts/rag-system` once before installing if you've just added the dependency to `Chart.yaml`.

### 4.2. Deploy Ollama

Ollama is used by the RAG Query Service to generate answers.

1.  **Apply the Ollama StatefulSet and Service manifests**:
    ```bash
    kubectl apply -f deployment/local_k8s/dependencies/ollama-statefulset.yaml
    kubectl apply -f deployment/local_k8s/dependencies/ollama-service.yaml
    ```

2.  **Wait for Ollama to be ready**:
    ```bash
    kubectl get statefulset ollama -w
    kubectl get pods -l app=ollama -w
    # Wait until the ollama-0 pod is Running and Ready (1/1).
    ```
    The first time, it might take a bit longer if a PersistentVolumeClaim needs to be provisioned and bound. If the PVC remains `Pending`, you might need to configure a default `StorageClass` in your local Kubernetes cluster or specify an existing one in `ollama-statefulset.yaml`. For Minikube, `standard` often works. For Kind, you might need to set up a local path provisioner or use `hostPath` volumes for simpler local persistence if model persistence is key. If model persistence isn't critical for every local test run, you can change `volumeClaimTemplates` to an `emptyDir` volume in `ollama-statefulset.yaml`.

3.  **Pull a model into Ollama**:
    Once the `ollama-0` pod is running, you need to pull a model for the RAG service to use. The default in `rag-query-service-configmap.yaml` will be `llama2` (or similar).
    ```bash
    # Find the Ollama pod name (usually ollama-0)
    OLLAMA_POD=$(kubectl get pods -l app=ollama -o jsonpath='{.items[0].metadata.name}')
    echo "Ollama pod: $OLLAMA_POD"

    # Pull the model (e.g., llama2). This might take some time.
    kubectl exec -it $OLLAMA_POD -- ollama pull llama2
    # You can replace 'llama2' with another model like 'mistral' if preferred,
    # but ensure RAG_QUERY_SERVICE_LLM_MODEL_NAME is updated accordingly.
    ```
    You can check available models with `kubectl exec -it $OLLAMA_POD -- ollama list`.

### 4.3. Deploy RabbitMQ

RabbitMQ can be used for asynchronous task processing between services.

1.  **Apply the RabbitMQ Deployment and Service manifests**:
    ```bash
    kubectl apply -f deployment/local_k8s/dependencies/rabbitmq-deployment.yaml
    kubectl apply -f deployment/local_k8s/dependencies/rabbitmq-service.yaml
    ```

2.  **Wait for RabbitMQ to be ready**:
    ```bash
    kubectl get deployment rabbitmq -w
    kubectl get pods -l app=rabbitmq -w
    # Wait until the rabbitmq pod is Running and Ready (1/1).
    ```

3.  **(Optional) Access RabbitMQ Management UI**:
    The service `rabbitmq-service` exposes port `15672` for the management UI. To access it locally:
    ```bash
    # Find the RabbitMQ pod name
    RABBITMQ_POD=$(kubectl get pods -l app=rabbitmq -o jsonpath='{.items[0].metadata.name}')
    echo "RabbitMQ pod: $RABBITMQ_POD"

    # Port-forward to the management UI
    echo "Port-forwarding RabbitMQ management UI. Access at http://localhost:15672. Press Ctrl+C to stop."
    kubectl port-forward $RABBITMQ_POD 15672:15672
    ```
    Open `http://localhost:15672` in your browser. Login with the credentials defined in `rabbitmq-deployment.yaml` (default in example: `user` / `password`).

### 4.4. Deploy Redis (for Caching) - *Now potentially managed by Helm*

Redis is used by the RAG Query Service for caching.

**Option A: Manual Deployment (if `redis.enabled` is `false` in Helm `values.yaml`)**
1.  Apply the Redis Deployment and Service manifests:
    ```bash
    kubectl apply -f deployment/local_k8s/dependencies/redis-deployment.yaml
    kubectl apply -f deployment/local_k8s/dependencies/redis-service.yaml
    ```
2.  Wait for Redis to be ready (as described before).

**Option B: Helm Subchart Deployment (if `redis.enabled` is `true` in Helm `values.yaml`)**
   If you set `redis.enabled: true` in your `charts/rag-system/values.yaml` (or via `--set redis.enabled=true`), Helm will deploy Redis when you install the `rag-system` chart.
   The RAG Query Service configuration in `values.yaml` is set up to point to the Helm-deployed Redis service name (`{{ .Release.Name }}-redis-master`).
   No separate `kubectl apply` is needed for Redis in this case.

### 4.5. (Future) Deploy PostgreSQL & MongoDB - *Helm Subcharts Recommended*
Placeholder for when these are needed. You would typically use Helm charts, potentially as subcharts to the main `rag-system` chart in a similar way to Qdrant and Redis.

## 5. Configure Services to Find Dependencies

The `rag-query-service` needs to know the URL for the Ollama API. This is defined in its ConfigMap.
Ensure `deployment/local_k8s/rag-query-service-configmap.yaml` has:
```yaml
data:
  # ... other configs ...
  OLLAMA_API_URL: "http://ollama-service:11434" # Points to the K8s service for Ollama
  LLM_MODEL_NAME: "llama2" # Or the model you pulled, e.g., "mistral"
  # ...
```
This should already be set if you are applying the latest version of the ConfigMap.

## 6. Deploy Application Services (Option A: Using Raw Kubernetes Manifests)

If you prefer to deploy using the individual YAML files (e.g., for deeper inspection or if not using Helm):

```bash
kubectl apply -f deployment/local_k8s/internal-api-gateway-configmap.yaml
kubectl apply -f deployment/local_k8s/internal-api-gateway-deployment.yaml
kubectl apply -f deployment/local_k8s/internal-api-gateway-service.yaml

kubectl apply -f deployment/local_k8s/rag-query-service-configmap.yaml
kubectl apply -f deployment/local_k8s/rag-query-service-deployment.yaml
kubectl apply -f deployment/local_k8s/rag-query-service-service.yaml

kubectl apply -f deployment/local_k8s/doc-processing-service-configmap.yaml
kubectl apply -f deployment/local_k8s/doc-processing-service-deployment.yaml
kubectl apply -f deployment/local_k8s/doc-processing-service-service.yaml
```
Or apply all service-specific YAMLs at once (ensure dependencies like ConfigMaps are created before Deployments if not using `kubectl apply -k` or similar which handles ordering):
```bash
# Apply ConfigMaps first
kubectl apply -f deployment/local_k8s/internal-api-gateway-configmap.yaml
kubectl apply -f deployment/local_k8s/rag-query-service-configmap.yaml
kubectl apply -f deployment/local_k8s/doc-processing-service-configmap.yaml

# Then Deployments and Services
kubectl apply -f deployment/local_k8s/internal-api-gateway-deployment.yaml
kubectl apply -f deployment/local_k8s/internal-api-gateway-service.yaml
kubectl apply -f deployment/local_k8s/rag-query-service-deployment.yaml
kubectl apply -f deployment/local_k8s/rag-query-service-service.yaml
kubectl apply -f deployment/local_k8s/doc-processing-service-deployment.yaml
kubectl apply -f deployment/local_k8s/doc-processing-service-service.yaml
```

## 6. Deploy Application Services (Option B: Using Helm Chart - Recommended)

This is the recommended method for deploying the core RAG system services.

1.  **Navigate to the repository root.**
2.  **Update `charts/rag-system/values.yaml` (Important!):**
    *   Open `charts/rag-system/values.yaml`.
    *   Change `image.repository` to your actual Docker image repository (e.g., `yourdockerhubusername/rag-system` or `localhost:5000/rag-system` if using a local registry).
    *   Change `image.tag` to the tag you used during the `docker build` step (e.g., `iter7-local` or your specific tag).
    *   Review other default values (ports, resources, dependency service names like `QDRANT_HOST`, `OLLAMA_API_URL`, `REDIS_HOST`) and adjust if your dependency deployments use different names or your local K8s environment has specific needs.

3.  **Install the Helm chart:**
    Give your deployment a release name, e.g., `my-rag-instance`.
    ```bash
    helm install my-rag-instance ./charts/rag-system -f ./charts/rag-system/values.yaml --namespace default
    # Or, if you want to override specific values without modifying values.yaml:
    # helm install my-rag-instance ./charts/rag-system \
    #   --set image.repository="yourdockerhubusername/rag-system" \
    #   --set image.tag="your-tag" \
    #   --namespace default
    ```
    The output will include `NOTES.txt` with information on how to access the services.

4.  **To upgrade an existing Helm release:**
    ```bash
    helm upgrade my-rag-instance ./charts/rag-system -f ./charts/rag-system/values.yaml --namespace default
    ```

5.  **To uninstall a Helm release:**
    ```bash
    helm uninstall my-rag-instance --namespace default
    ```

**Note on Dependencies**: This Helm chart only deploys the core RAG application services. Dependencies like Qdrant, Ollama, RabbitMQ, and Redis must still be deployed separately (e.g., using `kubectl apply -f deployment/local_k8s/dependencies/`) as described in Section 4. Ensure they are running *before* installing the Helm chart.

## 7. Verify Deployment

Check the status of your pods, services, and deployments:

```bash
kubectl get pods -w # Watch pods until they are Running
kubectl get services
kubectl get deployments
```

If pods are in `ImagePullBackOff` or `ErrImagePull`, ensure your image name is correct in the deployment YAMLs and that it's accessible to your Kubernetes cluster (pushed to the right registry, or loaded into Kind/Minikube).

Check logs for each service:
```bash
kubectl logs deployment/internal-api-gateway -c gateway-container -f
kubectl logs deployment/rag-query-service -c rag-query-container -f
kubectl logs deployment/doc-processing-service -c doc-processing-container -f
# For Qdrant if deployed via manifest:
kubectl logs deployment/qdrant-db -c qdrant-container -f
```
Look for successful startup messages and any error messages.

## 7. Access the API Gateway

The Internal API Gateway service is exposed via `NodePort`. Find its external port:

```bash
kubectl get service internal-api-gateway
```
Look for the `PORT(S)` column, e.g., `80:30080/TCP`. The `30080` is the NodePort.

If using Minikube, get the Minikube IP:
```bash
minikube ip
```
The gateway will be accessible at `http://<minikube-ip>:<NodePort>`.

If using Kind or Docker Desktop, it's usually `http://localhost:<NodePort>`.

## 8. Smoke Test Plan & Execution

Perform these tests using `curl` or Postman. Replace `<gateway-url>` with the URL found in the previous step (e.g., `http://localhost:30080`).
You will also need a valid API key. The default example keys in the ConfigMap/values.yaml are `changeme-local-key1` or `local-raw-key1`. Use one of these or one you configured.

**Set your API Key as an environment variable for convenience:**
```bash
export MY_API_KEY="changeme-local-key1" # Or your configured key
```

### 8.1. Gateway Health Check
(This endpoint is typically not protected by API key for basic liveness/readiness)
```bash
curl <gateway-url>/gateway_health
```
Expected: `{"status":"healthy","service":"Internal API Gateway"}`

### 8.2. RAG Query Service Health (via Gateway - Protected)
This assumes the RAG Query Service's `/health` endpoint is at its root. The gateway forwards `/rag/health`.
```bash
curl -H "X-API-Key: $MY_API_KEY" <gateway-url>/rag/health
```
Expected: `{"status":"healthy", ...}` or `{"status":"degraded", ...}` if components within RAG service aren't fully ready. Check RAG service logs.

### 8.3. Document Processing Service Health (via Gateway - Protected)
```bash
curl -H "X-API-Key: $MY_API_KEY" <gateway-url>/document/health
```
Expected: `{"status":"healthy", ...}` (details depend on service state)

### 8.4. Test Document Processing Service (via Gateway - Protected)
Create a dummy text file, e.g., `sample.txt` with "Hello world".
```bash
curl -X POST "<gateway-url>/document/process_document" \
  -H "X-API-Key: $MY_API_KEY" \
  -F "file=@sample.txt" \
  -F "metadata_json={\"source\":\"local_test_auth\", \"user\":\"tester_auth\"}"
```
Expected:
```json
{
  "message": "Document received for processing (stub implementation).",
  "filename": "sample.txt",
  "metadata_received": {
    "source": "local_test",
    "user": "tester"
  },
  "status": "received_by_stub"
}
```
Check logs of `doc-processing-service` to see the "Document received" message.

### 8.5. Test RAG Query Service (LLM Integration - via Gateway)
**Note**: This test requires Ollama and Qdrant to be running. The RAG Query Service will use the LLM model specified in its config (default `llama2`) and the Qdrant collection (default `documents`). Ensure the `llama2` (or your configured) model is pulled in Ollama.

This test checks if the service can retrieve from Qdrant (even if empty) and generate an answer using the LLM.
```bash
curl -X POST "<gateway-url>/rag/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $MY_API_KEY" \
  -d '{"query": "What is the capital of France?", "top_k": 1, "generate_answer": true}'
```
Expected (if Qdrant is up, collection exists (even if empty), and Ollama with model is running):
```json
{
  "query": "What is the capital of France?",
  "search_results": [], // Or some results if you've indexed data
  "answer": "The capital of France is Paris.", // Or similar LLM response, might vary. If no context, it might answer from its general knowledge or state it couldn't find info.
  "llm_model_used": "llama2" // Or your configured model
}
```
Check RAG service logs for details of Qdrant interaction and Ollama calls. If you get an error or "issue generating answer", check Ollama pod logs and RAG service logs.

### 8.6. End-to-End Test: Process and Query Document

This test verifies the basic document processing and RAG query flow.

1.  **Create a simple text file `testdoc.txt`**:
    ```
    echo "Jules the AI agent enjoys software engineering and helping users." > testdoc.txt
    ```

2.  **Process `testdoc.txt` using the Document Processing Service (via Gateway)**:
    ```bash
    curl -X POST "<gateway-url>/document/process_document" \
      -H "X-API-Key: $MY_API_KEY" \
      -F "file=@testdoc.txt" \
      -F "metadata_json={\"source\":\"e2e_txt_test_auth\", \"doc_title\":\"Jules AI Agent TXT Auth\"}"
    ```
    Expected response should indicate successful indexing, e.g.:
    ```json
    {
      "message": "Document processed and indexed successfully.",
      "filename": "testdoc.txt",
      "qdrant_id": "some-uuid-or-custom-id", // The ID used in Qdrant
      "metadata_processed": {
        "source": "e2e_txt_test",
        "doc_title": "Jules AI Agent TXT",
        "original_filename": "testdoc.txt",
        "_internal_id": "some-uuid-or-custom-id"
      },
      "status": "indexed"
    }
    ```
    Check `doc-processing-service` logs for confirmation of embedding and Qdrant upsert.

3.  **Create a simple PDF file `testdoc.pdf`**:
    You can create one using any word processor and saving as PDF, or using a simple online converter with the text: "The quick brown fox jumps over the lazy dog."

4.  **Process `testdoc.pdf` using the Document Processing Service (via Gateway)**:
    ```bash
    curl -X POST "<gateway-url>/document/process_document" \
      -H "X-API-Key: $MY_API_KEY" \
      -F "file=@testdoc.pdf" \
      -F "metadata_json={\"source\":\"e2e_pdf_test_auth\", \"doc_title\":\"Lazy Fox PDF Auth\"}"
    ```
    Expected response should indicate successful indexing. Check `doc-processing-service` logs.

5.  **Create a simple DOCX file `testdoc.docx`**:
    Create a DOCX file (e.g., using Word, LibreOffice Writer) with the content: "DOCX files are processed by Jules." Save it as `testdoc.docx`.

6.  **Process `testdoc.docx` using the Document Processing Service (via Gateway)**:
    ```bash
    curl -X POST "<gateway-url>/document/process_document" \
      -H "X-API-Key: $MY_API_KEY" \
      -F "file=@testdoc.docx" \
      -F "metadata_json={\"source\":\"e2e_docx_test_auth\", \"doc_title\":\"Jules DOCX Test Auth\"}"
    ```
    Expected response should indicate successful indexing. Check `doc-processing-service` logs.

7.  **Wait a few seconds for indexing to settle.**

8.  **Query the RAG Service for content from the processed TXT document (via Gateway)**:
    ```bash
    curl -X POST "<gateway-url>/rag/query" \
      -H "Content-Type: application/json" \
      -H "X-API-Key: $MY_API_KEY" \
      -d '{"query": "What does Jules the AI agent enjoy?", "top_k": 1, "generate_answer": true}'
    ```
    Expected response (will vary based on LLM and exact context):
    ```json
    {
      "query": "What does Jules the AI agent enjoy?",
      "search_results": [
        {
          // ... details of testdoc.txt ...
          "text": "Jules the AI agent enjoys software engineering and helping users.",
          "metadata": {
            "source": "e2e_txt_test",
            // ... other metadata ...
          }
        }
      ],
      "answer": "Jules the AI agent enjoys software engineering and helping users.",
      "llm_model_used": "llama2"
    }
    ```

7.  **Query the RAG Service for content from the processed PDF document (via Gateway)**:
    ```bash
    curl -X POST "<gateway-url>/rag/query" \
      -H "Content-Type: application/json" \
      -H "X-API-Key: $MY_API_KEY" \
      -d '{"query": "What does the fox jump over?", "top_k": 1, "generate_answer": true}'
    ```
    Expected response (will vary):
    ```json
    {
      "query": "What does the fox jump over?",
      "search_results": [
        {
          // ... details of testdoc.pdf text content ...
          "text": "The quick brown fox jumps over the lazy dog.", // Or similar extracted text
          "metadata": {
            "source": "e2e_pdf_test",
            // ... other metadata ...
          }
        }
      ],
      "answer": "The fox jumps over the lazy dog.", // Or similar
      "llm_model_used": "llama2"
    }
    ```
    If `search_results` are empty or the answer is generic for PDF, check `doc-processing-service` logs for PDF extraction success and Qdrant indexing. Also, ensure the PDF content was simple and extractable.

9.  **Query the RAG Service for content from the processed DOCX document (via Gateway)**:
    ```bash
    curl -X POST "<gateway-url>/rag/query" \
      -H "Content-Type: application/json" \
      -H "X-API-Key: $MY_API_KEY" \
      -d '{"query": "What files are processed by Jules?", "top_k": 1, "generate_answer": true}'
    ```
    The first time you run this, it will fetch from Qdrant and Ollama. Subsequent identical requests (within the cache TTLs, default 1hr for search results, 24hr for LLM answers) should be faster and potentially indicate `cached_response: true` (or parts of it were cached).
    To test caching explicitly:
    *   Run the query once. Note the response time (e.g., using `time curl ...`).
    *   Run the exact same query again. It should be significantly faster.
    *   Check the RAG Query Service logs for "Cache HIT" messages.
    *   To bypass cache for a specific request, add `"force_no_cache": true` to the JSON payload:
        ```bash
        curl -X POST "<gateway-url>/rag/query" \
          -H "Content-Type: application/json" \
          -H "X-API-Key: $MY_API_KEY" \
          -d '{"query": "What files are processed by Jules?", "top_k": 1, "generate_answer": true, "force_no_cache": true}'
        ```
    Expected response (will vary):
    ```json
    {
      "query": "What files are processed by Jules?",
      "search_results": [ /* ... */ ],
      "answer": "Jules processes DOCX files.", // Or similar
      "llm_model_used": "llama2",
      "cached_response": false // or true if a previous identical non-forced query was made
    }
    ```
    Check logs if results are not as expected.

### 8.7. Check RAG Query Service Metrics (Conceptual)
After running some queries through the RAG Query Service (as in step 8.6), you can conceptually check its `/metrics` endpoint.
If you have Prometheus deployed and scraping this service (which is beyond this local setup guide for now), you would query Prometheus.
Locally, you could temporarily port-forward to the RAG Query Service to view its metrics endpoint:

1.  **Find a RAG Query Service pod name**:
    ```bash
    kubectl get pods -l app.kubernetes.io/name={{ .Release.Name }}-rag-query-service # If deployed via Helm
    # OR
    kubectl get pods -l app=rag-query-service # If deployed via raw manifests
    # Pick one pod name, e.g., my-rag-instance-rag-query-service-xxxxxxxxx-yyyyy
    ```
2.  **Port-forward to the pod**:
    ```bash
    # Replace <pod-name> with the actual pod name and 8001 with its containerPort
    # kubectl port-forward <pod-name> 8001:8001
    ```
3.  **Access metrics in a new terminal**:
    ```bash
    # curl http://localhost:8001/metrics
    ```
    You should see a text-based output of Prometheus metrics, including default FastAPI metrics and the custom ones like `rag_cache_hits_total`, `rag_qdrant_query_latency_seconds_bucket`, etc.
4.  Stop the port-forward when done.

### 8.8. Test RabbitMQ Producer/Consumer Examples (Manual Execution)
This test verifies basic RabbitMQ connectivity and message flow. It requires running the example scripts manually.

1.  **Ensure RabbitMQ is deployed and running in Kubernetes (see Section 4.3).**
2.  **Set Environment Variables for RabbitMQ connection (if not using defaults or if running scripts outside a K8s-aware environment that resolves `rabbitmq-service`):**
    Open two terminals. In both, navigate to the root of your cloned repository.
    If RabbitMQ is running in K8s and you want to connect from your local machine directly (not from within a K8s pod), you'll need to port-forward the AMQP port:
    ```bash
    # In a separate terminal, keep this running:
    kubectl port-forward service/rabbitmq-service 5672:5672
    ```
    Then, in your script terminals, the default `RABBITMQ_HOST=localhost` and `RABBITMQ_PORT=5672` (with user/password `user`/`password` as per example) should work.
    If your RabbitMQ setup uses different credentials or is accessed differently, set these:
    ```bash
    # export RABBITMQ_HOST="localhost" # If port-forwarding
    # export RABBITMQ_PORT="5672"
    # export RABBITMQ_USER="user"
    # export RABBITMQ_PASSWORD="password"
    ```

3.  **Run the Consumer Script:**
    In the first terminal:
    ```bash
    python Scripts/utils/rabbitmq_consumer_example.py
    ```
    The consumer will start and log "[*] Waiting for messages...".

4.  **Run the Producer Script:**
    In the second terminal:
    ```bash
    python Scripts/utils/rabbitmq_producer_example.py
    ```
    The producer will send a message (or multiple, if you modify it) and log what it sent.

5.  **Observe Consumer Output:**
    Switch back to the first terminal (consumer). You should see logs indicating that the message was received, processed, and acknowledged. Example:
    ```
    INFO:pika.adapters.blocking_connection:Successfully connected to 127.0.0.1:5672/_
    INFO:__main__:[*] Waiting for messages in queue 'example_task_queue'. To exit press CTRL+C
    INFO:__main__:[x] Received message (delivery_tag: 1)
    INFO:__main__:    Properties: <BasicProperties(['delivery_mode=2'])>
    INFO:__main__:    Body (JSON): {
      "task_id": "task_167...",
      "payload": "Process this item.",
      "timestamp": "...",
      "priority": "high"
    }
    INFO:__main__:[x] Done processing message (delivery_tag: 1). Acknowledged.
    ```

6.  **Stop the Consumer:** Press `CTRL+C` in the consumer terminal. Stop the port-forwarding if you started it.

This test confirms that the RabbitMQ instance is operational and that the example Pika scripts can connect, publish, and consume messages.

## 9. Troubleshooting

*   **Pods not starting (`Pending`, `CrashLoopBackOff`)**:
    *   `kubectl describe pod <pod-name>` to see events and reasons.
    *   `kubectl logs <pod-name> -c <container-name> --previous` (if it restarted).
    *   Check resource requests/limits vs. your local cluster capacity.
    *   Ensure ConfigMaps/Secrets referenced by the pod exist.
*   **`ImagePullBackOff` / `ErrImagePull`**:
    *   Image name/tag is incorrect in Deployment YAML.
    *   Image not pushed to the registry K8s is trying to pull from.
    *   If private registry, `imagePullSecrets` not configured or incorrect.
    *   For Kind/Minikube, ensure image is loaded into the cluster's Docker daemon.
*   **Service Connectivity Issues (503 Service Unavailable from Gateway)**:
    *   Backend service (RAG, DocProc) pods are not running or not healthy. Check their logs.
    *   Kubernetes Service names in `internal-api-gateway-configmap.yaml` (e.g., `RAG_QUERY_SERVICE_URL: "http://rag-query-service:8001"`) must match the `metadata.name` of the backend Kubernetes `Service` resources.
    *   Ports in the URLs must match the `port` defined in the backend `Service` resources (not necessarily the `targetPort`).
*   **Qdrant Issues**:
    *   Ensure Qdrant pod is running and healthy.
    *   Ensure the RAG Query Service is configured with the correct Qdrant service host and port.
    *   Ensure the Qdrant collection exists and has the correct vector configuration (e.g., vector size matching the embedding model). Manually create it via Qdrant UI or client if needed for the first run.

## 10. Cleanup

To delete all deployed application resources:
```bash
kubectl delete -f deployment/local_k8s/
```
To delete Qdrant (if deployed via manifest):
```bash
kubectl delete -f deployment/local_k8s/dependencies/qdrant-deployment.yaml
```
Or via Helm:
```bash
helm uninstall qdrant-db
```
Stop your local cluster:
*   Minikube: `minikube stop`
*   Kind: `kind delete cluster --name rag-dev-cluster`
*   Docker Desktop: Disable Kubernetes.

Remove local registry (if used):
```bash
docker stop kind-registry && docker rm kind-registry
```

## 11. Running Automated Tests Locally

After setting up your environment and installing dependencies from `requirements.txt` (ideally in a virtual environment), you can run the automated tests:

```bash
# From the root of the repository
pytest
# Or to run tests in a specific directory:
# pytest tests/unit/
# pytest tests/integration/
```
Ensure any services required by integration tests (like a local RAG Query Service for its health check test) are running, or mock them appropriately if running tests in complete isolation. The unit tests for `RedisCacheManager` use mocking and do not require a live Redis server.
