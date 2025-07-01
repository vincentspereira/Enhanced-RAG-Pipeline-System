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
kubectl apply -f deployment/local_k8s/dependencies/qdrant-deployment.yaml
```

**Option B: Using Helm (More Flexible)**
```bash
helm repo add qdrant https://qdrant.github.io/qdrant-helm
helm install qdrant-db qdrant/qdrant \
  --set service.name=qdrant-service \
  --set service.httpPort=6333 \
  --set service.grpcPort=6334 \
  --set persistence.enabled=false # For local testing, or true with storageClass for persistence
```
Ensure the service name matches `QDRANT_HOST` in `rag-query-service-configmap.yaml` (which is `qdrant-service`).

### 4.2. (Future) Deploy PostgreSQL & MongoDB
Placeholder for when these are needed. You would typically use Helm charts:
*   **PostgreSQL**: `helm install my-postgres bitnami/postgresql --set auth.database=yourdb --set auth.username=youruser --set auth.password=yourpass`
*   **MongoDB**: `helm install my-mongodb bitnami/mongodb --set auth.rootUser=admin --set auth.rootPassword=adminpass`

Remember to update service names and credentials in your application's ConfigMaps/Secrets if you deploy these.

## 5. Deploy Application Services

Apply the Kubernetes manifests for the RAG system services:

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
Or apply all at once:
```bash
kubectl apply -f deployment/local_k8s/
```

## 6. Verify Deployment

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

### 8.1. Gateway Health Check
```bash
curl <gateway-url>/gateway_health
```
Expected: `{"status":"healthy","service":"Internal API Gateway"}`

### 8.2. RAG Query Service Health (via Gateway)
This assumes the RAG Query Service's `/health` endpoint is at its root. The gateway forwards `/rag/health`.
```bash
curl <gateway-url>/rag/health
```
Expected: `{"status":"healthy", ...}` or `{"status":"degraded", ...}` if components within RAG service aren't fully ready. Check RAG service logs.

### 8.3. Document Processing Service Health (via Gateway)
```bash
curl <gateway-url>/document/health
```
Expected: `{"status":"healthy","service_type":"stub"}`

### 8.4. Test Document Processing Stub (via Gateway)
Create a dummy text file, e.g., `sample.txt` with "Hello world".
```bash
curl -X POST "<gateway-url>/document/process_document" \
  -F "file=@sample.txt" \
  -F "metadata_json={\"source\":\"local_test\", \"user\":\"tester\"}"
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

### 8.5. Test RAG Query Service (via Gateway)
**Note**: This test requires Qdrant to be running and the collection specified in `rag-query-service-configmap.yaml` (default: `documents`) to exist. For Iteration 1, the service *assumes* the collection exists. You might need to create it manually in Qdrant if it's the first time.
Also, to get meaningful search results, you'd need to have indexed some documents into Qdrant. The current RAG Query Service stub doesn't index anything, only queries.

This test will primarily check if the query endpoint is reachable and returns an empty list or an error if the collection isn't ready.
```bash
curl -X POST "<gateway-url>/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 1}'
```
Expected (if Qdrant is up and collection exists, but no data):
```json
{
  "query": "test query",
  "results": []
}
```
Or an error if Qdrant/collection is misconfigured. Check RAG service logs.

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
