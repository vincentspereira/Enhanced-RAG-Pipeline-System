# Deployment Strategies and `deploy.sh` Guide

This document provides conceptual guidance on structuring the `deploy.sh` script (referenced in the CI/CD pipeline) and outlines different deployment strategies that can be implemented for this RAG system, particularly when deploying to Kubernetes.

The `deploy.sh` script is intended to be the entry point for automated deployments triggered by the CI/CD pipeline after successful build and test stages. Its specific content will heavily depend on your target environment (e.g., specific Kubernetes cluster, cloud provider, namespace conventions) and chosen deployment tools.

## 1. Prerequisites for `deploy.sh`

The script will typically expect certain environment variables or tools to be available in the CI/CD runner environment:

*   **`KUBECONFIG`**: Configured to allow `kubectl` access to the target Kubernetes cluster. This is often set up as a secret in the CI/CD system.
*   **`kubectl`**: The Kubernetes command-line tool must be installed.
*   **`helm`**: The Helm package manager for Kubernetes must be installed if using Helm for deployment (which is the primary method for this project).
*   **Image Details**: Environment variables like `IMAGE_NAME`, `IMAGE_TAG` (e.g., passed from the CI build step).
*   **Target Namespace**: `TARGET_NAMESPACE` for deployment.
*   **Environment Specific Values**: A way to specify environment-specific Helm values (e.g., `VALUES_FILE_PATH` pointing to `values-prod.yaml` or `values-staging.yaml`).

## 2. Basic Deployment (Rolling Update via Helm)

This is the simplest strategy and is often the default for Helm.

**Conceptual `deploy.sh` structure:**

```bash
#!/bin/bash
set -eo pipefail # Exit on error, treat unset variables as an error, and propagate exit status

# --- Configuration - Passed from CI/CD or set here ---
TARGET_NAMESPACE="${TARGET_NAMESPACE:-default}"
HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-my-rag-system}"
CHART_PATH="./charts/rag-system"
IMAGE_TAG="${IMAGE_TAG:-latest}" # Should be the specific tag from the build
VALUES_FILE="${VALUES_FILE_PATH:-./charts/rag-system/values.yaml}" # Default values, ideally override with env-specific

echo "--- Starting Deployment ---"
echo "Target Namespace: $TARGET_NAMESPACE"
echo "Helm Release Name: $HELM_RELEASE_NAME"
echo "Chart Path: $CHART_PATH"
echo "Image Tag: $IMAGE_TAG"
echo "Values File: $VALUES_FILE"

# --- Pre-deployment checks (optional) ---
# Example: Check kubectl connection
# kubectl cluster-info

# --- Helm Deployment (Upgrade with Install if not exists) ---
# This command will perform a rolling update if the release already exists.
helm upgrade --install "$HELM_RELEASE_NAME" "$CHART_PATH" \
  --namespace "$TARGET_NAMESPACE" \
  --create-namespace \
  -f "$VALUES_FILE" \
  --set image.tag="$IMAGE_TAG" \
  # --set other.key="value" # Add other overrides as needed
  --wait # Optional: wait for resources to be ready
  --timeout 10m # Optional: timeout for the wait

echo "--- Deployment Successful ---"

# --- Post-deployment checks (optional) ---
# Example: Check status of deployed pods
# kubectl get pods -n "$TARGET_NAMESPACE" -l "app.kubernetes.io/instance=$HELM_RELEASE_NAME"
# Example: Run a simple E2E test or health check script against the new deployment
```

**How Rolling Update Works (Kubernetes Default):**
Kubernetes gradually replaces old pods with new ones, ensuring a certain number of pods are always available, minimizing downtime. The `maxUnavailable` and `maxSurge` parameters in the Deployment spec control this behavior.

## 3. Blue-Green Deployment (Conceptual via Helm & Kubernetes Services)

Blue-Green deployment reduces risk by maintaining two identical environments ("blue" and "green"). Only one environment serves live traffic at any time.

**Strategy Overview:**

1.  **Initial State:** "Blue" environment is live, serving traffic via a main Kubernetes Service (e.g., `rag-system-live`).
2.  **Deploy New Version:** Deploy the new application version to the "Green" environment. This involves deploying a separate Helm release with a different name or suffix (e.g., `my-rag-system-green`) or using canary capabilities of a service mesh/ingress.
3.  **Test Green:** Thoroughly test the "Green" environment (not serving live traffic).
4.  **Switch Traffic:** If Green is healthy, update the main Kubernetes Service (`rag-system-live`) selector to point to the "Green" deployment's pods.
5.  **Monitor:** Monitor the "Green" environment now serving live traffic.
6.  **Rollback (if needed):** If issues arise, quickly switch the Service selector back to "Blue".
7.  **Decommission Blue:** After a period of confidence, decommission or update the "Blue" environment.

**Conceptual `deploy.sh` additions for Blue-Green (simplified):**

This is more complex and often involves more sophisticated tooling or manual steps if not fully automated. The script would need to manage different Helm release names or labeling strategies.

```bash
# ... (previous config) ...

# --- Determine current live color and next deploy color ---
# This logic needs to be robust, potentially querying the live service selector
# For simplicity, let's assume we alternate or have a way to know.
# CURRENT_LIVE_COLOR="blue" # This would be determined dynamically
# NEXT_DEPLOY_COLOR="green"
# if [ "$CURRENT_LIVE_COLOR" == "green" ]; then
#   NEXT_DEPLOY_COLOR="blue"
# fi

# HELM_RELEASE_NAME_DEPLOY="$HELM_RELEASE_NAME-$NEXT_DEPLOY_COLOR"
# LIVE_SERVICE_NAME="rag-system-live-service" # The service users hit

echo "Deploying $IMAGE_TAG to $NEXT_DEPLOY_COLOR environment ($HELM_RELEASE_NAME_DEPLOY)"

# --- 1. Deploy to Standby (Green) Environment ---
helm upgrade --install "$HELM_RELEASE_NAME_DEPLOY" "$CHART_PATH" \
  --namespace "$TARGET_NAMESPACE" \
  --create-namespace \
  -f "$VALUES_FILE" \
  --set image.tag="$IMAGE_TAG" \
  # --set deployment.colorLabel="$NEXT_DEPLOY_COLOR" # Ensure pods get a color label
  --wait --timeout 10m

echo "--- Standby Environment ($NEXT_DEPLOY_COLOR) Deployed ---"

# --- 2. Run Tests on Standby Environment (Critical Step) ---
# This would involve running smoke tests, E2E tests against the standby environment's specific endpoint/IP.
# If tests fail, an automated rollback or alert should occur.
# Example: ./run-tests-on-standby.sh "$TARGET_NAMESPACE" "$HELM_RELEASE_NAME_DEPLOY"
# read -p "Tests on $NEXT_DEPLOY_COLOR passed? (y/n): " tests_passed
# if [ "$tests_passed" != "y" ]; then
#   echo "Tests failed on $NEXT_DEPLOY_COLOR. Rolling back deployment (deleting)."
#   helm uninstall "$HELM_RELEASE_NAME_DEPLOY" --namespace "$TARGET_NAMESPACE"
#   exit 1
# fi
echo "TODO: Implement actual tests for standby environment"


# --- 3. Switch Live Traffic to Green (Example: Modify Service Selector) ---
# This assumes your main service targets pods based on a 'color' label.
# The Helm chart Deployments would need to set this label.
# kubectl patch service "$LIVE_SERVICE_NAME" -n "$TARGET_NAMESPACE" \
#   -p '{"spec": {"selector": {"app.kubernetes.io/instance": "'"$HELM_RELEASE_NAME_DEPLOY"'", "color": "'"$NEXT_DEPLOY_COLOR"'"}}}'
# A better way is to have the service select on common app labels, and then update the
# 'color' label on the new Deployment's PodTemplate and let a rolling update happen for the label switch,
# or manage two distinct services and switch at an Ingress or DNS level.
#
# Simpler for Helm: If each color is a distinct Helm release, you might have two services
# and switch an Ingress rule or a higher-level DNS.
# For a very basic K8s service switch:
#   kubectl patch svc <main-service-name> -n <namespace> -p '{"spec":{"selector":{"deploymentLabel":"<new-version-label>"}}}'
# This requires deployments to have a unique label per version.
echo "TODO: Implement traffic switching logic (e.g., update Service selector, Ingress rule, or DNS)"
echo "Traffic switched conceptually to $NEXT_DEPLOY_COLOR"


# --- 4. Monitor Switched Environment ---
echo "Monitoring new live environment ($NEXT_DEPLOY_COLOR)..."
# sleep 5m # Wait for a monitoring period

# --- 5. Rollback (if needed, conceptual) ---
# if [ <monitoring_indicates_failure> ]; then
#   echo "Rollback: Switching traffic back to $CURRENT_LIVE_COLOR"
#   kubectl patch service "$LIVE_SERVICE_NAME" -n "$TARGET_NAMESPACE" \
#     -p '{"spec": {"selector": {"app.kubernetes.io/instance": "'"$HELM_RELEASE_NAME-$CURRENT_LIVE_COLOR"'", "color": "'"$CURRENT_LIVE_COLOR"'"}}}'
#   # Potentially scale down or delete the failed NEXT_DEPLOY_COLOR environment
# fi

# --- 6. Decommission Old Live Environment (after confidence period) ---
# OLD_LIVE_HELM_RELEASE_NAME="$HELM_RELEASE_NAME-$CURRENT_LIVE_COLOR"
# echo "TODO: Implement decommissioning of old environment ($OLD_LIVE_HELM_RELEASE_NAME) after confidence period."
# helm uninstall "$OLD_LIVE_HELM_RELEASE_NAME" --namespace "$TARGET_NAMESPACE" # Example

echo "--- Blue-Green Deployment Step Conceptually Completed ---"
```

**Considerations for Blue-Green:**
*   **Stateful Applications:** Requires careful handling of database schemas, data migration, and shared state.
*   **Resource Intensive:** Doubles the resource cost during the transition period.
*   **Complexity:** More complex to automate than rolling updates.
*   **Service Meshes / Ingress Controllers:** Tools like Istio, Linkerd, Nginx Ingress, or Traefik can provide more sophisticated traffic shifting capabilities (e.g., weighted routing, canary releases) that simplify Blue-Green or enable Canary deployments.

## 4. Canary Deployments (Conceptual)

Canary deployments involve releasing the new version to a small subset of users/traffic, monitoring its performance, and gradually rolling it out to everyone if it's stable.

*   **Implementation:** Typically requires an Ingress controller or Service Mesh that supports weighted traffic splitting.
*   **`deploy.sh` Role:** The script would deploy the new version alongside the old one (e.g., as a separate Kubernetes Deployment with a specific version label). Then, it would interact with the Ingress/Service Mesh API to configure traffic splitting (e.g., 90% to old version, 10% to new version).
*   **Monitoring & Promotion:** Based on monitoring, the script (or an automated process) would gradually increase traffic to the new version or roll back if issues occur.

## 5. Rollback Strategy

*   **Helm:** `helm rollback <RELEASE_NAME> [REVISION]` can be used to revert to a previous successful Helm release. The `deploy.sh` script could incorporate this.
    ```bash
    # In case of failure after `helm upgrade`
    # helm rollback "$HELM_RELEASE_NAME" # Rolls back to previous revision
    ```
*   **Kubernetes Deployments:** `kubectl rollout undo deployment/<deployment-name>` reverts to the previous ReplicaSet.
*   **Automated Rollback:** CI/CD can trigger automated rollbacks if post-deployment health checks or tests fail.

## Conclusion

The `deploy.sh` script should be tailored to your specific environment and desired deployment strategy.
*   Start with a **robust Helm-based rolling update**.
*   For more advanced strategies like **Blue-Green or Canary**, consider leveraging features of your Ingress controller or a Service Mesh if available, and design `deploy.sh` to interact with those systems.
*   Always include **pre-deployment checks and post-deployment verification/monitoring** steps in your script.
*   Implement **clear rollback procedures**, whether manual or automated.
