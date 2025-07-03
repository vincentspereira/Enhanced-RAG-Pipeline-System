# n8n Workflow Engine Integration Guide

This document outlines how to integrate and use n8n as the workflow engine for this RAG system. n8n allows for visual workflow creation and can automate sequences of tasks involving various services.

**For a high-level, non-technical overview of what workflows are and how they benefit the system, please see the "Automated Workflows (n8n)" section in the [`../docs/USER_DOCUMENTATION.md`](../docs/USER_DOCUMENTATION.md).**
**For details on configuring the RAG system services that n8n might interact with, refer to [`../docs/CONFIGURING_SERVICES.md`](../docs/CONFIGURING_SERVICES.md).**

## 1. Running n8n via Docker Compose

An n8n service is defined in the `docker-compose.yml` file at the root of this repository. To run n8n locally:

1.  **Ensure Docker and Docker Compose are installed.**
2.  **Navigate to the repository root directory.**
3.  **Start the n8n service (along with other services if needed):**
    ```bash
    docker-compose up -d n8n
    ```
    Or to start all services:
    ```bash
    docker-compose up -d
    ```
4.  **Access n8n:** Once started, n8n will typically be available at `http://localhost:5678`.
5.  **Data Persistence:** The `docker-compose.yml` file maps `./n8n_data` on your host to `/home/node/.n8n` in the n8n container. This ensures your n8n workflows and credentials persist across container restarts.

## 2. Example Workflow: Document Ingestion and Notification

Let's consider a simple workflow for processing an incoming document and notifying upon completion.

**Workflow Steps (Conceptual in n8n):**

1.  **Webhook Trigger:**
    *   n8n workflow starts when it receives an HTTP POST request to a specific webhook URL.
    *   This webhook could be called by an external system or another internal service when a new document is ready for processing.
    *   The payload to the webhook might include document metadata or a pointer to the document's location.

2.  **Call Document Processing Service:**
    *   n8n uses its "HTTP Request" node to make a POST request to our `doc-processing-service` (`http://doc-processing-service:8002/process_document` if running in the same Docker network).
    *   The payload to this service would include the document content/path and any relevant metadata received by the webhook.

3.  **Call Notification Service:**
    *   After the document processing step completes (n8n can wait for HTTP responses), n8n makes another HTTP Request.
    *   This request goes to our `notification-service` (`http://notification-service:8003/send_email`).
    *   The payload would include recipient details, a subject, and a body indicating the document processing status (e.g., success or failure, based on the response from `doc-processing-service`).

**Importing an Existing Workflow:**
*   The `n8n-workflow.json` file in this repository (located at `Scripts/n8n-workflow.json`) is an example workflow.
*   You can import this JSON file directly into your n8n instance:
    1.  In the n8n UI, go to "Workflows".
    2.  Click "Import from File" and select the `Scripts/n8n-workflow.json` file.
    3.  Review and activate the workflow. You might need to adjust node configurations (like service URLs or credentials) to match your environment.

## 3. Interaction Between Application Services and n8n

### Triggering n8n Workflows from Application Services

Services within our application can trigger n8n workflows by making HTTP POST requests to the webhook URLs exposed by n8n workflows. The `Scripts/utils/n8n_client.py` file provides a helper function `trigger_n8n_workflow` for this.

**Example (from `n8n_client.py`):**
```python
# async def trigger_n8n_workflow(webhook_url: str, payload: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
# ... implementation ...
```
This function can be called from any service needing to initiate an n8n workflow.

### n8n Workflows Calling Application APIs

n8n workflows can, in turn, call APIs exposed by our application services (e.g., `internal-api-gateway` or individual services directly if networked appropriately).

*   **HTTP Request Node:** Use n8n's "HTTP Request" node.
*   **Authentication:** If our APIs require authentication (e.g., API keys via `X-API-Key` header for the `internal-api-gateway`), configure this in the HTTP Request node's "Authentication" settings (e.g., "Header Auth"). Store sensitive keys in n8n's credential manager.
*   **Service Discovery:** Ensure n8n can resolve the hostnames of our internal services. If running via `docker-compose`, services are usually accessible by their service name (e.g., `http://internal-api-gateway:8000`).

## 4. Enhancements and Considerations

*   **Error Handling in n8n:** n8n workflows have built-in error handling capabilities. You can define error paths, retry mechanisms, or trigger specific error workflows.
*   **Logging in n8n:** n8n provides execution logs for each workflow run. For production, consider configuring n8n to output logs in a way that can be captured by a centralized logging system (e.g., JSON to stdout).
*   **Credentials Management:** Use n8n's built-in credential manager to securely store API keys, tokens, or other secrets needed by your workflows.
*   **Workflow Versioning:** n8n supports workflow versioning.
*   **Parameter Passing to Webhooks:** Send a JSON payload in the request body. n8n automatically parses this. Access data in nodes using expressions like `{{ $json.body.parameter_name }}`.
*   **Retrieving Results/Status from n8n Workflows:**
    1.  **Respond to Webhook Node (Synchronous-like):** For short workflows, use this node to send an immediate HTTP response.
    2.  **Callback URL (Asynchronous):** For longer workflows, the initial trigger payload can include a `callback_url`. n8n then calls this URL in your application upon completion.
    3.  **Polling n8n API (Advanced):** If workflows are triggered via n8n's own API (not simple webhooks), an execution ID might be returned, which could potentially be polled. This is generally more complex for webhook-triggered runs.
*   **Security:**
    *   Protect n8n webhook URLs.
    *   Use HTTPS for n8n in production.
    *   When n8n calls your application APIs, use API key authentication. Store the key in n8n's credential manager.
    *   Use Kubernetes network policies if applicable.

## 5. Starting Point

The `Scripts/n8n-workflow.json` file provides a basic workflow example. Import it into n8n and adapt it to your needs. Remember to update node configurations (especially URLs and credentials) after importing.
This guide provides a starting point for integrating n8n. As specific workflow needs are identified, this document and the integration patterns can be further refined.
