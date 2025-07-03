# n8n Workflow Engine Integration Guide

This document outlines how to integrate and use n8n as the workflow engine for this RAG system. n8n allows for visual workflow creation and can automate sequences of tasks involving various services.

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

Services within our application can trigger n8n workflows by making HTTP POST requests to the webhook URLs exposed by n8n workflows.

**Example Python Code (using `httpx`):**

```python
# Scripts/utils/n8n_client.py (New File)
import httpx
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def trigger_n8n_workflow(webhook_url: str, payload: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
    """
    Triggers an n8n workflow by sending a POST request to its webhook URL.

    Args:
        webhook_url: The full webhook URL of the n8n workflow.
        payload: The JSON payload to send to the workflow.
        timeout: Request timeout in seconds.

    Returns:
        The JSON response from n8n if successful, None otherwise.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload, timeout=timeout)
            response.raise_for_status()  # Raises an exception for 4XX/5XX responses

            # n8n webhooks typically return a success message or data from the first few nodes if configured.
            # The exact response depends on the n8n workflow's configuration.
            try:
                response_data = response.json()
                logger.info(f"Successfully triggered n8n workflow at {webhook_url}. Response: {response_data}")
                return response_data
            except ValueError: # If response is not JSON
                logger.info(f"Successfully triggered n8n workflow at {webhook_url}. Response (non-JSON): {response.text}")
                return {"status": "success", "content": response.text}

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error triggering n8n workflow at {webhook_url}: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Request error triggering n8n workflow at {webhook_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error triggering n8n workflow: {e}", exc_info=True)

    return None

# Example usage within another service:
# from Scripts.utils.n8n_client import trigger_n8n_workflow
#
# async def some_service_function():
#     N8N_MY_WORKFLOW_URL = "http://localhost:5678/webhook/my-workflow-id" # Replace with actual n8n webhook URL
#     data_for_workflow = {"document_id": "123", "user_email": "user@example.com"}
#     result = await trigger_n8n_workflow(N8N_MY_WORKFLOW_URL, data_for_workflow)
#     if result:
#         print("n8n workflow triggered, result:", result)
#     else:
#         print("Failed to trigger n8n workflow.")
```

### n8n Workflows Calling Application APIs

n8n workflows can, in turn, call APIs exposed by our application services (e.g., `internal-api-gateway` or individual services directly if networked appropriately).

*   **HTTP Request Node:** Use n8n's "HTTP Request" node.
*   **Authentication:** If our APIs require authentication (e.g., API keys via `X-API-Key` header for the `internal-api-gateway`), configure this in the HTTP Request node's "Authentication" settings (e.g., "Header Auth"). Store sensitive keys in n8n's credential manager.
*   **Service Discovery:** Ensure n8n can resolve the hostnames of our internal services. If running via `docker-compose`, services are usually accessible by their service name (e.g., `http://internal-api-gateway:8000`).

## 4. Enhancements and Considerations

*   **Error Handling in n8n:** n8n workflows have built-in error handling capabilities. You can define error paths, retry mechanisms, or trigger specific error workflows.
*   **Logging in n8n:** n8n provides execution logs for each workflow run, which can be viewed in its UI. For production, consider configuring n8n to output logs in a way that can be captured by a centralized logging system if needed (e.g., JSON to stdout).
*   **Credentials Management:** Use n8n's built-in credential manager to securely store API keys, tokens, or other secrets needed by your workflows to interact with our services or external ones.
*   **Workflow Versioning:** n8n supports workflow versioning, allowing you to iterate on workflows without breaking existing functionality.
*   **Parameter Passing:** Data can be passed into n8n workflows via the webhook payload (JSON body). n8n nodes can then process and transform this data.
*   **Retrieving Results/Status:**
    *   **Synchronous (Wait Mode):** Some n8n nodes (like "Execute Workflow") can run other workflows and wait for their completion. Webhook responses can also be configured to return data from the workflow.
    *   **Asynchronous (Callback/Polling):** For long-running workflows, n8n could make a final HTTP request back to one of our application's API endpoints to signal completion or provide results. Alternatively, our application might need to poll an n8n API for workflow status if available (this is less common for webhook-triggered workflows).
*   **Security:**
    *   **Webhook Security:** Protect n8n webhook URLs. If n8n is exposed publicly, consider using hard-to-guess URLs or an additional authentication layer in front of n8n if it doesn't natively support per-webhook authentication easily. For internal calls (e.g., from another Docker container in the same network), this is less of an issue.
    *   **API Key for Callbacks:** When n8n calls back into your application's APIs (e.g., the `internal-api-gateway`), ensure it uses an API key. Store this API key securely in n8n's built-in credential manager.
        *   In the n8n "HTTP Request" node, under "Authentication", select "Header Auth".
        *   Set the "Name" to `X-API-Key` (or your gateway's API key header name).
        *   For the "Value", click "Add Credential" -> "Header Auth" and create a new credential storing your application's API key.
    *   **Network Policies:** If running in Kubernetes, use network policies to restrict which services can call n8n and which services n8n can call.
    *   **HTTPS:** Always use HTTPS for n8n in production.

### Example: n8n Workflow Calling Back to Application API

Let's extend the "Document Ingestion and Notification" example. Suppose after the `doc-processing-service` is called, we want n8n to call a status update endpoint in our application.

**Workflow Steps in n8n:**

1.  **Webhook Trigger:** (As before) Receives initial data (e.g., `{"doc_id": "xyz123", "callback_url": "http://internal-api-gateway/api/v1/workflow_status_update"}`).
2.  **Call Document Processing Service:** (As before) n8n calls `http://doc-processing-service:8002/process_document`. Let's assume this service responds with `{"status": "processing_started", "internal_doc_id": "uuid-abc"}`.
3.  **Set Variables (Optional but good practice):** Use a "Set" node in n8n to extract `internal_doc_id` from the previous step's output.
4.  **HTTP Request Node (Callback to App):**
    *   **URL:** Use the `callback_url` received in the initial webhook trigger (e.g., `{{ $json.body.callback_url }}`).
    *   **Method:** POST
    *   **Authentication:** Header Auth (as described above, using an API key for your `internal-api-gateway`).
    *   **Body (JSON):**
        ```json
        {
          "original_doc_id": "{{ $json.body.doc_id }}",
          "processed_doc_id": "{{ $item.json.internal_doc_id }}", // From Set node or step 2 output
          "status": "document_processing_invoked",
          "timestamp": "{{ $now.toISO() }}"
        }
        ```
    *   This node calls back to your application to log that processing for `doc_id` has been initiated by n8n.

### Passing Complex Parameters to n8n Webhooks

When triggering an n8n workflow via its webhook, you can send a JSON payload in the request body. n8n automatically parses this JSON.

*   **Example Payload from your Application:**
    ```json
    {
      "document_url": "s3://mybucket/docs/report.pdf",
      "priority": "high",
      "processing_options": {
        "ocr_enabled": true,
        "extract_tables": true
      },
      "notification_config": {
        "email_to": ["user1@example.com", "admin@example.com"],
        "slack_channel": "#document-alerts"
      },
      "callback_url": "http://my-app/api/n8n_callback/workflow123"
    }
    ```
*   **Accessing in n8n:**
    *   In n8n nodes (like "Set", "IF", "HTTP Request"), you can access these using expressions:
        *   `{{ $json.body.document_url }}`
        *   `{{ $json.body.priority }}`
        *   `{{ $json.body.processing_options.ocr_enabled }}`
        *   `{{ $json.body.notification_config.email_to[0] }}` (to get the first email)

### Retrieving Results/Status from n8n Workflows

1.  **Respond to Webhook Node (Synchronous-like):**
    *   If your n8n workflow is relatively short and you need an immediate response, you can use the "Respond to Webhook" node as the *last* step (or in an error branch).
    *   This node allows you to construct a custom JSON response that will be sent back as the HTTP response to the initial webhook call.
    *   **Limitation:** The initial HTTP request that triggered the workflow will hang until the "Respond to Webhook" node is executed or the workflow times out. Not suitable for very long-running workflows.
    *   **Example:** An n8n workflow that just validates input and returns "validation_ok" or "validation_failed".

2.  **Callback URL (Asynchronous):**
    *   This is the most common pattern for longer-running workflows.
    *   The initial trigger payload from your application includes a `callback_url` (an endpoint in your application).
    *   Once the n8n workflow (or a significant part of it) completes, an n8n "HTTP Request" node makes a call to this `callback_url`, sending status, results, or any relevant data.
    *   Your application needs an endpoint to receive these callbacks. This makes the interaction asynchronous.

3.  **Polling n8n API for Workflow Execution Status (Generally Complex for Webhook Triggers):**
    *   n8n has a REST API. However, for workflows triggered by a simple webhook URL (not via n8n's API that might return an execution ID), getting a persistent, pollable *execution ID* for that specific run can be tricky or might require specific n8n setup (e.g., immediately writing the execution ID to an external store that your app can query).
    *   If you trigger workflows via n8n's own API endpoints (e.g., `POST /api/v1/workflows/{id}/activate` to run a saved workflow by its n8n ID), it might return an execution ID that can then be polled using `GET /api/v1/executions/{execution_id}`.
    *   **Recommendation:** For most webhook-triggered scenarios, the callback URL pattern is more straightforward for asynchronous status updates. If precise polling is needed, triggering via n8n's own API and managing execution IDs would be the way, but this is a more advanced integration.

## 5. Starting Point

The `Scripts/n8n-workflow.json` file provides a basic workflow example. This can be imported into n8n to see a practical demonstration. Remember to update node configurations (especially URLs and credentials) to match your environment after importing.

This guide provides a starting point for integrating n8n. As specific workflow needs are identified, this document and the integration patterns can be further refined.
