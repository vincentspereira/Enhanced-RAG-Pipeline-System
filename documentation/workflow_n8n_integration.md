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
    *   Protect n8n webhook URLs if they are publicly accessible.
    *   Use HTTPS for n8n, especially in production.
    *   If n8n calls internal application APIs, ensure appropriate network policies and authentication are in place.

## 5. Starting Point

The `Scripts/n8n-workflow.json` file provides a basic workflow example. This can be imported into n8n to see a practical demonstration. Remember to update node configurations (especially URLs and credentials) to match your environment after importing.

This guide provides a starting point for integrating n8n. As specific workflow needs are identified, this document and the integration patterns can be further refined.
