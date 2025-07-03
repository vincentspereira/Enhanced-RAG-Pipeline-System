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

async def example_usage():
    """Example of how to use the trigger_n8n_workflow function."""
    # This is an example webhook URL. Replace with your actual n8n webhook URL.
    # You can get this from n8n by creating a workflow with a Webhook trigger node.
    # Make sure your n8n instance is running and accessible.
    # If n8n is running via docker-compose as defined, and your calling script is outside docker,
    # localhost:5678 should work. If the calling script is another docker container in the same
    # docker-compose network, you might use http://n8n:5678/...

    sample_webhook_url = "http://localhost:5678/webhook-test/my-simple-workflow-trigger" # Example URL

    sample_payload = {
        "document_id": "doc_12345",
        "user_email": "test@example.com",
        "message": "Please process this document."
    }

    logger.info(f"Attempting to trigger n8n workflow at: {sample_webhook_url}")
    result = await trigger_n8n_workflow(sample_webhook_url, sample_payload)

    if result:
        logger.info(f"n8n workflow trigger successful. Response from n8n: {result}")
    else:
        logger.error("Failed to trigger n8n workflow.")

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # To run this example:
    # 1. Start n8n (e.g., using the docker-compose setup).
    # 2. In n8n, create a new workflow.
    # 3. Add a "Webhook" node as the trigger.
    # 4. After activating the workflow, copy the "TEST URL" from the Webhook node.
    # 5. Replace `sample_webhook_url` above with your actual test URL.
    # 6. You can add an "Execute Command" node after the webhook in n8n to see the payload, e.g., `echo {{ $json | jsonPretty }}`
    #    or a "Respond to Webhook" node.
    # 7. Run this script: python Scripts/utils/n8n_client.py

    # Note: For a real POST webhook in n8n, you'd use the "PRODUCTION URL".
    # The test URL is for GET requests during workflow setup.
    # You might need to use a tool like ngrok if your n8n is local and you're trying to hit it from an external service,
    # but for service-to-service calls within a Docker network or on localhost, direct URLs are fine.

    asyncio.run(example_usage())
