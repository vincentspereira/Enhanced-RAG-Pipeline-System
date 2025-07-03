import httpx
import os
import time
import logging
import json
import uuid

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Service URLs from environment variables
GATEWAY_URL = os.getenv("TEST_GATEWAY_URL", "http://localhost:30080") # Assuming NodePort for gateway from local K8s
API_KEY = os.getenv("TEST_API_KEY", "secretkey1") # Use a key configured in rag-gateway-apikeys-secret

# Helper to ensure service readiness (very basic)
async def wait_for_service(url: str, service_name: str, max_retries: int = 10, delay_seconds: int = 5):
    logger.info(f"Waiting for {service_name} at {url} to be ready...")
    for i in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                logger.info(f"{service_name} is ready!")
                return True
        except httpx.RequestError as e:
            logger.debug(f"{service_name} not ready yet (attempt {i+1}/{max_retries}): {e}")
        if i < max_retries -1 :
             await asyncio.sleep(delay_seconds)
    logger.error(f"{service_name} did not become ready after {max_retries * delay_seconds} seconds.")
    return False

async def run_e2e_test():
    logger.info("--- Starting Basic RAG E2E Flow Test ---")

    # 0. Check health of gateway (unprotected endpoint)
    gateway_health_url = f"{GATEWAY_URL}/gateway_health"
    if not await wait_for_service(gateway_health_url, "API Gateway"):
        logger.error("API Gateway not healthy. Aborting E2E test.")
        return False

    # Define document details
    doc_id = f"e2e-doc-{uuid.uuid4()}"
    doc_content = "The E2E test document states that effective end-to-end testing is crucial for system validation. Jules likes green apples."
    doc_filename = "e2e_test_doc.txt"
    metadata = {"source": "e2e_test_suite", "document_id": doc_id, "custom_tag": "e2e_v1"}

    doc_proc_url = f"{GATEWAY_URL}/document/process_document"
    rag_query_url = f"{GATEWAY_URL}/rag/query"
    headers = {"X-API-Key": API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Upload a document to Document Processing Service via Gateway
        logger.info(f"Step 1: Uploading document '{doc_filename}' with ID '{doc_id}'...")
        files = {'file': (doc_filename, doc_content.encode('utf-8'), 'text/plain')}
        data = {'metadata_json': json.dumps(metadata)}

        try:
            response_upload = await client.post(doc_proc_url, headers=headers, files=files, data=data)
            response_upload.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
            upload_json = response_upload.json()
            logger.info(f"Upload response: {upload_json}")
            assert upload_json.get("status") == "indexed_chunked" # or "indexed" if not chunking short docs
            assert upload_json.get("parent_document_id") == doc_id
            assert upload_json.get("chunks_indexed", 0) > 0
            logger.info("Document upload and initial processing successful.")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during document upload: {e.response.status_code} - {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"Error during document upload: {e}", exc_info=True)
            return False

        # 2. Wait for processing/indexing (simple sleep for now)
        processing_wait_time = 10 # seconds
        logger.info(f"Step 2: Waiting {processing_wait_time} seconds for document indexing...")
        await asyncio.sleep(processing_wait_time)

        # 3. Query RAG Query Service for content from that document via Gateway
        query = "What is crucial for system validation according to the E2E test document?"
        logger.info(f"Step 3: Querying RAG service with: '{query}'")
        query_payload = {
            "query": query,
            "top_k": 3,
            "search_type": "hybrid", # Test hybrid search
            "generate_answer": True,
            "force_no_cache": True # Ensure we hit the backend, not just cache from a previous failed run
        }
        try:
            response_query = await client.post(rag_query_url, headers=headers, json=query_payload)
            response_query.raise_for_status()
            query_json = response_query.json()
            logger.info(f"Query response: {json.dumps(query_json, indent=2)}")

            # 4. Assertions on the query response
            assert query_json.get("query") == query
            assert "search_results" in query_json
            assert "answer" in query_json

            found_in_search = False
            if query_json["search_results"]:
                for res in query_json["search_results"]:
                    if res.get("id", "").startswith(doc_id): # Check if a chunk from our doc is found
                        found_in_search = True
                        assert doc_content in res.get("text", ""), "Original text not found in search result text"
                        break
            assert found_in_search, f"Uploaded document (ID starting with {doc_id}) not found in search results."

            # Check if the answer contains relevant keywords (very basic check)
            expected_answer_keywords = ["end-to-end testing", "crucial", "validation"]
            llm_answer = query_json.get("answer", "").lower()
            for kw in expected_answer_keywords:
                assert kw in llm_answer, f"Expected keyword '{kw}' not in LLM answer: '{llm_answer}'"

            logger.info("E2E RAG flow test successful!")
            return True

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during RAG query: {e.response.status_code} - {e.response.text}")
            return False
        except AssertionError as e:
            logger.error(f"Assertion failed: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error during RAG query: {e}", exc_info=True)
            return False

if __name__ == "__main__":
    import asyncio
    # This script needs a running K8s environment with all services deployed and configured.
    # Set GATEWAY_URL and API_KEY env vars if not using defaults.
    # Example:
    # export TEST_GATEWAY_URL="http://your-minikube-ip:nodeport"
    # export TEST_API_KEY="your-actual-api-key"
    # python Scripts/tests/e2e/basic_rag_flow_test.py

    logger.info("Starting E2E test script. Ensure all services (Gateway, DocProc, RAGQuery, Qdrant, Ollama, Redis) are running and configured.")

    success = asyncio.run(run_e2e_test())

    if success:
        logger.info("E2E Test Passed!")
        exit(0)
    else:
        logger.error("E2E Test Failed.")
        exit(1)

# Instructions to add to DEPLOY_LOCAL_TEST_GUIDE.md (Section 12):
#
# ### 12.x Running Basic E2E Test
# This script tests the flow of uploading a document and then querying for its content.
#
# 1.  **Prerequisites**:
#     *   All services (API Gateway, Document Processing, RAG Query) and dependencies (Qdrant, Ollama with a model like `llama2` pulled, Redis) must be deployed and running correctly in your Kubernetes cluster.
#     *   The API Gateway must be accessible (determine its URL, e.g., `$GATEWAY_URL`).
#     *   You need a valid API Key (e.g., `secretkey1` if using defaults).
#
# 2.  **Set Environment Variables (in the terminal where you'll run the script)**:
#     ```bash
#     export TEST_GATEWAY_URL="<your-gateway-url>" # e.g., http://localhost:30080 or http://<minikube-ip>:<nodeport>
#     export TEST_API_KEY="<your-valid-api-key>"   # e.g., secretkey1
#     # Optional: If your services run on different ports than defaults assumed by script, set them too.
#     ```
#
# 3.  **Run the E2E test script from the repository root**:
#     ```bash
#     python Scripts/tests/e2e/basic_rag_flow_test.py
#     ```
# 4.  **Observe Logs**: The script will log its progress. A final "E2E Test Passed!" or "E2E Test Failed." message will be printed. Check logs for details on any failures.
#
# **Note**: This E2E test is basic. It uses a short `time.sleep()` to wait for indexing. In more complex scenarios or for CI, more robust checks or a status polling mechanism for document processing might be needed.
