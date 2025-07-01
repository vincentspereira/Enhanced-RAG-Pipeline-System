import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import config loader
try:
    from Scripts.utils.config_loader import get_config_value
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from utils.config_loader import get_config_value
    except ImportError as e:
        logger.error(f"Critical: Failed to import get_config_value for Document Processing Service. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None): # Basic fallback
            return os.getenv(env_var_name, default)

# --- Global Variables ---
app = FastAPI(title="Document Processing Service (Stub)")

# --- Pydantic Models ---
class DocumentMetadata(BaseModel):
    source: Optional[str] = None
    document_id: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    # Add any other relevant metadata fields

class ProcessingResponse(BaseModel):
    message: str
    filename: Optional[str] = None
    metadata_received: Optional[Dict[str, Any]] = None
    status: str = "received"

# --- Service Initialization and Shutdown ---
@app.on_event("startup")
async def startup_event():
    logger.info("Document Processing Service (Stub) starting up...")
    # In future iterations, initialize connections to MongoDB, Kafka, etc.
    # Example:
    # MONGO_DB_NAME = get_config_value("MONGO_DB", yaml_path="database.mongodb.dbname", default="doc_metadata_db")
    # mongo_connector = MongoDBConnector() # Assuming MongoDBConnector is initialized globally or passed
    # if mongo_connector._initialized:
    #     logger.info(f"Connected to MongoDB for metadata, using DB: {MONGO_DB_NAME}")
    # else:
    #     logger.error("Failed to connect to MongoDB for metadata.")
    logger.info("Document Processing Service (Stub) startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Document Processing Service (Stub) shutting down...")
    # Close any connections if opened in startup
    logger.info("Document Processing Service (Stub) shutdown complete.")

# --- API Endpoints ---
@app.post("/process_document", response_model=ProcessingResponse, tags=["Processing"])
async def process_document_endpoint(
    file: UploadFile = File(None, description="The document file to process."),
    document_content: Optional[str] = Form(None, description="Text content of the document if not sending a file."),
    metadata_json: Optional[str] = Form("{}", description="JSON string of document metadata.")
    # For more structured metadata, consider using multiple Form fields or a JSON body with file upload
    # For JSON body with file upload, you might need a different approach or a library.
    # FastAPI example: https://fastapi.tiangolo.com/tutorial/request-forms-and-files/
):
    filename = None
    content_summary = "No direct content provided."

    if file:
        filename = file.filename
        logger.info(f"Received file: {filename}, content type: {file.content_type}")
        # For Iteration 1, we don't read or store the file content.
        # In a real implementation:
        # content_bytes = await file.read()
        # content_text = content_bytes.decode('utf-8') # Or other appropriate decoding
        # Store or process content_text
        content_summary = f"File '{filename}' received."
    elif document_content:
        logger.info(f"Received document content (first 100 chars): {document_content[:100]}")
        content_summary = f"Text content received (length: {len(document_content)})."
    else:
        raise HTTPException(status_code=400, detail="Either a 'file' or 'document_content' must be provided.")

    try:
        import json
        metadata = json.loads(metadata_json)
        logger.info(f"Received metadata: {metadata}")
    except json.JSONDecodeError:
        logger.error(f"Failed to parse metadata_json: {metadata_json}")
        raise HTTPException(status_code=400, detail="Invalid JSON format for metadata.")

    # --- Placeholder for actual processing logic ---
    # In future iterations:
    # 1. Store document content (e.g., to S3 or local filestore).
    # 2. Store metadata (e.g., to MongoDB using the MongoDBConnector).
    #    Example: mongo_connector.insert_one("document_metadata", {"filename": filename, **metadata, "status": "received"})
    # 3. Extract text if it's a binary file (PDF, DOCX).
    # 4. Chunk the document.
    # 5. Generate embeddings for chunks.
    # 6. Store chunks and embeddings in Qdrant.
    # 7. Optionally, send a message to Kafka/RabbitMQ for further asynchronous processing.

    logger.info(f"Document '{filename if filename else 'text_content'}' received for processing (stub). Metadata: {metadata}")

    return ProcessingResponse(
        message="Document received for processing (stub implementation).",
        filename=filename,
        metadata_received=metadata,
        status="received_by_stub"
    )

@app.get("/health", tags=["Health"])
async def health_check():
    # Basic health check, can be expanded
    return {"status": "healthy", "service_type": "stub"}

if __name__ == "__main__":
    import uvicorn
    SERVICE_PORT = int(get_config_value("DOC_PROCESSING_SERVICE_PORT", default=8002))
    SERVICE_HOST = get_config_value("DOC_PROCESSING_SERVICE_HOST", default="0.0.0.0")

    logger.info(f"Starting Document Processing Service (Stub) on {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)

# To run this:
# python Scripts/services/doc_processing_service.py

# Example curl to test:
# curl -X POST "http://localhost:8002/process_document" \
# -F "file=@/path/to/your/sample.txt" \
# -F "metadata_json={\"source\":\"upload_test\", \"user\":\"jules\"}"
#
# Or with direct content:
# curl -X POST "http://localhost:8002/process_document" \
# -F "document_content=This is the text of the document." \
# -F "metadata_json={\"source\":\"direct_content_test\", \"user\":\"jules\"}"
