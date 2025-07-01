import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union

# Qdrant and SentenceTransformer imports
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer
import torch # For device selection
import uuid # For generating document IDs

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

# --- Configuration ---
QDRANT_HOST = get_config_value("QDRANT_HOST", yaml_path="vector_store.qdrant.host", default="localhost")
QDRANT_PORT = int(get_config_value("QDRANT_PORT", yaml_path="vector_store.qdrant.port", default=6333))
QDRANT_COLLECTION_NAME = get_config_value("QDRANT_COLLECTION_NAME", yaml_path="vector_store.qdrant.collection_name", default="documents")
QDRANT_API_KEY = get_config_value("QDRANT_API_KEY", yaml_path="vector_store.qdrant.api_key")

EMBEDDING_MODEL_NAME = get_config_value("EMBEDDING_MODEL_NAME", yaml_path="model.embedding_model", default="all-MiniLM-L6-v2")
MODEL_DEVICE = get_config_value("MODEL_DEVICE", yaml_path="model.device", default="cpu")


# --- Global Variables ---
app = FastAPI(title="Document Processing Service")
qdrant_client: Optional[QdrantClient] = None
embedding_model: Optional[SentenceTransformer] = None


# --- Pydantic Models ---
class ProcessingResponse(BaseModel):
    message: str
    filename: Optional[str] = None
    qdrant_id: Optional[Union[str, int]] = None # ID used in Qdrant
    metadata_processed: Optional[Dict[str, Any]] = None
    status: str


# --- Service Initialization and Shutdown ---
@app.on_event("startup")
async def startup_event():
    global qdrant_client, embedding_model
    logger.info("Document Processing Service starting up...")

    # Initialize Qdrant Client
    try:
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )
        # Check if collection exists, optionally create it.
        # For this service, we assume the collection might need to be created if it doesn't exist,
        # or at least log a clear warning.
        try:
            qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            logger.info(f"Successfully connected to Qdrant. Collection '{QDRANT_COLLECTION_NAME}' exists.")
        except Exception as e: # Broad exception as Qdrant client might raise different errors for non-existent collection
            logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found or connection error: {e}. Attempting to create it.")
            try:
                # Determine vector size from the embedding model
                temp_model_for_size = SentenceTransformer(EMBEDDING_MODEL_NAME)
                vector_size = temp_model_for_size.get_sentence_embedding_dimension()
                del temp_model_for_size # free memory

                qdrant_client.recreate_collection( # Use recreate_collection for idempotency
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE)
                )
                logger.info(f"Successfully created Qdrant collection '{QDRANT_COLLECTION_NAME}' with vector size {vector_size}.")
            except Exception as creation_e:
                logger.error(f"Failed to create Qdrant collection '{QDRANT_COLLECTION_NAME}': {creation_e}")
                qdrant_client = None # Mark as not initialized if collection handling fails
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        qdrant_client = None

    # Initialize Embedding Model
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on device: {MODEL_DEVICE}")
        device_to_use = MODEL_DEVICE
        if device_to_use == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA specified but not available. Falling back to CPU.")
            device_to_use = "cpu"

        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device_to_use)
        logger.info(f"Embedding model loaded successfully. Vector size: {embedding_model.get_sentence_embedding_dimension()}")
    except Exception as e:
        logger.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        embedding_model = None

    if not qdrant_client or not embedding_model:
        logger.error("Document Processing Service startup has issues: one or more critical components failed to initialize.")
    else:
        logger.info("Document Processing Service startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    global qdrant_client
    logger.info("Document Processing Service shutting down...")
    if qdrant_client:
        try:
            # Qdrant client typically doesn't require explicit close for HTTP.
            pass
        except Exception as e:
            logger.error(f"Error during Qdrant client cleanup (if any): {e}")
    logger.info("Document Processing Service shutdown complete.")


# --- API Endpoints ---
@app.post("/process_document", response_model=ProcessingResponse, tags=["Processing"])
async def process_document_endpoint(
    file: UploadFile = File(None, description="The document file to process (.txt supported initially)."),
    document_content: Optional[str] = Form(None, description="Text content of the document if not sending a file."),
    metadata_json: Optional[str] = Form("{}", description="JSON string of document metadata.")
):
    filename = None
    text_to_embed = None
    doc_id_for_qdrant = None

    if file:
        filename = file.filename
        logger.info(f"Received file: {filename}, content type: {file.content_type}")

        if not filename.endswith(".txt"): # Basic check for this iteration
            logger.warning(f"Received file '{filename}' is not a .txt file. This iteration only supports .txt for content processing.")
            # Metadata will still be parsed and returned, but no indexing will occur.
        else:
            try:
                content_bytes = await file.read()
                text_to_embed = content_bytes.decode('utf-8') # Assuming UTF-8 for .txt
                logger.info(f"File '{filename}' read successfully (length: {len(text_to_embed)}).")
            except Exception as e:
                logger.error(f"Error reading or decoding file {filename}: {e}")
                raise HTTPException(status_code=400, detail=f"Could not read or decode file: {str(e)}")
    elif document_content:
        text_to_embed = document_content
        logger.info(f"Text content received (length: {len(text_to_embed)}).")
    else:
        # If neither file nor content, it's an error, but allow metadata-only if a non-txt file was given.
        # This logic is a bit convoluted due to supporting metadata-only for non-txt.
        # Cleaner would be to reject non-txt outright or have separate endpoints.
        if not filename: # No file at all was provided
             raise HTTPException(status_code=400, detail="Either a '.txt' file or 'document_content' must be provided for processing.")

    parsed_metadata = {}
    try:
        import json
        parsed_metadata = json.loads(metadata_json)
        logger.info(f"Received metadata: {parsed_metadata}")
    except json.JSONDecodeError:
        logger.error(f"Failed to parse metadata_json: {metadata_json}")
        raise HTTPException(status_code=400, detail="Invalid JSON format for metadata.")

    # --- Actual processing and Qdrant indexing ---
    if text_to_embed: # Only proceed if we have text content from .txt or direct input
        if not qdrant_client or not embedding_model:
            logger.error("Cannot process document: Qdrant client or embedding model not initialized.")
            raise HTTPException(status_code=503, detail="Service not ready to process and index documents.")
        try:
            logger.info("Generating embedding for the document content...")
            # For simplicity, embedding the whole text. Chunking would be done here in a real scenario.
            embedding = embedding_model.encode(text_to_embed, convert_to_tensor=False).tolist()
            logger.info(f"Embedding generated. Vector dimension: {len(embedding)}")

            # Prepare payload for Qdrant
            # Ensure metadata is a dictionary
            if not isinstance(parsed_metadata, dict):
                logger.warning(f"Parsed metadata is not a dictionary: {parsed_metadata}. Resetting to empty dict.")
                parsed_metadata = {}

            payload_for_qdrant = {"text": text_to_embed, "metadata": parsed_metadata.copy()} # Use a copy

            if filename:
                payload_for_qdrant["metadata"]["original_filename"] = filename
                if "source" not in payload_for_qdrant["metadata"]: # Don't overwrite if 'source' was in parsed_metadata
                     payload_for_qdrant["metadata"]["source"] = filename

            # Use provided document_id from metadata or generate a new one
            doc_id_for_qdrant = parsed_metadata.get("document_id") or str(uuid.uuid4())
            # Store the used ID back into the metadata payload for consistency
            payload_for_qdrant["metadata"]["_internal_id"] = doc_id_for_qdrant


            points_to_upsert = [
                qdrant_models.PointStruct(
                    id=doc_id_for_qdrant,
                    vector=embedding,
                    payload=payload_for_qdrant
                )
            ]

            logger.info(f"Upserting document to Qdrant collection '{QDRANT_COLLECTION_NAME}' with ID: {doc_id_for_qdrant}")
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points_to_upsert,
                wait=True
            )
            logger.info(f"Document '{doc_id_for_qdrant}' successfully indexed into Qdrant.")

            return ProcessingResponse(
                message="Document processed and indexed successfully.",
                filename=filename,
                qdrant_id=doc_id_for_qdrant,
                metadata_processed=payload_for_qdrant["metadata"],
                status="indexed"
            )

        except Exception as e:
            logger.error(f"Error during document embedding or Qdrant indexing: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    else:
        # This case handles non .txt files where text_to_embed was not set, or empty content.
        status_message = "Document metadata received. "
        if filename and not filename.endswith(".txt"):
            status_message += f"File '{filename}' was not a .txt, so content not processed for indexing."
        elif not text_to_embed and (filename or document_content is not None): # Content was explicitly empty or not provided from file
             status_message += "No text content provided or extracted for indexing."
        else: # Only metadata_json was provided, no file or content
            status_message += "No file or document_content provided."


        logger.info(status_message + f" Metadata: {parsed_metadata}")
        return ProcessingResponse(
            message=status_message,
            filename=filename,
            metadata_processed=parsed_metadata,
            status="received_metadata_only"
        )


@app.get("/health", tags=["Health"])
async def health_check():
    q_ready = bool(qdrant_client)
    em_ready = bool(embedding_model)

    # Basic health check, can be expanded to check Qdrant/model actual responsiveness
    if q_ready and em_ready:
        status = "healthy"
        # Optionally, try a lightweight Qdrant operation if concerned about staleness
        # try:
        #     qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        # except Exception:
        #     status = "degraded" # Qdrant might be down
    else:
        status = "degraded"

    return {
        "status": status,
        "service_type": "document_processor",
        "components": {
            "qdrant_initialized": q_ready,
            "embedding_model_initialized": em_ready
        }
    }

if __name__ == "__main__":
    import uvicorn
    SERVICE_PORT = int(get_config_value("DOC_PROCESSING_SERVICE_PORT", default=8002))
    SERVICE_HOST = get_config_value("DOC_PROCESSING_SERVICE_HOST", default="0.0.0.0")

    logger.info(f"Starting Document Processing Service on {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)

# To run this:
# python Scripts/services/doc_processing_service.py

# Example curl to test:
# curl -X POST "http://localhost:8002/process_document" \
# -F "file=@/path/to/your/sample.txt" \
# -F "metadata_json={\"source\":\"upload_test\", \"user\":\"jules\", \"document_id\":\"my-custom-id-123\"}"
#
# Or with direct content:
# curl -X POST "http://localhost:8002/process_document" \
# -F "document_content=This is the text of the document." \
# -F "metadata_json={\"source\":\"direct_content_test\", \"user\":\"jules\"}"
#
# To test non-txt file (metadata only):
# echo "pdf content" > sample.pdf
# curl -X POST "http://localhost:8002/process_document" \
# -F "file=@sample.pdf" \
# -F "metadata_json={\"source\":\"pdf_test\"}"
