import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union

# Vector Store and Embedding Model Imports
from sentence_transformers import SentenceTransformer
import torch # For device selection
from Scripts.storage.vector_store_base import VectorStoreBase, DocumentChunk # Assuming DocumentChunk is defined here or imported by stores
from Scripts.storage.qdrant_vector_store import QdrantVectorStore
from Scripts.storage.weaviate_vector_store import WeaviateVectorStore
from Scripts.storage.milvus_vector_store import MilvusVectorStore
from Scripts.storage.chroma_vector_store import ChromaVectorStore
from Scripts.storage.faiss_vector_store import FaissVectorStore
import uuid # For generating document IDs
import fitz # PyMuPDF
from docx import Document as DocxDocument # For reading .docx files
from pptx import Presentation # For reading .pptx files
from openpyxl import load_workbook # For reading .xlsx files
import io # For reading file stream

# Configure logging
logger = logging.getLogger(__name__)
# BasicConfig should ideally be called only once at application entry point.
# Assuming it's called here if service is run standalone, or by main app otherwise.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import config loader
try:
    from Scripts.utils.config_loader import get_config_value
    from Scripts.utils.rabbitmq_producer import RabbitMQProducer # Import RabbitMQProducer
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from utils.config_loader import get_config_value
        from utils.rabbitmq_producer import RabbitMQProducer # Import RabbitMQProducer
    except ImportError as e:
        logger.error(f"Critical: Failed to import get_config_value or RabbitMQProducer for Document Processing Service. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None): # Basic fallback
            return os.getenv(env_var_name, default)
        # Dummy RabbitMQProducer if import fails
        class RabbitMQProducer:
            def __init__(self, *args, **kwargs): logger.error("Dummy RabbitMQProducer initialized.")
            def publish_message(self, *args, **kwargs): logger.warning("Dummy RabbitMQProducer: publish_message called.")
            def close(self): logger.info("Dummy RabbitMQProducer: close called.")


# --- Configuration ---
QDRANT_HOST = get_config_value("QDRANT_HOST", yaml_path="vector_store.qdrant.host", default="localhost")
QDRANT_PORT = int(get_config_value("QDRANT_PORT", yaml_path="vector_store.qdrant.port", default=6333))
QDRANT_COLLECTION_NAME = get_config_value("QDRANT_COLLECTION_NAME", yaml_path="vector_store.qdrant.collection_name", default="documents")
QDRANT_API_KEY = get_config_value("QDRANT_API_KEY", yaml_path="vector_store.qdrant.api_key")

EMBEDDING_MODEL_NAME = get_config_value("EMBEDDING_MODEL_NAME", yaml_path="model.embedding_model", default="all-MiniLM-L6-v2")
MODEL_DEVICE = get_config_value("MODEL_DEVICE", yaml_path="model.device", default="cpu")

# Chunking Configuration
CHUNK_SIZE = int(get_config_value("CHUNK_SIZE", yaml_path="document_processing.chunk_size", default=1000))
CHUNK_OVERLAP = int(get_config_value("CHUNK_OVERLAP", yaml_path="document_processing.chunk_overlap", default=200))


# --- Global Variables ---
app = FastAPI(title="Document Processing Service")
vector_store: Optional[VectorStoreBase] = None # Unified vector store instance
embedding_model: Optional[SentenceTransformer] = None
rabbitmq_producer: Optional[RabbitMQProducer] = None


# --- Pydantic Models ---
class ProcessingResponse(BaseModel):
    message: str
    filename: Optional[str] = None
    parent_document_id: Optional[Union[str, uuid.UUID]] = None # ID of the parent document
    chunks_indexed: int = 0
    metadata_processed: Optional[Dict[str, Any]] = None # Metadata of the parent document
    status: str


# --- Service Initialization and Shutdown ---
@app.on_event("startup")
async def startup_event():
    global vector_store, embedding_model, rabbitmq_producer
    logger.info("Document Processing Service starting up...")

    # Initialize Embedding Model First (to get vector_size if needed by some stores)
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on device: {MODEL_DEVICE}")
        device_to_use = MODEL_DEVICE
        if device_to_use == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA specified but not available. Falling back to CPU.")
            device_to_use = "cpu"

        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device_to_use)
        logger.info(f"Embedding model loaded successfully. Vector size: {embedding_model.get_sentence_embedding_dimension()}")
    except Exception as e:
        logger.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
        embedding_model = None # Crucial for health checks and subsequent ops
        # Depending on policy, might want to raise HTTPException or exit if embedding model is critical

    # Initialize Vector Store
    if embedding_model: # Only proceed if embedding model loaded, as we need vector_size
        provider = get_config_value("VECTOR_STORE_PROVIDER", yaml_path="vector_store.provider", default="qdrant")
        collection_name = get_config_value(f"VECTOR_STORE_{provider.upper()}_COLLECTION_NAME",
                                           yaml_path=f"vector_store.{provider}.collection_name",
                                           default="documents")
        vector_size = embedding_model.get_sentence_embedding_dimension()
        distance_metric = get_config_value(f"VECTOR_STORE_{provider.upper()}_DISTANCE",
                                           yaml_path=f"vector_store.{provider}.distance_metric",
                                           default="Cosine") # Default, specific stores might override

        logger.info(f"Initializing vector store provider: {provider}")
        global vector_store
        try:
            if provider == "qdrant":
                host = get_config_value("QDRANT_HOST", yaml_path="vector_store.qdrant.host", default="localhost")
                port = int(get_config_value("QDRANT_PORT", yaml_path="vector_store.qdrant.port", default=6333))
                api_key = get_config_value("QDRANT_API_KEY", yaml_path="vector_store.qdrant.api_key")
                vector_store = QdrantVectorStore(host=host, port=port, api_key=api_key)
            elif provider == "weaviate":
                url = get_config_value("WEAVIATE_URL", yaml_path="vector_store.weaviate.url", default="http://localhost:8080")
                api_key = get_config_value("WEAVIATE_API_KEY", yaml_path="vector_store.weaviate.api_key")
                vector_store = WeaviateVectorStore(url=url, api_key=api_key)
                collection_name = get_config_value(f"WEAVIATE_CLASS_NAME", yaml_path=f"vector_store.weaviate.class_name", default="Document") # Weaviate uses Class names
            elif provider == "milvus":
                host = get_config_value("MILVUS_HOST", yaml_path="vector_store.milvus.host", default="localhost")
                port = get_config_value("MILVUS_PORT", yaml_path="vector_store.milvus.port", default="19530")
                # Add user/password/alias if needed for Milvus from config
                vector_store = MilvusVectorStore(host=host, port=port)
            elif provider == "chroma":
                chroma_path = get_config_value("CHROMA_PATH", yaml_path="vector_store.chroma.path")
                chroma_host = get_config_value("CHROMA_HOST", yaml_path="vector_store.chroma.host")
                chroma_port = get_config_value("CHROMA_PORT", yaml_path="vector_store.chroma.port")
                if chroma_host and chroma_port: # HTTP client mode
                    vector_store = ChromaVectorStore(host=chroma_host, port=int(chroma_port))
                elif chroma_path: # Persistent client mode
                    vector_store = ChromaVectorStore(path=chroma_path)
                else: # Ephemeral (in-memory)
                    vector_store = ChromaVectorStore()
            elif provider == "faiss":
                index_path = get_config_value("FAISS_INDEX_PATH", yaml_path="vector_store.faiss.index_file_path", default="data/faiss_index.bin")
                metadata_path = get_config_value("FAISS_METADATA_PATH", yaml_path="vector_store.faiss.metadata_file_path", default="data/faiss_metadata.pkl")
                vector_store = FaissVectorStore(index_file_path=index_path, metadata_file_path=metadata_path)
            else:
                logger.error(f"Unsupported vector store provider: {provider}")
                raise ValueError(f"Unsupported vector store provider: {provider}")

            if vector_store:
                await vector_store.initialize(collection_name, vector_size, distance_metric)
                logger.info(f"Vector store '{provider}' initialized successfully for collection '{collection_name}'.")
            else: # Should have been caught by ValueError above
                 logger.error(f"Vector store provider '{provider}' could not be instantiated.")

        except Exception as e:
            logger.error(f"Failed to initialize vector store provider '{provider}': {e}", exc_info=True)
            vector_store = None # Ensure it's None if init fails
    else:
        logger.error("Embedding model failed to load. Vector store initialization skipped.")
        vector_store = None


    if not vector_store or not embedding_model: # Check unified vector_store now
        logger.error("Document Processing Service startup has issues: Vector Store or Embedding Model failed to initialize.")
    else:
        logger.info("Document Processing Service core components (Vector Store, Embedding Model) initialized.")

    # Initialize RabbitMQ Producer
    # global rabbitmq_producer # Already global
    try:
        # Configuration for RabbitMQ should ideally come from config_loader or env vars
        # Example: RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USER, RABBITMQ_PASSWORD, RABBITMQ_VHOST
        # For simplicity, using defaults or expecting environment setup if RabbitMQProducer handles it.
        rabbitmq_host = get_config_value("RABBITMQ_HOST", default="localhost")
        # rabbitmq_port = int(get_config_value("RABBITMQ_PORT", default=5672)) # pika default
        # rabbitmq_user = get_config_value("RABBITMQ_USER")
        # rabbitmq_password = get_config_value("RABBITMQ_PASSWORD")

        # Assuming RabbitMQProducer can be initialized with just the host or picks up from env
        rabbitmq_producer = RabbitMQProducer(host=rabbitmq_host) # Adjust constructor as needed
        logger.info("RabbitMQ Producer initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize RabbitMQ Producer: {e}", exc_info=True)
        rabbitmq_producer = None


@app.on_event("shutdown")
async def shutdown_event():
    global vector_store, rabbitmq_producer # Updated from qdrant_client
    logger.info("Document Processing Service shutting down...")
    if vector_store:
        try:
            await vector_store.close() # Use the new abstract close method
            logger.info("Vector store connection closed.")
        except Exception as e:
            logger.error(f"Error closing vector store: {e}", exc_info=True)

    if rabbitmq_producer:
        try:
            rabbitmq_producer.close()
            logger.info("RabbitMQ Producer connection closed.")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ Producer connection: {e}", exc_info=True)

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
        content_bytes = await file.read()

        if filename.lower().endswith(".txt"):
            try:
                text_to_embed = content_bytes.decode('utf-8')
                logger.info(f"File '{filename}' (.txt) read successfully (length: {len(text_to_embed)}).")
            except Exception as e:
                logger.error(f"Error decoding .txt file {filename}: {e}")
                raise HTTPException(status_code=400, detail=f"Could not decode .txt file: {str(e)}")
        elif filename.lower().endswith(".pdf"):
            try:
                pdf_document = fitz.open(stream=content_bytes, filetype="pdf")
                text_parts = []
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    text_parts.append(page.get_text("text"))
                text_to_embed = "\n".join(text_parts)
                pdf_document.close()
                if not text_to_embed.strip():
                    logger.warning(f"PDF file '{filename}' contained no extractable text.")
                    # text_to_embed will be empty or whitespace, handled later
                else:
                    logger.info(f"File '{filename}' (.pdf) text extracted successfully (length: {len(text_to_embed)}).")
            except Exception as e:
                logger.error(f"Error processing PDF file {filename}: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Could not process PDF file: {str(e)}")
        elif filename.lower().endswith(".docx"):
            try:
                # python-docx reads from a file-like object. BytesIO can wrap the byte stream.
                docx_file_stream = io.BytesIO(content_bytes)
                document = DocxDocument(docx_file_stream)
                text_parts = [para.text for para in document.paragraphs]
                text_to_embed = "\n".join(text_parts)
                if not text_to_embed.strip():
                    logger.warning(f"DOCX file '{filename}' contained no extractable text from paragraphs.")
                else:
                    logger.info(f"File '{filename}' (.docx) text extracted successfully (length: {len(text_to_embed)}).")
            except Exception as e:
                logger.error(f"Error processing DOCX file {filename}: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Could not process DOCX file: {str(e)}")
        elif filename.lower().endswith(".pptx"):
            try:
                pptx_file_stream = io.BytesIO(content_bytes)
                presentation = Presentation(pptx_file_stream)
                text_parts = []
                for slide in presentation.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text_frame") and shape.text_frame:
                            for paragraph in shape.text_frame.paragraphs:
                                for run in paragraph.runs:
                                    text_parts.append(run.text)
                        elif hasattr(shape, "text") and shape.text: # For shapes with direct text property
                             text_parts.append(shape.text)
                    # Extract text from slide notes if present
                    if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                        text_parts.append(slide.notes_slide.notes_text_frame.text)

                text_to_embed = "\n".join(filter(None, text_parts)) # Filter out empty strings
                if not text_to_embed.strip():
                    logger.warning(f"PPTX file '{filename}' contained no extractable text.")
                else:
                    logger.info(f"File '{filename}' (.pptx) text extracted successfully (length: {len(text_to_embed)}).")
            except Exception as e:
                logger.error(f"Error processing PPTX file {filename}: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Could not process PPTX file: {str(e)}")
        elif filename.lower().endswith(".xlsx"):
            try:
                xlsx_file_stream = io.BytesIO(content_bytes)
                workbook = load_workbook(filename=xlsx_file_stream, read_only=True, data_only=True)
                text_parts = []
                for sheet_name in workbook.sheetnames:
                    sheet = workbook[sheet_name]
                    for row in sheet.iter_rows():
                        for cell in row:
                            if cell.value is not None and isinstance(cell.value, (str, int, float)):
                                text_parts.append(str(cell.value))
                text_to_embed = "\n".join(filter(None, text_parts))
                if not text_to_embed.strip():
                    logger.warning(f"XLSX file '{filename}' contained no extractable text content.")
                else:
                    logger.info(f"File '{filename}' (.xlsx) text extracted successfully (length: {len(text_to_embed)}).")
            except Exception as e:
                logger.error(f"Error processing XLSX file {filename}: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Could not process XLSX file: {str(e)}")
        else:
            logger.warning(f"Received file '{filename}' is not a .txt, .pdf, .docx, .pptx, or .xlsx file. This iteration only supports these for content processing.")
            # Metadata will still be parsed and returned, but no indexing will occur if text_to_embed is None.

    elif document_content:
        text_to_embed = document_content
        logger.info(f"Text content received (length: {len(text_to_embed)}).")
    else:
        raise HTTPException(status_code=400, detail="Either a '.txt', '.pdf', '.docx', '.pptx', '.xlsx' file or 'document_content' must be provided for processing.")

    parsed_metadata = {}
    try:
        import json
        parsed_metadata = json.loads(metadata_json)
        logger.info(f"Received metadata: {parsed_metadata}")
    except json.JSONDecodeError:
        logger.error(f"Failed to parse metadata_json: {metadata_json}")
        raise HTTPException(status_code=400, detail="Invalid JSON format for metadata.")

    # --- Actual processing and Vector Store indexing ---
    if text_to_embed and text_to_embed.strip(): # Only proceed if we have non-empty text content
        if not vector_store or not embedding_model: # Check unified vector_store
            logger.error("Cannot process document: Vector Store or Embedding Model not initialized.")
            raise HTTPException(status_code=503, detail="Service not ready to process and index documents.")
        try:
            # 1. Generate parent document ID and basic keywords/TFs from full text
            parent_doc_id = parsed_metadata.get("document_id") or str(uuid.uuid4())
            full_text_keywords = []
            full_text_term_frequencies = {}

            try:
                import re
                full_text_words = re.findall(r'\b\w+\b', text_to_embed.lower())
                stop_words = set(["the", "a", "is", "in", "it", "to", "of", "and", "for", "on", "with", "this", "that", "an", "by", "as", "at", "or", "if", "not", "be"]) # Basic list
                full_text_keywords = list(set(word for word in full_text_words if word not in stop_words and len(word) > 2 and not word.isdigit()))[:150]
                full_text_term_frequencies = {kw: full_text_words.count(kw) for kw in full_text_keywords}
                logger.info(f"Generated {len(full_text_keywords)} keywords and TFs for parent doc {parent_doc_id}.")
            except Exception as kw_e:
                logger.warning(f"Could not generate keywords/TFs for parent doc {parent_doc_id}: {kw_e}")

            # Store these parent-level keywords and TFs in the base metadata
            # Ensure metadata is a dict
            if not isinstance(parsed_metadata, dict):
                logger.warning(f"Parsed metadata is not a dictionary: {parsed_metadata}. Resetting to empty dict.")
                parsed_metadata = {}

            parent_metadata = parsed_metadata.copy()
            parent_metadata["_internal_id"] = parent_doc_id # This is the ID for the overall document
            parent_metadata["keywords"] = full_text_keywords
            parent_metadata["term_frequencies"] = full_text_term_frequencies
            if filename:
                parent_metadata["original_filename"] = filename
                if "source" not in parent_metadata:
                    parent_metadata["source"] = filename


            # 2. Chunk the document text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_text(text_to_embed)
            logger.info(f"Split document {parent_doc_id} into {len(chunks)} chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}).")

            if not chunks:
                logger.warning(f"No chunks generated for document {parent_doc_id}. Text might be too short or empty.")
                return ProcessingResponse(
                    message="Document processed, but no text chunks were generated (text might be too short). Metadata logged.",
                    filename=filename,
                    parent_document_id=parent_doc_id,
                    chunks_indexed=0,
                    metadata_processed=parent_metadata,
                    status="processed_no_chunks"
                )

            points_to_upsert = []
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{parent_doc_id}_chunk_{i}"
                logger.debug(f"Generating embedding for chunk {i} of document {parent_doc_id}...")
                embedding = embedding_model.encode(chunk_text, convert_to_tensor=False).tolist()

                # Each chunk point stores its own text, and references parent metadata
                chunk_payload = {
                    "chunk_text": chunk_text,
                    "parent_document_id": parent_doc_id,
                    "chunk_sequence_number": i,
                    "metadata": parent_metadata # Parent's full metadata including keywords, TFs, source etc.
                }
                # Using DocumentChunk from vector_store_base
                document_chunks_to_add.append(DocumentChunk(
                    id=chunk_id, # Ensure this ID is unique and suitable for the chosen DB
                    text=chunk_text,
                    vector=embedding,
                    metadata=chunk_metadata
                ))

            logger.info(f"Prepared {len(document_chunks_to_add)} document chunks for upsertion for parent document {parent_doc_id}.")

            # Upsert all document chunks to the configured vector store
            if document_chunks_to_add:
                # Determine collection name from config, as it might vary per provider
                provider = get_config_value("VECTOR_STORE_PROVIDER", yaml_path="vector_store.provider", default="qdrant")
                collection_name_cfg_key = f"vector_store.{provider}.collection_name"
                # For Weaviate, it's class_name
                if provider == "weaviate":
                    collection_name_cfg_key = f"vector_store.weaviate.class_name"

                target_collection_name = get_config_value(f"VECTOR_STORE_{provider.upper()}_COLLECTION_NAME", # Env var
                                           yaml_path=collection_name_cfg_key,
                                           default="documents")

                added_ids = await vector_store.add_documents(
                    collection_name=target_collection_name,
                    documents=document_chunks_to_add
                    # Add other provider-specific args from **kwargs if needed by specific implementations
                )
                logger.info(f"{len(added_ids)} chunks for document '{parent_doc_id}' successfully processed by vector store '{provider}'.")
            else:
                logger.info(f"No chunks to add for document {parent_doc_id}.")


            # Publish event to RabbitMQ
            if rabbitmq_producer:
                event_message = {
                    "event_type": "document_processed",
                    "parent_document_id": str(parent_doc_id),
                    "filename": filename,
                    "chunks_indexed": len(chunks),
                    "status": "indexed_chunked",
                    "timestamp": datetime.utcnow().isoformat()
                }
                # Define your exchange and routing key based on your RabbitMQ setup
                # For this example, using a direct exchange (default) and routing key 'doc_events'
                # Queue 'document_processing_events' should be bound to this routing key on an exchange.
                # This assumes RabbitMQProducer's publish_message handles exchange/routing_key declaration or uses defaults.
                try:
                    rabbitmq_producer.publish_message(
                        exchange_name='', # Default exchange
                        routing_key='document_processing_events', # Queue name
                        message_body=json.dumps(event_message)
                    )
                    logger.info(f"Published 'document_processed' event for {parent_doc_id} to RabbitMQ.")
                except Exception as mq_e:
                    logger.error(f"Failed to publish document_processed event to RabbitMQ for {parent_doc_id}: {mq_e}", exc_info=True)
            else:
                logger.warning("RabbitMQ producer not available. Skipping event publishing.")


            return ProcessingResponse(
                message=f"Document processed. {len(chunks)} chunks indexed.",
                filename=filename,
                parent_document_id=parent_doc_id,
                chunks_indexed=len(chunks),
                metadata_processed=parent_metadata, # Return the parent metadata
                status="indexed_chunked"
            )

        except Exception as e:
            logger.error(f"Error during document embedding or Qdrant indexing: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    else:
        # This case handles:
        # This case handles:
        # 1. Files of unsupported types (text_to_embed remains None)
        # 2. Supported files that yielded no text (text_to_embed is empty or whitespace)
        # 3. Direct document_content that was empty or whitespace
        status_message = "Document metadata received. "
        supported_types = [".txt", ".pdf", ".docx", ".pptx", ".xlsx"] # Added .xlsx
        file_type_supported = False
        if filename:
            for ext in supported_types:
                if filename.lower().endswith(ext):
                    file_type_supported = True
                    break

        if filename and not file_type_supported:
            status_message += f"File '{filename}' type not supported for content extraction. Supported: {', '.join(supported_types)}."
        elif text_to_embed is not None and not text_to_embed.strip(): # Content was extracted/provided but is empty
             status_message += "No processable text content found in the document for indexing."
        elif text_to_embed is None and filename and file_type_supported : # File was of supported type but extraction failed/empty
             status_message += f"File '{filename}' was of a supported type but no text could be extracted or content was empty."
        elif text_to_embed is None and filename and not file_type_supported: # Explicitly state not supported
             status_message += f"File '{filename}' type not supported for content extraction. Supported: {', '.join(supported_types)}."
        else: # Only metadata_json was provided, or document_content was None/empty
            status_message += "No text content provided or extracted for indexing."

        logger.info(status_message + f" Metadata: {parsed_metadata}")
        return ProcessingResponse(
            message=status_message,
            filename=filename,
            metadata_processed=parsed_metadata,
            status="received_metadata_only" # Or a more specific status
        )


@app.get("/health", tags=["Health"])
async def health_check():
    em_ready = bool(embedding_model)
    vs_ready = False
    vs_provider = "N/A"

    if vector_store:
        vs_ready = await vector_store.health_check()
        vs_provider = get_config_value("VECTOR_STORE_PROVIDER", yaml_path="vector_store.provider", default="unknown")

    overall_status = "healthy"
    if not em_ready or not vs_ready:
        overall_status = "degraded"
        if not em_ready: logger.warning("Health Check: Embedding model not ready.")
        if not vs_ready: logger.warning(f"Health Check: Vector store '{vs_provider}' not ready.")


    return {
        "status": overall_status,
        "service_type": "document_processor",
        "components": {
            "embedding_model_initialized": em_ready,
            "vector_store_provider": vs_provider,
            "vector_store_healthy": vs_ready,
            "rabbitmq_producer_connected": rabbitmq_producer is not None and rabbitmq_producer.channel is not None and rabbitmq_producer.channel.is_open
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
