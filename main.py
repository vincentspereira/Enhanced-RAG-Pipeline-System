import asyncio
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import yaml
from pathlib import Path

from Scripts.rag_pipeline import EnhancedRAGPipeline

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(
    level=config["logging"]["level"],
    format=config["logging"]["format"],
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config["logging"]["file"])
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Enhanced RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
pipeline = EnhancedRAGPipeline()

# Pydantic models for request/response
class DocumentRequest(BaseModel):
    document_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    rerank: Optional[bool] = True

class FeedbackRequest(BaseModel):
    query_id: str
    document_id: str
    is_relevant: bool

@app.post("/documents")
async def process_document(request: DocumentRequest, background_tasks: BackgroundTasks):
    """Process a new document through the RAG pipeline."""
    try:
        # Process document asynchronously
        task_id = await pipeline.process_document(
            request.document_id,
            request.content,
            request.metadata
        )
        return {"status": "success", "message": f"Document processing started", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: SearchRequest):
    """Search for relevant documents."""
    try:
        results = await pipeline.search(
            request.query,
            top_k=request.top_k,
            rerank=request.rerank
        )
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def add_feedback(request: FeedbackRequest):
    """Add relevance feedback for active learning."""
    try:
        pipeline.add_relevance_feedback(
            request.query_id,
            request.document_id,
            request.is_relevant
        )
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Error adding feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get system status and statistics."""
    try:
        return {
            "status": "healthy",
            "stats": {
                "document_count": await pipeline.get_document_count(),
                "vector_count": await pipeline.get_vector_count(),
                "model_info": pipeline.get_model_info()
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point for the application."""
    try:
        # Create required directories
        Path(config["logging"]["file"]).parent.mkdir(exist_ok=True)
        Path(config["cache_dir"]).mkdir(exist_ok=True)
        
        # Start the API server
        uvicorn.run(
            "main:app",
            host=config["api"]["host"],
            port=config["api"]["port"],
            workers=config["api"]["workers"],
            timeout_keep_alive=config["api"]["timeout"],
            log_config=None  # Use our custom logging config
        )
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
