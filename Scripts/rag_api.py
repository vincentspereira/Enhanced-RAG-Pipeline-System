from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import uvicorn
import logging
from rag_pipeline import RAGPipeline
import json
import httpx
from datetime import datetime

# Import API routers
from api_v1.prompt_management_router import router as prompt_router
from api_v1.embedding_training_router import router as embedding_router
from api_v1.api_hub_router import router as api_hub_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API with GitHub Copilot Agent Integration")

# Include API routers
app.include_router(prompt_router, prefix="/api/v1")
app.include_router(embedding_router, prefix="/api/v1")
app.include_router(api_hub_router, prefix="/api/v1")

# Initialize the RAG pipeline
pipeline = RAGPipeline(collection_name="documents")

class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 5
    context: Optional[Dict[str, Any]] = None

class ProcessRequest(BaseModel):
    directory_path: str
    batch_size: Optional[int] = 32

class SearchResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float

class CopilotQuery(BaseModel):
    query: str
    conversation_id: Optional[str]
    context: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = 1000

class CopilotResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: str
    metadata: Dict[str, Any]

async def verify_copilot_token(x_copilot_token: str = Header(...)):
    """Verify GitHub Copilot Agent token."""
    if not x_copilot_token.startswith("gca_"):
        raise HTTPException(status_code=401, detail="Invalid Copilot Agent token")
    return x_copilot_token

@app.post("/search", response_model=List[SearchResult])
async def search(query: SearchQuery):
    """Search for documents using a query."""
    try:
        results = pipeline.search(query.query, limit=query.limit)
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_documents(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Process documents from a directory (async operation)."""
    try:
        path = Path(request.directory_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {path}")
        
        # Add document processing to background tasks
        background_tasks.add_task(pipeline.process_documents, path, request.batch_size)
        return {"message": f"Started processing documents from {path}"}
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get the current status of the RAG pipeline."""
    try:
        return pipeline.get_collection_info()
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/copilot/query", response_model=CopilotResponse)
async def copilot_query(
    query: CopilotQuery,
    token: str = Depends(verify_copilot_token)
):
    """
    GitHub Copilot Agent endpoint for RAG queries.
    
    Example request:
    ```json
    {
        "query": "What are the key implementation steps for XYZ feature?",
        "conversation_id": "optional-conversation-id",
        "context": {
            "project": "RAG Pipeline",
            "domain": "Technical Documentation"
        },
        "max_tokens": 1000
    }
    ```
    """
    try:
        # Get relevant documents
        results = pipeline.search(query.query, limit=5)
        
        # Format context for Copilot
        context = "\n\n".join([
            f"Source: {r.metadata.get('source', 'Unknown')}\n{r.text}"
            for r in results
        ])
        
        # Generate response using context
        response = {
            "answer": f"Based on the available documentation:\n\n{context}\n\nHere's the answer to your query...",
            "sources": [
                {
                    "text": r.text,
                    "metadata": r.metadata,
                    "score": r.score
                }
                for r in results
            ],
            "conversation_id": query.conversation_id or datetime.utcnow().isoformat(),
            "metadata": {
                "query_time": datetime.utcnow().isoformat(),
                "context_used": bool(results),
                **query.context if query.context else {}
            }
        }
        
        return CopilotResponse(**response)
        
    except Exception as e:
        logger.error(f"Copilot query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_response(request: dict):
    """Generate a response to a question using RAG."""
    try:
        # Extract parameters
        question = request.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        template_name = request.get("template_name", "qa_prompt")
        limit = request.get("limit", 5)
        
        # Get relevant documents
        results = pipeline.search(question, limit=limit)
        
        # Generate response using LLM
        response = pipeline.generate_response(
            question=question, 
            search_results=results, 
            template_name=template_name
        )
        
        return response
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
