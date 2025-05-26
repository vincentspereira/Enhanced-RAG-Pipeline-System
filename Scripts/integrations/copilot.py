"""
GitHub Copilot Agent integration for RAG pipeline.
"""
from typing import List, Dict, Any, Optional
import logging
import json
import asyncio
import aiohttp
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CopilotRequest(BaseModel):
    """Request model for GitHub Copilot Agent interactions."""
    query: str
    context: List[Dict[str, Any]] = Field(default_factory=list)
    max_context_length: Optional[int] = 4000
    temperature: Optional[float] = 0.7
    stream: bool = False

class CopilotResponse(BaseModel):
    """Response model for GitHub Copilot Agent interactions."""
    response: str
    context_used: List[Dict[str, Any]]
    tokens_used: int
    model: str

class CopilotAgent:
    """GitHub Copilot Agent integration class."""
    
    def __init__(self, api_key: str, endpoint: str = "https://api.githubcopilot.com/chat/completions"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.session = None
    
    async def __aenter__(self):
        """Set up async context with aiohttp session."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "RAG-Pipeline/1.0",
                "Accept": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context."""
        if self.session:
            await self.session.close()
            self.session = None

    def _prepare_messages(self, request: CopilotRequest) -> List[Dict[str, str]]:
        """Prepare messages for the Copilot API."""
        messages = []
        
        # Add context as system messages
        total_length = 0
        for ctx in request.context:
            ctx_text = json.dumps(ctx)
            if request.max_context_length and (total_length + len(ctx_text)) > request.max_context_length:
                break
            messages.append({
                "role": "system",
                "content": ctx_text
            })
            total_length += len(ctx_text)
            
        # Add user query
        messages.append({
            "role": "user",
            "content": request.query
        })
        
        return messages

    async def _make_request(self, messages: List[Dict[str, str]], 
                          temperature: float, stream: bool) -> Dict[str, Any]:
        """Make request to Copilot API."""
        if not self.session:
            raise RuntimeError("CopilotAgent must be used as an async context manager")
            
        async with self.session.post(
            self.endpoint,
            json={
                "messages": messages,
                "temperature": temperature,
                "stream": stream,
                "model": "copilot-chat"  # or other available models
            }
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Copilot API error: {error_text}")
                
            return await response.json()

    async def get_completion(self, request: CopilotRequest) -> CopilotResponse:
        """Get completion from GitHub Copilot Agent."""
        messages = self._prepare_messages(request)
        
        response_data = await self._make_request(
            messages=messages,
            temperature=request.temperature or 0.7,
            stream=request.stream
        )
        
        return CopilotResponse(
            response=response_data["choices"][0]["message"]["content"],
            context_used=request.context,
            tokens_used=response_data["usage"]["total_tokens"],
            model=response_data["model"]
        )

    async def stream_completion(self, request: CopilotRequest):
        """Stream completion from GitHub Copilot Agent."""
        if not request.stream:
            request.stream = True
            
        messages = self._prepare_messages(request)
        
        async with self.session.post(
            self.endpoint,
            json={
                "messages": messages,
                "temperature": request.temperature or 0.7,
                "stream": True,
                "model": "copilot-chat"
            }
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Copilot API error: {error_text}")
                
            async for line in response.content:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "choices" in data and data["choices"]:
                            yield data["choices"][0]["delta"].get("content", "")
                    except json.JSONDecodeError:
                        continue