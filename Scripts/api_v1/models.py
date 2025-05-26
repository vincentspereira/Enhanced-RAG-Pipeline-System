from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime

class WebhookEvent(BaseModel):
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class WebhookConfig(BaseModel):
    url: HttpUrl
    events: List[str]
    secret: Optional[str]
    is_active: bool = True
    created_at: datetime = None
    updated_at: datetime = None

class BulkDocumentRequest(BaseModel):
    documents: List[Dict[str, Any]]
    collection_name: Optional[str] = "default"
    batch_size: Optional[int] = 100
    synchronous: Optional[bool] = False

class BulkSearchRequest(BaseModel):
    queries: List[str]
    collection_name: Optional[str] = "default"
    limit: Optional[int] = 10
    with_vectors: Optional[bool] = False
    with_payload: Optional[bool] = True

class BulkResponse(BaseModel):
    success: bool
    total: int
    processed: int
    failed: int
    errors: Optional[List[Dict[str, Any]]] = None
    results: List[Dict[str, Any]]
