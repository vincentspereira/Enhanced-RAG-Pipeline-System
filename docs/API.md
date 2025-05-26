# API Documentation

This document provides detailed information about the Enhanced RAG Pipeline API endpoints, their usage, and examples.

## Base URL

```
http://localhost:8000
```

## Authentication

API uses JWT authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_token>
```

## Endpoints

### Document Management

#### 1. Process Document

Process and index a new document.

```http
POST /documents
Content-Type: application/json
Authorization: Bearer <token>

{
    "document_id": "string",
    "content": "string",
    "metadata": {
        "source": "string",
        "author": "string",
        "date": "string",
        "tags": ["string"]
    }
}
```

Response:
```json
{
    "status": "success",
    "message": "Document processed successfully",
    "task_id": "string"
}
```

#### 2. Get Document Status

Check document processing status.

```http
GET /documents/{document_id}/status
Authorization: Bearer <token>
```

Response:
```json
{
    "status": "completed|processing|failed",
    "progress": 0.85,
    "error": "string|null"
}
```

### Search

#### 1. Search Documents

Search for relevant documents.

```http
POST /search
Content-Type: application/json
Authorization: Bearer <token>

{
    "query": "string",
    "top_k": 5,
    "rerank": true,
    "filters": {
        "date_range": {
            "start": "2025-01-01",
            "end": "2025-12-31"
        },
        "metadata": {
            "author": "string",
            "tags": ["string"]
        }
    }
}
```

Response:
```json
{
    "results": [
        {
            "document_id": "string",
            "score": 0.95,
            "content": "string",
            "metadata": {
                "source": "string",
                "author": "string",
                "date": "string"
            },
            "relevance_score": 0.88
        }
    ],
    "total": 10,
    "processing_time": 0.15
}
```

### Feedback

#### 1. Add Relevance Feedback

Submit relevance feedback for search results.

```http
POST /feedback
Content-Type: application/json
Authorization: Bearer <token>

{
    "query_id": "string",
    "document_id": "string",
    "is_relevant": true,
    "feedback_source": "user",
    "confidence": 1.0
}
```

Response:
```json
{
    "status": "success",
    "message": "Feedback recorded"
}
```

### Categories

#### 1. Get Document Categories

Retrieve categories for a document.

```http
GET /documents/{document_id}/categories
Authorization: Bearer <token>
```

Response:
```json
{
    "categories": [
        {
            "id": "string",
            "name": "string",
            "confidence": 0.92,
            "subcategories": [
                {
                    "id": "string",
                    "name": "string",
                    "confidence": 0.85
                }
            ]
        }
    ]
}
```

### System Management

#### 1. System Status

Get system status and metrics.

```http
GET /status
Authorization: Bearer <token>
```

Response:
```json
{
    "status": "healthy",
    "uptime": 3600,
    "document_count": 1000,
    "vector_count": 5000,
    "gpu_utilization": 0.75,
    "memory_usage": 0.65
}
```

## Error Handling

### Error Response Format

```json
{
    "error": {
        "code": "string",
        "message": "string",
        "details": {}
    }
}
```

### Common Error Codes

- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error

## Rate Limiting

API endpoints are rate-limited to:
- 100 requests per minute for document processing
- 1000 requests per minute for search
- 5000 requests per minute for feedback

## Examples

### Python Client Example

```python
import requests
import json

class RAGClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def process_document(self, document_id, content, metadata=None):
        url = f"{self.base_url}/documents"
        payload = {
            "document_id": document_id,
            "content": content,
            "metadata": metadata or {}
        }
        response = requests.post(url, json=payload, headers=self.headers)
        return response.json()
    
    def search(self, query, top_k=5, rerank=True):
        url = f"{self.base_url}/search"
        payload = {
            "query": query,
            "top_k": top_k,
            "rerank": rerank
        }
        response = requests.post(url, json=payload, headers=self.headers)
        return response.json()

# Usage
client = RAGClient("http://localhost:8000", "your_api_key")

# Process document
result = client.process_document(
    "doc1",
    "Document content",
    {"author": "John Doe"}
)

# Search
results = client.search("search query", top_k=5)
```

## Best Practices

1. **Rate Limiting**
   - Implement exponential backoff
   - Cache frequently accessed results
   - Batch requests when possible

2. **Error Handling**
   - Always check response status
   - Implement proper error handling
   - Log errors for debugging

3. **Performance**
   - Use batch operations for multiple documents
   - Enable result caching
   - Monitor API usage

4. **Security**
   - Keep API keys secure
   - Use HTTPS in production
   - Validate input data
