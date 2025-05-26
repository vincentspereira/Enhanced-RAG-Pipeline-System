# Streaming and LangChain Integration Guide

This guide demonstrates how to use the streaming and LangChain integration features of the Qdrant RAG system.

## Table of Contents
1. [Basic Setup](#basic-setup)
2. [Streaming Responses](#streaming-responses)
3. [LangChain Integration](#langchain-integration)
4. [Combined Usage](#combined-usage)
5. [Advanced Features](#advanced-features)

## Basic Setup

First, ensure you have all required dependencies:

```bash
pip install qdrant-client langchain openai aiohttp websockets
```

## Streaming Responses

### Server-Side Events (SSE)

```python
from streaming.response_stream import stream_manager, StreamEvent, StreamEventType
from streaming.client import StreamClient, StreamClientConfig

# Initialize client
client = StreamClient(StreamClientConfig(base_url="http://localhost:8000"))

# Create a stream
stream_id = await client.create_stream()

# Subscribe to updates
async for event in client.subscribe_sse(stream_id):
    if event["type"] == "data":
        print(f"Received data: {event['data']}")
    elif event["type"] == "progress":
        progress = event["data"]
        print(f"Progress: {progress['percentage']}%")
    elif event["type"] == "complete":
        break
```

### WebSocket

```python
# Using WebSocket instead of SSE
async for event in client.subscribe_websocket(stream_id):
    # Handle events same as SSE
    pass
```

### Custom Progress Tracking

```python
# Create a custom progress callback
progress_callback = client.create_progress_callback(
    description="Processing documents",
    show_percentage=True
)

async for event in client.subscribe_sse(stream_id, progress_callback):
    if event["type"] == "complete":
        break
```

## LangChain Integration

### Setting Up Vector Store

```python
from integrations.langchain_integration import (
    QdrantVectorStore,
    LangChainConfig
)

# Initialize vector store
config = LangChainConfig(
    chunk_size=500,
    chunk_overlap=50,
    embedding_batch_size=100
)

store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="documents",
    embeddings=embeddings,
    config=config
)

# Add documents with progress streaming
stream_id = str(uuid4())
await stream_manager.create_stream(stream_id)

store.add_texts(
    texts=documents,
    stream_id=stream_id
)
```

### Streaming Search Results

```python
# Search with streaming updates
results = store.similarity_search_with_score(
    query="What is Qdrant?",
    k=4,
    stream_id=stream_id
)
```

## Combined Usage

### Question Answering System

```python
from streaming_qa import StreamingQA

# Initialize QA system
qa = StreamingQA(
    qdrant_client=client,
    collection_name="documents",
    openai_api_key="your-api-key"
)

# Process documents
documents = [
    "Qdrant is a vector similarity search engine.",
    "It provides a production-ready service with HTTP API."
]

process_stream_id = await qa.process_documents(documents)

# Subscribe to processing updates
client = StreamClient()
async for event in client.subscribe_sse(process_stream_id):
    if event["type"] == "progress":
        print(f"Processing: {event['data']['percentage']}%")
    elif event["type"] == "complete":
        break

# Ask a question
question = "What is Qdrant?"
answer_stream_id = await qa.answer_question(question)

# Subscribe to answer stream
token_callback = client.create_token_callback()
async for event in client.subscribe_websocket(answer_stream_id, token_callback):
    if event["type"] == "complete":
        break
```

## Advanced Features

### Custom Callback Handlers

```python
from integrations.langchain_integration import StreamingCallbackHandler

class CustomCallbackHandler(StreamingCallbackHandler):
    async def on_llm_new_token(self, token: str, **kwargs):
        # Custom token handling
        await stream_manager.push_event(
            self.stream_id,
            StreamEvent(StreamEventType.DATA, {
                "type": "custom_token",
                "content": token.upper()
            })
        )
```

### Error Handling

```python
try:
    async for event in client.subscribe_sse(stream_id):
        if event["type"] == "error":
            print(f"Error: {event['data']['error']}")
            break
        # Handle other events
except Exception as e:
    print(f"Connection error: {e}")
```

### Batch Processing

```python
# Process large document collections in batches
from typing import List

async def process_document_batch(
    documents: List[str],
    batch_size: int = 100
) -> List[str]:
    stream_ids = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        stream_id = await qa.process_documents(batch)
        stream_ids.append(stream_id)
    
    return stream_ids
```

### Custom Streaming Patterns

```python
# Broadcast to multiple streams
async def broadcast_progress(
    message: str,
    progress: float,
    stream_ids: List[str]
):
    event = StreamEvent(
        StreamEventType.PROGRESS,
        {
            "message": message,
            "progress": progress
        }
    )
    
    for stream_id in stream_ids:
        await stream_manager.push_event(stream_id, event)
```

## Best Practices

1. Always handle connection errors and retries
2. Use appropriate timeouts for long-running operations
3. Clean up streams when they're no longer needed
4. Monitor memory usage with large document collections
5. Implement proper error handling and recovery
6. Use batch processing for large datasets
7. Implement proper request validation
8. Add appropriate logging and monitoring
