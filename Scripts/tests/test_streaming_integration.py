import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import json
import os
from typing import List, Dict, Any

from ..streaming.response_stream import (
    stream_manager,
    StreamEvent,
    StreamEventType
)
from ..streaming.client import StreamClient, StreamClientConfig
from ..integrations.langchain_integration import (
    QdrantVectorStore,
    StreamingCallbackHandler,
    StreamingRetriever,
    LangChainConfig
)
from ..streaming_qa import StreamingQA

# Mock Qdrant client for testing
class MockQdrantClient:
    def __init__(self):
        self.points = []
        self.search_results = []
    
    async def upload_points(self, collection_name: str, points: List[Dict[str, Any]]):
        self.points.extend(points)
        return True
    
    async def search(self, collection_name: str, query_vector: List[float], limit: int, score_threshold: float):
        return self.search_results

# Test fixtures
@pytest.fixture
def mock_qdrant_client():
    return MockQdrantClient()

@pytest.fixture
def mock_embeddings():
    embeddings = Mock()
    embeddings.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]] * 5)
    embeddings.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    return embeddings

@pytest.fixture
def mock_chat_model():
    chat = AsyncMock()
    chat.agenerate.return_value = AsyncMock()
    return chat

@pytest.fixture
async def stream_client():
    config = StreamClientConfig(base_url="http://localhost:8000")
    return StreamClient(config)

# Test streaming functionality
@pytest.mark.asyncio
async def test_stream_manager():
    # Create stream
    stream_id = "test-stream"
    await stream_manager.create_stream(stream_id)
    
    # Push events
    events = [
        StreamEvent(StreamEventType.DATA, {"message": "Test data"}),
        StreamEvent(StreamEventType.PROGRESS, {"current": 5, "total": 10}),
        StreamEvent(StreamEventType.COMPLETE, {"message": "Done"})
    ]
    
    received_events = []
    
    async def collect_events():
        async for event in stream_manager.subscribe(stream_id, "test-client"):
            received_events.append(event)
            if event.type == StreamEventType.COMPLETE:
                break
    
    # Start collecting events in background
    collect_task = asyncio.create_task(collect_events())
    
    # Push events
    for event in events:
        await stream_manager.push_event(stream_id, event)
    
    # Wait for collection to complete
    await collect_task
    
    # Verify events
    assert len(received_events) == len(events)
    assert [e.type for e in received_events] == [e.type for e in events]
    assert [e.data for e in received_events] == [e.data for e in events]

# Test LangChain integration
@pytest.mark.asyncio
async def test_qdrant_vectorstore(mock_qdrant_client, mock_embeddings):
    config = LangChainConfig()
    store = QdrantVectorStore(
        client=mock_qdrant_client,
        collection_name="test",
        embeddings=mock_embeddings,
        config=config
    )
    
    # Test adding texts
    texts = ["Test document 1", "Test document 2"]
    stream_id = "test-stream"
    await stream_manager.create_stream(stream_id)
    
    ids = store.add_texts(texts, stream_id=stream_id)
    
    assert len(ids) == 2
    assert len(mock_qdrant_client.points) == 2
    
    # Test similarity search
    mock_qdrant_client.search_results = [
        Mock(payload={"text": "Test document 1"}, score=0.9),
        Mock(payload={"text": "Test document 2"}, score=0.8)
    ]
    
    results = store.similarity_search_with_score(
        "test query",
        k=2,
        stream_id=stream_id
    )
    
    assert len(results) == 2
    assert all(isinstance(doc, tuple) for doc in results)
    assert all(len(doc) == 2 for doc in results)

# Test StreamingQA
@pytest.mark.asyncio
async def test_streaming_qa(mock_qdrant_client, mock_embeddings, mock_chat_model):
    qa = StreamingQA(
        qdrant_client=mock_qdrant_client,
        collection_name="test",
        openai_api_key="test-key"
    )
    qa.embeddings = mock_embeddings
    qa.chat = mock_chat_model
    
    # Test document processing
    documents = ["Test document 1", "Test document 2"]
    stream_id = await qa.process_documents(documents)
    
    assert stream_id
    assert len(mock_qdrant_client.points) == 2
    
    # Test question answering
    question = "What is the test about?"
    stream_id = await qa.answer_question(question)
    
    assert stream_id
    assert mock_chat_model.agenerate.called

# Test client functionality
@pytest.mark.asyncio
async def test_stream_client(stream_client):
    # Create stream
    stream_id = await stream_client.create_stream()
    assert stream_id
    
    # Test callbacks
    events_received = []
    
    async def test_callback(data: dict):
        events_received.append(data)
    
    # Push some events through stream manager
    await stream_manager.create_stream(stream_id)
    await stream_manager.push_event(
        stream_id,
        StreamEvent(StreamEventType.DATA, {"message": "Test"})
    )
    await stream_manager.push_event(
        stream_id,
        StreamEvent(StreamEventType.COMPLETE, {"message": "Done"})
    )
    
    # Test SSE subscription
    async for event in stream_client.subscribe_sse(stream_id, test_callback):
        if event["type"] == "complete":
            break
    
    assert len(events_received) == 2
    assert events_received[0]["data"]["message"] == "Test"
    assert events_received[1]["type"] == "complete"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
