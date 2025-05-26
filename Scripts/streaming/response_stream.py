from typing import AsyncIterator, Any, Dict, Optional, List
import asyncio
from dataclasses import dataclass
import json
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StreamEventType(Enum):
    DATA = "data"
    PROGRESS = "progress"
    ERROR = "error"
    COMPLETE = "complete"

@dataclass
class StreamEvent:
    type: StreamEventType
    data: Any
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_sse(self) -> str:
        """Convert event to Server-Sent Events format"""
        event_data = {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp
        }
        return f"event: message\ndata: {json.dumps(event_data)}\n\n"

    def to_websocket(self) -> str:
        """Convert event to WebSocket message format"""
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp
        })

class StreamManager:
    def __init__(self):
        self._streams: Dict[str, asyncio.Queue] = {}
        self._active_connections: Dict[str, set] = {}

    async def create_stream(self, stream_id: str) -> str:
        """Create a new stream"""
        if stream_id in self._streams:
            raise ValueError(f"Stream {stream_id} already exists")
        
        self._streams[stream_id] = asyncio.Queue()
        self._active_connections[stream_id] = set()
        return stream_id

    async def delete_stream(self, stream_id: str):
        """Delete a stream and notify all connected clients"""
        if stream_id in self._streams:
            # Send completion event to all clients
            await self.push_event(
                stream_id,
                StreamEvent(StreamEventType.COMPLETE, {"message": "Stream closed"})
            )
            
            # Clean up
            del self._streams[stream_id]
            del self._active_connections[stream_id]

    async def push_event(self, stream_id: str, event: StreamEvent):
        """Push an event to all clients connected to the stream"""
        if stream_id not in self._streams:
            raise ValueError(f"Stream {stream_id} does not exist")
        
        await self._streams[stream_id].put(event)

    async def subscribe(self, stream_id: str, client_id: str) -> AsyncIterator[StreamEvent]:
        """Subscribe to a stream and yield events"""
        if stream_id not in self._streams:
            raise ValueError(f"Stream {stream_id} does not exist")
        
        self._active_connections[stream_id].add(client_id)
        try:
            while True:
                event = await self._streams[stream_id].get()
                yield event
                if event.type == StreamEventType.COMPLETE:
                    break
        finally:
            self._active_connections[stream_id].remove(client_id)

    async def push_progress(self, stream_id: str, current: int, total: int, message: str = ""):
        """Helper method to push progress updates"""
        await self.push_event(
            stream_id,
            StreamEvent(
                StreamEventType.PROGRESS,
                {
                    "current": current,
                    "total": total,
                    "percentage": round((current / total) * 100, 2),
                    "message": message
                }
            )
        )

    async def push_error(self, stream_id: str, error: str):
        """Helper method to push error events"""
        await self.push_event(
            stream_id,
            StreamEvent(StreamEventType.ERROR, {"error": error})
        )

    def get_active_connections(self, stream_id: str) -> set:
        """Get set of active client connections for a stream"""
        return self._active_connections.get(stream_id, set())

    async def broadcast(self, event: StreamEvent):
        """Broadcast an event to all active streams"""
        for stream_id in self._streams:
            await self.push_event(stream_id, event)

# Global stream manager instance
stream_manager = StreamManager()
