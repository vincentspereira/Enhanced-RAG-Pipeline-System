from typing import Optional
from fastapi import APIRouter, WebSocket, HTTPException
from sse_starlette.sse import EventSourceResponse
import logging
from uuid import uuid4

from .response_stream import stream_manager, StreamEvent, StreamEventType

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/stream/{stream_id}/sse")
async def stream_sse(stream_id: str):
    """Server-Sent Events endpoint for streaming responses"""
    client_id = str(uuid4())
    
    try:
        async def event_generator():
            async for event in stream_manager.subscribe(stream_id, client_id):
                yield event.to_sse()
        
        return EventSourceResponse(event_generator())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in SSE stream: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.websocket("/stream/{stream_id}/ws")
async def stream_websocket(websocket: WebSocket, stream_id: str):
    """WebSocket endpoint for streaming responses"""
    client_id = str(uuid4())
    
    try:
        await websocket.accept()
        async for event in stream_manager.subscribe(stream_id, client_id):
            await websocket.send_text(event.to_websocket())
    except ValueError as e:
        await websocket.close(code=4000, reason=str(e))
    except Exception as e:
        logger.error(f"Error in WebSocket stream: {e}")
        await websocket.close(code=1011, reason="Internal server error")
    finally:
        if not websocket.client_state.DISCONNECTED:
            await websocket.close()

@router.post("/stream/create")
async def create_stream(stream_id: Optional[str] = None):
    """Create a new stream"""
    stream_id = stream_id or str(uuid4())
    try:
        await stream_manager.create_stream(stream_id)
        return {"stream_id": stream_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/stream/{stream_id}")
async def delete_stream(stream_id: str):
    """Delete a stream"""
    try:
        await stream_manager.delete_stream(stream_id)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/stream/{stream_id}/status")
async def get_stream_status(stream_id: str):
    """Get stream status and active connections"""
    try:
        active_connections = stream_manager.get_active_connections(stream_id)
        return {
            "stream_id": stream_id,
            "active_connections": len(active_connections),
            "client_ids": list(active_connections)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
