from typing import AsyncIterator, Optional, Callable, Any
import asyncio
import json
import aiohttp
import websockets
import logging
from urllib.parse import urljoin
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StreamClientConfig:
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1

class StreamClient:
    """Client for consuming streaming responses via SSE or WebSocket"""
    
    def __init__(self, config: Optional[StreamClientConfig] = None):
        self.config = config or StreamClientConfig()
    
    async def create_stream(self) -> str:
        """Create a new stream and return its ID"""
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.config.base_url, "/stream/create")
            async with session.post(url) as response:
                response.raise_for_status()
                data = await response.json()
                return data["stream_id"]

    async def subscribe_sse(
        self,
        stream_id: str,
        callback: Optional[Callable[[dict], Any]] = None
    ) -> AsyncIterator[dict]:
        """Subscribe to a stream using Server-Sent Events"""
        url = urljoin(self.config.base_url, f"/stream/{stream_id}/sse")
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config.retry_attempts):
                try:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                data = json.loads(line[6:])
                                if callback:
                                    await callback(data)
                                yield data
                                
                                if data["type"] == "complete":
                                    break
                    break  # Success, exit retry loop
                    
                except aiohttp.ClientError as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise
                    logger.warning(f"SSE connection failed, retrying: {e}")
                    await asyncio.sleep(self.config.retry_delay)

    async def subscribe_websocket(
        self,
        stream_id: str,
        callback: Optional[Callable[[dict], Any]] = None
    ) -> AsyncIterator[dict]:
        """Subscribe to a stream using WebSocket"""
        ws_url = urljoin(
            self.config.base_url.replace('http', 'ws'),
            f"/stream/{stream_id}/ws"
        )
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with websockets.connect(ws_url) as websocket:
                    while True:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            if callback:
                                await callback(data)
                            yield data
                            
                            if data["type"] == "complete":
                                break
                                
                        except websockets.ConnectionClosed:
                            break
                break  # Success, exit retry loop
                
            except (websockets.WebSocketException, ConnectionError) as e:
                if attempt == self.config.retry_attempts - 1:
                    raise
                logger.warning(f"WebSocket connection failed, retrying: {e}")
                await asyncio.sleep(self.config.retry_delay)

    @staticmethod
    def create_progress_callback(
        description: str = "Progress",
        show_percentage: bool = True
    ) -> Callable[[dict], Any]:
        """Create a callback that prints progress updates"""
        
        async def callback(data: dict):
            if data["type"] == "progress":
                progress = data["data"]
                current = progress["current"]
                total = progress["total"]
                percentage = progress["percentage"]
                message = progress["message"] or description
                
                if show_percentage:
                    print(f"{message}: {percentage}% ({current}/{total})")
                else:
                    print(f"{message}: {current}/{total}")
            
            elif data["type"] == "error":
                print(f"Error: {data['data']['error']}")
        
        return callback

    @staticmethod
    def create_token_callback(
        end_tokens: Optional[list] = None
    ) -> Callable[[dict], Any]:
        """Create a callback that prints streaming tokens"""
        
        async def callback(data: dict):
            if data["type"] == "token":
                token = data["data"]["content"]
                if not end_tokens or token not in end_tokens:
                    print(token, end="", flush=True)
            elif data["type"] == "error":
                print(f"\nError: {data['data']['error']}")
        
        return callback

# Example usage
async def example_client_usage():
    # Initialize client
    client = StreamClient(StreamClientConfig(base_url="http://localhost:8000"))
    
    # Create a new stream
    stream_id = await client.create_stream()
    print(f"Created stream: {stream_id}")
    
    # Example progress callback
    progress_callback = client.create_progress_callback("Processing documents")
    
    # Subscribe using SSE
    print("\nListening for updates (SSE):")
    async for event in client.subscribe_sse(stream_id, progress_callback):
        if event["type"] == "complete":
            break
    
    # Example token callback
    token_callback = client.create_token_callback(end_tokens=["\n"])
    
    # Subscribe using WebSocket
    print("\nListening for updates (WebSocket):")
    async for event in client.subscribe_websocket(stream_id, token_callback):
        if event["type"] == "complete":
            break

if __name__ == "__main__":
    asyncio.run(example_client_usage())
