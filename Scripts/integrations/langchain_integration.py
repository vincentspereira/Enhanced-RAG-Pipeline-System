from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator
from langchain.schema import Document
from langchain.vectorstores import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import json

from ..streaming.response_stream import stream_manager, StreamEvent, StreamEventType

logger = logging.getLogger(__name__)

@dataclass
class LangChainConfig:
    openai_api_key: str
    model_name: str = "gpt-3.5-turbo"
    embeddings_model: str = "text-embedding-3-small"
    temperature: float = 0.7
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_dimension: int = 1536
    top_k: int = 4
    similarity_threshold: float = 0.7
    enable_streaming: bool = True
    max_tokens: int = 2000
    enable_memory: bool = True
    memory_k: int = 5
    embedding_batch_size: int = 100
    retrieval_k: int = 4

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LangChain responses"""
    
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        await stream_manager.push_event(
            self.stream_id,
            StreamEvent(StreamEventType.DATA, {
                "type": "llm_start",
                "prompts": prompts
            })
        )
    
    async def on_llm_new_token(self, token: str, **kwargs):
        await stream_manager.push_event(
            self.stream_id,
            StreamEvent(StreamEventType.DATA, {
                "type": "token",
                "content": token
            })
        )
    
    async def on_llm_end(self, response: LLMResult, **kwargs):
        await stream_manager.push_event(
            self.stream_id,
            StreamEvent(StreamEventType.DATA, {
                "type": "llm_end",
                "response": response.dict()
            })
        )
    
    async def on_llm_error(self, error: Exception, **kwargs):
        await stream_manager.push_error(self.stream_id, str(error))

class QdrantVectorStore(VectorStore):
    """LangChain vector store implementation for Qdrant"""
    
    def __init__(
        self,
        client: Any,  # QdrantClient
        collection_name: str,
        embeddings: Embeddings,
        config: LangChainConfig = LangChainConfig()
    ):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.config = config

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        stream_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store with optional streaming progress"""
        ids = []
        total_texts = len(texts)
        
        async def process_batch():
            for i in range(0, total_texts, self.config.embedding_batch_size):
                batch = texts[i:i + self.config.embedding_batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                batch_metadata = metadatas[i:i + self.config.embedding_batch_size] if metadatas else None
                
                # Generate IDs for the batch
                batch_ids = [str(uuid4()) for _ in batch]
                ids.extend(batch_ids)
                
                # Upload to Qdrant
                points = []
                for j, (text, embedding) in enumerate(zip(batch, batch_embeddings)):
                    point = {
                        "id": batch_ids[j],
                        "vector": embedding,
                        "payload": {
                            "text": text,
                            **(batch_metadata[j] if batch_metadata else {})
                        }
                    }
                    points.append(point)
                
                await self.client.upload_points(
                    collection_name=self.collection_name,
                    points=points
                )
                
                if stream_id:
                    await stream_manager.push_progress(
                        stream_id,
                        min(i + len(batch), total_texts),
                        total_texts,
                        "Processing documents"
                    )
        
        # Create event loop and run processing
        loop = asyncio.get_event_loop()
        loop.run_until_complete(process_batch())
        
        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        stream_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return similar documents and relevance scores with optional streaming"""
        async def search():
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            if stream_id:
                await stream_manager.push_event(
                    stream_id,
                    StreamEvent(StreamEventType.DATA, {
                        "type": "search_start",
                        "query": query
                    })
                )
            
            # Search in Qdrant
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=self.config.similarity_threshold
            )
            
            documents_with_scores = []
            for res in results:
                doc = Document(
                    page_content=res.payload.get("text", ""),
                    metadata=res.payload
                )
                documents_with_scores.append((doc, res.score))
            
            if stream_id:
                await stream_manager.push_event(
                    stream_id,
                    StreamEvent(StreamEventType.DATA, {
                        "type": "search_complete",
                        "results_count": len(documents_with_scores)
                    })
                )
            
            return documents_with_scores
        
        # Create event loop and run search
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(search())

class LangChainIntegration:
    def __init__(self, config: LangChainConfig):
        self.config = config
        self._setup_components()
        
    def _setup_components(self):
        """Initialize LangChain components"""
        from langchain.embeddings import OpenAIEmbeddings
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embeddings_model,
            openai_api_key=self.config.openai_api_key
        )
        
        # Initialize chat model
        self.chat_model = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            openai_api_key=self.config.openai_api_key,
            streaming=self.config.enable_streaming,
            max_tokens=self.config.max_tokens
        )
        
        # Initialize conversation memory
        if self.config.enable_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                k=self.config.memory_k
            )
        else:
            self.memory = None

    def initialize_vectorstore(
        self,
        client: QdrantClient,
        collection_name: str
    ) -> QdrantVectorStore:
        """Initialize Qdrant vector store"""
        # Ensure collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_dimension,
                    distance=Distance.COSINE
                )
            )
            
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embeddings=self.embeddings
        )

    def create_retrieval_chain(
        self,
        vectorstore: VectorStore
    ) -> ConversationalRetrievalChain:
        """Create a conversational retrieval chain"""
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": self.config.top_k,
                "score_threshold": self.config.similarity_threshold
            }
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.chat_model,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True
        )

    async def process_query(
        self,
        chain: ConversationalRetrievalChain,
        query: str,
        chat_history: Optional[List[tuple]] = None
    ) -> AsyncIterator[str]:
        """Process a query and stream the response"""
        handler = StreamingCallbackHandler()
        
        # Run the chain in a background task
        task = asyncio.create_task(
            chain.arun(
                question=query,
                chat_history=chat_history or [],
                callbacks=[handler]
            )
        )
        
        # Stream tokens as they become available
        async for token in handler.get_tokens():
            yield token
            
        # Ensure the task completes
        await task

    async def batch_process_documents(
        self,
        vectorstore: VectorStore,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """Process and store documents in batches"""
        ids = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            batch_ids = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: vectorstore.add_texts(texts, metadatas)
            )
            ids.extend(batch_ids)
        return ids

    def get_relevant_context(
        self,
        vectorstore: VectorStore,
        query: str,
        max_tokens: Optional[int] = None
    ) -> List[Document]:
        """Get relevant context for a query"""
        docs = vectorstore.similarity_search(
            query,
            k=self.config.top_k
        )
        
        if max_tokens:
            # Trim context to fit within token limit
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
            trimmed_docs = []
            total_tokens = 0
            
            for doc in docs:
                tokens = tokenizer.encode(doc.page_content)
                if total_tokens + len(tokens) <= max_tokens:
                    trimmed_docs.append(doc)
                    total_tokens += len(tokens)
                else:
                    # Trim the last document to fit
                    tokens = tokens[:max_tokens - total_tokens]
                    doc.page_content = tokenizer.decode(tokens)
                    trimmed_docs.append(doc)
                    break
                    
            return trimmed_docs
        
        return docs

    async def update_memory(
        self,
        query: str,
        response: str
    ):
        """Update conversation memory"""
        if self.memory:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.memory.save_context(
                    {"input": query},
                    {"output": response}
                )
            )

    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
