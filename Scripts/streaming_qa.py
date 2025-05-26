import asyncio
from typing import List, Optional
import logging
from uuid import uuid4

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate

from .streaming.response_stream import stream_manager, StreamEvent, StreamEventType
from .integrations.langchain_integration import (
    QdrantVectorStore,
    StreamingCallbackHandler,
    StreamingRetriever,
    LangChainConfig
)

logger = logging.getLogger(__name__)

class StreamingQA:
    """Example class demonstrating streaming Q&A using LangChain and Qdrant"""
    
    def __init__(
        self,
        qdrant_client,
        collection_name: str,
        openai_api_key: str,
        config: Optional[LangChainConfig] = None
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.config = config or LangChainConfig()
        
        # Initialize LangChain components
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
            config=self.config
        )
        self.retriever = StreamingRetriever(vectorstore=self.vectorstore)
        
        # Initialize chat model
        self.chat = ChatOpenAI(
            openai_api_key=openai_api_key,
            streaming=True,
            temperature=0.7
        )
    
    async def process_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> str:
        """Process and index documents with streaming progress"""
        stream_id = str(uuid4())
        await stream_manager.create_stream(stream_id)
        
        try:
            # Add documents to vector store with streaming progress
            self.retriever.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                stream_id=stream_id
            )
            
            return stream_id
        except Exception as e:
            await stream_manager.push_error(stream_id, str(e))
            raise
    
    async def answer_question(self, question: str) -> str:
        """Answer a question using streaming responses"""
        stream_id = str(uuid4())
        await stream_manager.create_stream(stream_id)
        
        try:
            # Set up streaming callback
            callback_handler = StreamingCallbackHandler(stream_id)
            
            # Search for relevant documents
            relevant_docs = self.retriever.get_relevant_documents(
                question,
                stream_id=stream_id
            )
            
            # Create chat prompt
            context = "\n".join(doc.page_content for doc in relevant_docs)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}"),
                ("human", "{question}")
            ])
            
            messages = prompt.format_messages(
                context=context,
                question=question
            )
            
            # Get streaming response
            response = await self.chat.agenerate(
                messages=[messages],
                callbacks=[callback_handler]
            )
            
            return stream_id
        except Exception as e:
            await stream_manager.push_error(stream_id, str(e))
            raise

async def example_usage():
    """Example usage of streaming Q&A"""
    # Initialize QdrantClient (example)
    from qdrant_client import QdrantClient
    client = QdrantClient("localhost", port=6333)
    
    # Initialize StreamingQA
    qa = StreamingQA(
        qdrant_client=client,
        collection_name="documents",
        openai_api_key="your-openai-key"
    )
    
    # Example documents
    documents = [
        "Qdrant is a vector similarity search engine.",
        "It provides a production-ready service with HTTP API for object storage and search."
    ]
    
    # Process documents
    process_stream_id = await qa.process_documents(documents)
    print(f"Document processing stream ID: {process_stream_id}")
    
    # Ask a question
    question = "What is Qdrant?"
    answer_stream_id = await qa.answer_question(question)
    print(f"Answer stream ID: {answer_stream_id}")

if __name__ == "__main__":
    asyncio.run(example_usage())
