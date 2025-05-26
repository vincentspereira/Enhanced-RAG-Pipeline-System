"""
Module for initializing and registering various vector stores.
"""
from typing import Dict, Any, Optional

from Scripts.vector_stores.base import store_registry, QdrantVectorStore
from Scripts.vector_stores.weaviate_store import WeaviateVectorStore
from Scripts.vector_stores.pinecone_store import PineconeVectorStore
from Scripts.vector_stores.milvus_store import MilvusVectorStore

def register_qdrant(collection_name: str, vector_size: int, 
                   host: str = "localhost", port: int = 6333):
    """Register a Qdrant vector store."""
    store = QdrantVectorStore(
        collection_name=collection_name,
        vector_size=vector_size,
        host=host,
        port=port
    )
    store_registry.register_store("qdrant", store)
    return store

def register_weaviate(class_name: str, 
                     host: str = "localhost", 
                     port: int = 8080,
                     api_key: Optional[str] = None):
    """Register a Weaviate vector store."""
    store = WeaviateVectorStore(
        class_name=class_name,
        host=host,
        port=port,
        api_key=api_key
    )
    store_registry.register_store("weaviate", store)
    return store

def register_pinecone(index_name: str, 
                     api_key: str,
                     environment: str,
                     dimension: int = 1536,
                     namespace: Optional[str] = None):
    """Register a Pinecone vector store."""
    store = PineconeVectorStore(
        index_name=index_name,
        api_key=api_key,
        environment=environment,
        dimension=dimension,
        namespace=namespace
    )
    store_registry.register_store("pinecone", store)
    return store

def register_milvus(collection_name: str, 
                   host: str = "localhost", 
                   port: int = 19530,
                   user: Optional[str] = None,
                   password: Optional[str] = None,
                   vector_dim: int = 1536):
    """Register a Milvus vector store."""
    store = MilvusVectorStore(
        collection_name=collection_name,
        host=host,
        port=port,
        user=user,
        password=password,
        vector_dim=vector_dim
    )
    store_registry.register_store("milvus", store)
    return store

def init_vector_stores(config: Dict[str, Any]):
    """Initialize all vector stores from configuration."""
    # Get vector store configuration
    vector_store_config = config.get("vector_store", {})
    
    # Determine provider
    provider = vector_store_config.get("provider", "qdrant")
    
    # Initialize based on provider
    if provider == "qdrant":
        qdrant_config = vector_store_config.get("qdrant", {})
        register_qdrant(
            collection_name=qdrant_config.get("collection_name", "documents"),
            vector_size=qdrant_config.get("vector_size", 1536),
            host=qdrant_config.get("host", "localhost"),
            port=qdrant_config.get("port", 6333)
        )
    elif provider == "weaviate":
        weaviate_config = vector_store_config.get("weaviate", {})
        register_weaviate(
            class_name=weaviate_config.get("class_name", "Document"),
            host=weaviate_config.get("host", "localhost"),
            port=weaviate_config.get("port", 8080),
            api_key=weaviate_config.get("api_key")
        )
    elif provider == "pinecone":
        pinecone_config = vector_store_config.get("pinecone", {})
        register_pinecone(
            index_name=pinecone_config.get("index_name", "documents"),
            api_key=pinecone_config.get("api_key", ""),
            environment=pinecone_config.get("environment", ""),
            dimension=pinecone_config.get("dimension", 1536),
            namespace=pinecone_config.get("namespace")
        )
    elif provider == "milvus":
        milvus_config = vector_store_config.get("milvus", {})
        register_milvus(
            collection_name=milvus_config.get("collection_name", "documents"),
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
            user=milvus_config.get("user"),
            password=milvus_config.get("password"),
            vector_dim=milvus_config.get("vector_dim", 1536)
        )
    else:
        raise ValueError(f"Unsupported vector store provider: {provider}")
        
    return store_registry
