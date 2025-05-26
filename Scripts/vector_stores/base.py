from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """Add vectors with associated metadata to the store."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]):
        """Delete vectors by their IDs."""
        pass

class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(self, collection_name: str, vector_size: int, 
                 host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], 
                   ids: Optional[List[str]] = None):
        points = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            point_id = ids[i] if ids else str(i)
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=meta
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        
        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "metadata": hit.payload
            }
            for hit in results
        ]
    
    def delete_vectors(self, ids: List[str]):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )

class VectorStoreRegistry:
    """Registry for managing different vector store implementations."""
    
    def __init__(self):
        self._stores: Dict[str, VectorStore] = {}
        
    def register_store(self, name: str, store: VectorStore):
        """Register a new vector store."""
        self._stores[name] = store
        
    def get_store(self, name: str) -> VectorStore:
        """Get a registered store by name."""
        if name not in self._stores:
            raise KeyError(f"Store {name} not found in registry")
        return self._stores[name]
        
    def list_stores(self) -> List[str]:
        """List all registered store names."""
        return list(self._stores.keys())

# Create global registry instance
store_registry = VectorStoreRegistry()

# Register default QdrantVectorStore
def register_qdrant(collection_name: str, vector_size: int, 
                   host: str = "localhost", port: int = 6333):
    store = QdrantVectorStore(
        collection_name=collection_name,
        vector_size=vector_size,
        host=host,
        port=port
    )
    store_registry.register_store("qdrant", store)
    return store

# Functions to initialize and register other vector stores will be added by users as needed