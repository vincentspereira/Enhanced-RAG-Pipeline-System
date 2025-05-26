from typing import List, Dict, Any, Optional
import numpy as np
import pinecone
from uuid import uuid4

from Scripts.vector_stores.base import VectorStore

class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, index_name: str, 
                 api_key: str,
                 environment: str,
                 dimension: int = 1536,
                 metric: str = "cosine",
                 namespace: Optional[str] = None):
        """Initialize Pinecone vector store.
        
        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key
            environment: Pinecone environment
            dimension: Dimension of vectors
            metric: Distance metric (cosine, dotproduct, or euclidean)
            namespace: Optional namespace for multi-tenancy
        """
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize Pinecone client
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric
            )
            
        # Connect to index
        self.index = pinecone.Index(index_name)
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], 
                   ids: Optional[List[str]] = None):
        """Add vectors with associated metadata to the store."""
        # Prepare vectors for upsert
        items = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            # Generate ID if not provided
            vector_id = ids[i] if ids and i < len(ids) else str(uuid4())
            
            items.append((vector_id, vector.tolist(), meta))
        
        # Batch upsert to Pinecone
        self.index.upsert(
            vectors=items,
            namespace=self.namespace
        )
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        results = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace
        )
        
        # Format results
        return [
            {
                "id": match.id, 
                "score": match.score,
                "metadata": match.metadata
            } 
            for match in results.matches
        ]
    
    def delete_vectors(self, ids: List[str]):
        """Delete vectors by their IDs."""
        self.index.delete(
            ids=ids,
            namespace=self.namespace
        )
        
    def __del__(self):
        """Clean up resources."""
        # Pinecone client doesn't require explicit closure
