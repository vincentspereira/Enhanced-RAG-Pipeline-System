from typing import List, Dict, Any, Optional
import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

from Scripts.vector_stores.base import VectorStore

class MilvusVectorStore(VectorStore):
    """Milvus/Zilliz vector store implementation."""
    
    def __init__(self, collection_name: str, 
                 host: str = "localhost", 
                 port: int = 19530,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 vector_dim: int = 1536):
        """Initialize Milvus vector store.
        
        Args:
            collection_name: Name of the Milvus collection
            host: Milvus server host
            port: Milvus server port
            user: Optional username for authentication
            password: Optional password for authentication
            vector_dim: Dimension of vectors
        """
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=host,
            port=port,
            user=user,
            password=password
        )
        
        # Create collection if it doesn't exist
        if not utility.has_collection(collection_name):
            self._create_collection()
        
        # Get collection
        self.collection = Collection(collection_name)
        self.collection.load()
    
    def _create_collection(self):
        """Create a new collection with the required schema."""
        # Define fields
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        # Create schema
        schema = CollectionSchema(fields=fields)
        
        # Create collection
        collection = Collection(name=self.collection_name, schema=schema)
        
        # Create index for vectors
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="vector", index_params=index_params)
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], 
                   ids: Optional[List[str]] = None):
        """Add vectors with associated metadata to the store."""
        # Prepare data for insertion
        vector_ids = []
        if ids:
            vector_ids = ids
        else:
            # Generate simple IDs
            vector_ids = [f"vector_{i}" for i in range(len(vectors))]
        
        # Prepare data
        data = [
            vector_ids,
            vectors.tolist(),
            metadata
        ]
        
        # Insert data
        self.collection.insert(data)
        self.collection.flush()
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        # Define search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 32}
        }
        
        # Execute search
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["metadata"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "metadata": hit.entity.get("metadata")
                })
        
        return formatted_results
    
    def delete_vectors(self, ids: List[str]):
        """Delete vectors by their IDs."""
        expr = f"id in {ids}"
        self.collection.delete(expr)
        
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'collection'):
            self.collection.release()
        connections.disconnect("default")
