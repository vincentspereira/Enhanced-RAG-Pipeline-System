from typing import List, Dict, Any, Optional
import numpy as np
import weaviate
from weaviate.auth import AuthApiKey

from Scripts.vector_stores.base import VectorStore

class WeaviateVectorStore(VectorStore):
    """Weaviate vector store implementation."""
    
    def __init__(self, class_name: str, 
                 host: str = "localhost", 
                 port: int = 8080,
                 api_key: Optional[str] = None,
                 batch_size: int = 100):
        """Initialize Weaviate vector store.
        
        Args:
            class_name: The name of the Weaviate class
            host: Weaviate host
            port: Weaviate port
            api_key: Optional API key for authentication
            batch_size: Size of batches for bulk operations
        """
        self.class_name = class_name
        self.batch_size = batch_size
        
        # Setup client connection
        if api_key:
            self.client = weaviate.Client(
                url=f"http://{host}:{port}",
                auth_client_secret=AuthApiKey(api_key=api_key)
            )
        else:
            self.client = weaviate.Client(
                url=f"http://{host}:{port}"
            )
            
        # Create schema if it doesn't exist
        self._create_schema_if_not_exists()
        
    def _create_schema_if_not_exists(self):
        """Create schema if it doesn't exist."""
        # Check if class exists
        if not self.client.schema.exists(self.class_name):
            # Define class schema
            class_definition = {
                "class": self.class_name,
                "vectorizer": "none",  # We'll provide vectors directly
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                    }
                ]
            }
            # Create class
            self.client.schema.create_class(class_definition)
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], 
                   ids: Optional[List[str]] = None):
        """Add vectors with associated metadata to the store."""
        # Create batch for efficient insertion
        with self.client.batch as batch:
            batch.batch_size = self.batch_size
            
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                # Get ID or create one
                object_id = ids[i] if ids and i < len(ids) else None
                
                # Extract content if available
                content = meta.get("content", "")
                
                # Add object to batch
                batch.add_data_object(
                    data_object={
                        "content": content,
                        "metadata": meta
                    },
                    class_name=self.class_name,
                    uuid=object_id,
                    vector=vector.tolist()
                )
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        results = (
            self.client.query
            .get(self.class_name, ["content", "metadata"])
            .with_near_vector({
                "vector": query_vector.tolist()
            })
            .with_limit(top_k)
            .do()
        )
        
        # Extract results
        if "data" in results and "Get" in results["data"]:
            objects = results["data"]["Get"][self.class_name]
            return [
                {
                    "id": obj["id"],
                    "score": obj.get("_additional", {}).get("distance", 0.0),
                    "metadata": obj["metadata"],
                    "content": obj["content"]
                }
                for obj in objects
            ]
        return []
    
    def delete_vectors(self, ids: List[str]):
        """Delete vectors by their IDs."""
        for id in ids:
            try:
                self.client.data_object.delete(
                    class_name=self.class_name,
                    uuid=id
                )
            except Exception as e:
                # Log error but continue with other deletions
                print(f"Error deleting object {id}: {str(e)}")
                
    def __del__(self):
        """Close client connection."""
        if hasattr(self, 'client'):
            self.client.close()
