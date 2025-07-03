from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import uuid

# Define a common SearchResult structure if not already globally available
# This can be refined or imported from a common models location
class SearchResult(BaseModel):
    id: Union[str, int, uuid.UUID]
    score: float
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None # Optional, not all stores return it by default

class DocumentChunk(BaseModel):
    id: Union[str, int, uuid.UUID] = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorStoreBase(ABC):
    """
    Abstract base class for vector store implementations.
    """

    @abstractmethod
    async def initialize(self, collection_name: str, vector_size: int, distance_metric: str = "Cosine", **kwargs):
        """
        Initialize the vector store, including creating a collection if it doesn't exist.
        Args:
            collection_name (str): Name of the collection/index.
            vector_size (int): Dimension of the vectors.
            distance_metric (str): Distance metric (e.g., "Cosine", "Euclidean", "Dot").
            **kwargs: Additional provider-specific parameters.
        """
        pass

    @abstractmethod
    async def add_documents(self, collection_name: str, documents: List[DocumentChunk], **kwargs) -> List[Union[str, int, uuid.UUID]]:
        """
        Adds or updates documents (points/vectors) in the specified collection.
        Args:
            collection_name (str): Name of the collection/index.
            documents (List[DocumentChunk]): A list of DocumentChunk objects to add.
            **kwargs: Additional provider-specific parameters.
        Returns:
            List of IDs of the added/updated documents.
        """
        pass

    @abstractmethod
    async def search(self, collection_name: str, query_vector: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None, **kwargs) -> List[SearchResult]:
        """
        Performs a similarity search in the specified collection.
        Args:
            collection_name (str): Name of the collection/index.
            query_vector (List[float]): The vector to search for.
            top_k (int): The number of nearest neighbors to return.
            filters (Optional[Dict[str, Any]]): Filters to apply during search (provider-specific).
            **kwargs: Additional provider-specific parameters.
        Returns:
            List of SearchResult objects.
        """
        pass

    @abstractmethod
    async def delete_documents(self, collection_name: str, document_ids: List[Union[str, int, uuid.UUID]], **kwargs) -> bool:
        """
        Deletes documents from the collection by their IDs.
        Args:
            collection_name (str): Name of the collection.
            document_ids (List[Union[str, int, uuid.UUID]]): List of document IDs to delete.
            **kwargs: Additional provider-specific parameters.
        Returns:
            True if deletion was successful or partially successful, False otherwise.
        """
        pass

    @abstractmethod
    async def get_collection_info(self, collection_name: str, **kwargs) -> Dict[str, Any]:
        """
        Retrieves information about a specific collection.
        Args:
            collection_name (str): Name of the collection.
            **kwargs: Additional provider-specific parameters.
        Returns:
            A dictionary containing collection information (e.g., point count, status).
        """
        pass

    @abstractmethod
    async def close(self):
        """
        Closes any open connections to the vector store.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Performs a health check on the vector store.
        Returns:
            True if healthy, False otherwise.
        """
        pass
