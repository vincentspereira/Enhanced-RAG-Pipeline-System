import logging
from typing import List, Dict, Any, Optional, Union
import uuid

from qdrant_client import QdrantClient, models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from .vector_store_base import VectorStoreBase, DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

class QdrantVectorStore(VectorStoreBase):
    def __init__(self, host: str = "localhost", port: int = 6333, api_key: Optional[str] = None,
                 grpc_port: int = 6334, prefer_grpc: bool = False, timeout: Optional[float]=None, **kwargs):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.timeout = timeout
        self.client: Optional[QdrantClient] = None
        self._initialize_client()

    def _initialize_client(self):
        try:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                api_key=self.api_key,
                timeout=self.timeout,
                **({"https_managed": False} if self.host == "localhost" and not self.api_key else {}) # Common for local dev
            )
            logger.info(f"Qdrant client initialized for host {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}", exc_info=True)
            self.client = None
            # raise # Or handle gracefully depending on application startup requirements

    async def initialize(self, collection_name: str, vector_size: int, distance_metric: str = "Cosine", **kwargs):
        if not self.client:
            logger.error("Qdrant client not initialized. Cannot create collection.")
            return

        distance_map = {
            "Cosine": qdrant_models.Distance.COSINE,
            "Euclidean": qdrant_models.Distance.EUCLID,
            "Dot": qdrant_models.Distance.DOT
        }
        qdrant_distance = distance_map.get(distance_metric.capitalize(), qdrant_models.Distance.COSINE)

        try:
            # Check if collection already exists
            try:
                self.client.get_collection(collection_name=collection_name)
                logger.info(f"Collection '{collection_name}' already exists in Qdrant.")
                # Optionally, verify vector_size and distance if needed, though recreate_collection handles it
                return
            except (UnexpectedResponse, ValueError) as e: # ValueError for 404 in some client versions
                 if isinstance(e, UnexpectedResponse) and e.status_code == 404:
                    logger.info(f"Collection '{collection_name}' does not exist. Creating now.")
                 elif isinstance(e, ValueError) and "Not found" in str(e): # Qdrant client might raise ValueError for 404
                    logger.info(f"Collection '{collection_name}' does not exist (ValueError). Creating now.")
                 else: # Some other unexpected error
                    raise e


            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_distance),
                # Add other configurations like hnsw_config, wal_config, optimizers_config if needed
                # e.g., optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
                # hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
            )
            logger.info(f"Successfully created or ensured Qdrant collection '{collection_name}' with vector size {vector_size} and distance {qdrant_distance}.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection '{collection_name}': {e}", exc_info=True)
            # raise # Or handle gracefully

    async def add_documents(self, collection_name: str, documents: List[DocumentChunk], **kwargs) -> List[Union[str, int, uuid.UUID]]:
        if not self.client:
            logger.error("Qdrant client not initialized. Cannot add documents.")
            return []

        points_to_upsert = []
        doc_ids = []
        for doc in documents:
            doc_id = str(doc.id) # Ensure ID is string for Qdrant
            points_to_upsert.append(qdrant_models.PointStruct(
                id=doc_id,
                vector=doc.vector,
                payload={ # Ensure payload is JSON serializable, Qdrant client handles dict directly
                    "text": doc.text,
                    "metadata": doc.metadata
                }
            ))
            doc_ids.append(doc.id)

        if not points_to_upsert:
            logger.info("No documents provided to add.")
            return []

        try:
            # Consider using batching for very large lists of documents, though client might handle some internal batching.
            # For very large scale, use client.upsert_points_batch for async/streaming.
            self.client.upsert(
                collection_name=collection_name,
                points=points_to_upsert,
                wait=kwargs.get("wait", True) # Default to wait=True for simpler flow
            )
            logger.info(f"Successfully added/updated {len(points_to_upsert)} documents in Qdrant collection '{collection_name}'.")
            return doc_ids
        except Exception as e:
            logger.error(f"Failed to add documents to Qdrant collection '{collection_name}': {e}", exc_info=True)
            return [] # Or raise

    async def search(self, collection_name: str, query_vector: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None, **kwargs) -> List[SearchResult]:
        if not self.client:
            logger.error("Qdrant client not initialized. Cannot perform search.")
            return []

        qdrant_filter = None
        if filters:
            # Convert generic filters to Qdrant specific filters if necessary
            # Example: assuming filters like {"metadata.field": "value"}
            must_conditions = []
            for key, value in filters.items():
                 # This is a simplified filter conversion. Qdrant supports complex conditions.
                 # Example: key = "metadata.source", value = "news"
                field_path = key.split('.') # e.g. ["metadata", "source"]
                if len(field_path) > 1 and field_path[0] == "metadata": # Assuming filters are on metadata
                    actual_key = ".".join(field_path[1:]) # "source"
                    must_conditions.append(qdrant_models.FieldCondition(
                        key=f"metadata.{actual_key}", # Correctly reference nested metadata
                        match=qdrant_models.MatchValue(value=value)
                    ))
                else: # Top-level payload field filter
                     must_conditions.append(qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value)
                    ))

            if must_conditions:
                qdrant_filter = qdrant_models.Filter(must=must_conditions)

        try:
            search_params = kwargs.get("search_params") # e.g., models.SearchParams(hnsw_ef=128, exact=False)

            hits = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                search_params=search_params,
                with_payload=True, # Ensure payload is returned
                with_vectors=kwargs.get("with_vectors", False) # Optionally return vectors
            )

            results = []
            for hit in hits:
                results.append(SearchResult(
                    id=hit.id,
                    score=hit.score,
                    payload=hit.payload, # Payload is already a dict
                    vector=hit.vector if hit.vector else None
                ))
            return results
        except Exception as e:
            logger.error(f"Failed to search Qdrant collection '{collection_name}': {e}", exc_info=True)
            return []

    async def delete_documents(self, collection_name: str, document_ids: List[Union[str, int, uuid.UUID]], **kwargs) -> bool:
        if not self.client:
            logger.error("Qdrant client not initialized. Cannot delete documents.")
            return False

        # Ensure all IDs are in a format Qdrant expects (e.g., string or int, depending on how they were inserted)
        # Qdrant typically uses UUIDs as strings, or integers if specified.
        # For this implementation, DocumentChunk uses UUID which is converted to string for PointStruct ID.
        qdrant_ids = [str(doc_id) for doc_id in document_ids]

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.PointIdsList(points=qdrant_ids),
                wait=kwargs.get("wait", True)
            )
            logger.info(f"Successfully deleted {len(qdrant_ids)} documents from Qdrant collection '{collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from Qdrant collection '{collection_name}': {e}", exc_info=True)
            return False

    async def get_collection_info(self, collection_name: str, **kwargs) -> Dict[str, Any]:
        if not self.client:
            logger.error("Qdrant client not initialized. Cannot get collection info.")
            return {}
        try:
            collection_info = self.client.get_collection(collection_name=collection_name)
            # Convert Qdrant models to dict for consistent return type
            return {
                "name": collection_name,
                "status": str(collection_info.status), # Enum to string
                "optimizer_status": str(collection_info.optimizer_status), # Enum to string
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count if collection_info.indexed_vectors_count is not None else 0,
                "points_count": collection_info.points_count if collection_info.points_count is not None else 0,
                "segments_count": collection_info.segments_count,
                "config": {
                    "params": collection_info.config.params.dict() if collection_info.config.params else None,
                    "hnsw_config": collection_info.config.hnsw_config.dict() if collection_info.config.hnsw_config else None,
                    "optimizer_config": collection_info.config.optimizer_config.dict() if collection_info.config.optimizer_config else None,
                    "wal_config": collection_info.config.wal_config.dict() if collection_info.config.wal_config else None,
                },
                "payload_schema": {k: str(v.data_type) for k, v in collection_info.payload_schema.items()} if collection_info.payload_schema else {}
            }
        except Exception as e:
            logger.error(f"Failed to get info for Qdrant collection '{collection_name}': {e}", exc_info=True)
            return {}

    async def close(self):
        if self.client:
            try:
                self.client.close()
                logger.info("Qdrant client connection closed.")
            except Exception as e:
                logger.error(f"Error closing Qdrant client: {e}", exc_info=True)
            finally:
                self.client = None

    async def health_check(self) -> bool:
        if not self.client:
            return False
        try:
            # Qdrant client doesn't have a direct 'ping'.
            # We can try a lightweight operation like listing collections or root path.
            # For instance, if Qdrant HTTP is on self.host:self.port
            # A simple GET to "/" or "/cluster" could work.
            # Using client.get_collections() is a reasonable check.
            self.client.get_collections() # This will raise an error if not connected
            return True
        except Exception:
            return False

# Example of how it might be used (for testing this file directly)
if __name__ == '__main__':
    async def main():
        logging.basicConfig(level=logging.INFO)
        # Assume Qdrant is running on localhost:6333
        qdrant_store = QdrantVectorStore(host="localhost", port=6333)

        if not await qdrant_store.health_check():
            logger.error("Qdrant not healthy. Exiting example.")
            return

        collection_name = "test_collection_qvs"
        vector_size = 3 # Example size

        await qdrant_store.initialize(collection_name, vector_size, distance_metric="Cosine")

        info = await qdrant_store.get_collection_info(collection_name)
        logger.info(f"Collection info for '{collection_name}': {info}")

        docs_to_add = [
            DocumentChunk(id=str(uuid.uuid4()), text="This is document 1 about apples.", vector=[0.1, 0.2, 0.3], metadata={"source": "A"}),
            DocumentChunk(id=str(uuid.uuid4()), text="Document 2 is about bananas.", vector=[0.4, 0.5, 0.6], metadata={"source": "B"}),
            DocumentChunk(id=123, text="Document 3 is about oranges and apples.", vector=[0.7, 0.8, 0.9], metadata={"source": "A", "year": 2023}),
        ]
        added_ids = await qdrant_store.add_documents(collection_name, docs_to_add)
        logger.info(f"Added document IDs: {added_ids}")

        info_after_add = await qdrant_store.get_collection_info(collection_name)
        logger.info(f"Collection info after add: {info_after_add}")

        if added_ids:
            query_vec = [0.11, 0.21, 0.31] # Similar to doc1
            search_results = await qdrant_store.search(collection_name, query_vec, top_k=2)
            logger.info(f"Search results for {query_vec}:")
            for res in search_results:
                logger.info(f"  ID: {res.id}, Score: {res.score:.4f}, Payload: {res.payload}")

            # Test filtering
            filtered_search = await qdrant_store.search(collection_name, query_vec, top_k=2, filters={"metadata.source": "A"})
            logger.info(f"Search results for {query_vec} (filtered by source A):")
            for res in filtered_search:
                 logger.info(f"  ID: {res.id}, Score: {res.score:.4f}, Payload: {res.payload}")


            # Test deletion
            # await qdrant_store.delete_documents(collection_name, [added_ids[0]])
            # info_after_delete = await qdrant_store.get_collection_info(collection_name)
            # logger.info(f"Collection info after delete: {info_after_delete}")

        await qdrant_store.close()

    import asyncio
    asyncio.run(main())
