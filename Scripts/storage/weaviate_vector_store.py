import logging
from typing import List, Dict, Any, Optional, Union
import uuid
import weaviate
from weaviate.exceptions import WeaviateQueryError, WeaviateStartUpError, WeaviateInsertError, WeaviateBatchError

from .vector_store_base import VectorStoreBase, DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

class WeaviateVectorStore(VectorStoreBase):
    def __init__(self, url: str, api_key: Optional[str] = None, **kwargs):
        self.url = url
        self.api_key = api_key
        self.client: Optional[weaviate.Client] = None
        self._initialize_client(**kwargs)

    def _initialize_client(self, **kwargs):
        try:
            auth_config = None
            if self.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.api_key)

            # Additional headers can be passed via kwargs if needed, e.g., for OIDC
            additional_headers = kwargs.get("additional_headers", {})

            self.client = weaviate.Client(
                url=self.url,
                auth_client_secret=auth_config,
                additional_headers=additional_headers,
                timeout_config=kwargs.get("timeout_config", (10, 60)) # (connect_timeout, read_timeout)
            )
            if not self.client.is_ready():
                raise WeaviateStartUpError("Weaviate client connected but instance is not ready.")
            logger.info(f"Weaviate client initialized for URL {self.url}")
        except WeaviateStartUpError as e:
            logger.error(f"Failed to initialize Weaviate client (instance not ready): {e}", exc_info=True)
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}", exc_info=True)
            self.client = None

    async def initialize(self, class_name: str, vector_size: int, distance_metric: str = "cosine", **kwargs):
        if not self.client:
            logger.error("Weaviate client not initialized. Cannot create class.")
            return

        # Weaviate class name must start with an uppercase letter
        if not class_name[0].isupper():
            logger.warning(f"Weaviate class name '{class_name}' does not start with an uppercase letter. This might cause issues.")
            # class_name = class_name.capitalize() # Optionally auto-capitalize

        vectorizer = kwargs.get("vectorizer", "none") # "none" if providing own vectors
        vector_index_type = kwargs.get("vector_index_type", "hnsw")

        # Map common distance names to Weaviate distance metrics for HNSW
        # For other index types, distance might be implicit or configured differently.
        distance_map_hnsw = {
            "cosine": "cosine",
            "dot": "dot",
            "euclidean": "l2-squared", # Weaviate's common Euclidean equivalent for HNSW
            "l2-squared": "l2-squared",
            "manhattan": "manhattan", # Not always available or optimal for HNSW with all vectorizers
            "hamming": "hamming"      # For binary vectors
        }
        weaviate_distance = distance_map_hnsw.get(distance_metric.lower(), "cosine")

        class_obj = {
            "class": class_name,
            "description": kwargs.get("description", f"Class for storing {class_name} objects"),
            "vectorizer": vectorizer,
            "vectorIndexType": vector_index_type,
            "vectorIndexConfig": kwargs.get("vector_index_config", { # Defaults for HNSW
                "distance": weaviate_distance,
                # Other HNSW params: efConstruction, maxConnections, ef, etc.
            }),
            "properties": [
                {"name": "text", "dataType": ["text"]},
                # Metadata properties need to be explicitly defined if they are to be filterable without _additional fields
                # Example: {"name": "source", "dataType": ["string"]},
                # For generic metadata, it can be stored as a JSON/object property if Weaviate version supports it,
                # or individual known metadata fields must be specified.
                # For simplicity, we'll rely on storing it within a "metadata" text field or assume it's handled by user.
                # A more robust solution would parse metadata keys and map them to Weaviate property types.
                {"name": "metadata_json", "dataType": ["text"]}, # Store stringified JSON metadata
            ]
        }
        # If vectorizer is "none", moduleConfig is not needed for it.
        # If using a specific Weaviate vectorizer module (e.g. text2vec-transformers),
        # moduleConfig would be specified here.

        try:
            if not self.client.schema.exists(class_name):
                self.client.schema.create_class(class_obj)
                logger.info(f"Successfully created Weaviate class '{class_name}'.")
            else:
                logger.info(f"Class '{class_name}' already exists in Weaviate.")
                # Optionally, update schema if needed and if `kwargs.get("update_if_exists")`
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate class '{class_name}': {e}", exc_info=True)

    async def add_documents(self, class_name: str, documents: List[DocumentChunk], **kwargs) -> List[Union[str, int, uuid.UUID]]:
        if not self.client:
            logger.error("Weaviate client not initialized. Cannot add documents.")
            return []

        added_ids = []
        with self.client.batch(
            batch_size=kwargs.get("batch_size", 100),
            dynamic=kwargs.get("dynamic_batching", True), # Enable dynamic batching
            timeout_retries=kwargs.get("timeout_retries", 3),
            # callback=weaviate.util.check_batch_result # Optional: for detailed error handling per item
        ) as batch:
            for doc in documents:
                data_object = {
                    "text": doc.text,
                    "metadata_json": json.dumps(doc.metadata or {}) # Store metadata as JSON string
                }
                # Weaviate uses UUIDs for object IDs. If doc.id is not UUID, generate one.
                try:
                    doc_uuid = uuid.UUID(str(doc.id))
                except ValueError:
                    doc_uuid = uuid.uuid4() # Generate new if doc.id is not a valid UUID

                batch.add_data_object(
                    data_object=data_object,
                    class_name=class_name,
                    vector=doc.vector, # Provide the precomputed vector
                    uuid=doc_uuid
                )
                added_ids.append(doc_uuid) # Store the UUID used

        # Check for batch errors if not using a callback that raises exceptions
        if batch.num_errors > 0:
            logger.error(f"Weaviate batch import finished with {batch.num_errors} errors.")
            # Detailed errors are in batch.errors if callback not used or doesn't raise
            # For simplicity, we assume success if no exception from context manager or callback
            # and return all attempted IDs. A more robust error handling would filter out failed IDs.
            # For now, returning all attempted IDs, client should check results or use a strict callback.
            # To be more precise, one would need to iterate batch.get_results() if available, or parse errors.
            # This example assumes if it gets here, it's mostly successful or errors handled by callback.

        logger.info(f"Batch add process completed for {len(documents)} documents to class '{class_name}'.")
        return added_ids


    async def search(self, class_name: str, query_vector: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None, **kwargs) -> List[SearchResult]:
        if not self.client:
            logger.error("Weaviate client not initialized. Cannot perform search.")
            return []

        try:
            query = self.client.query.get(class_name, ["text", "metadata_json"])

            # Weaviate filters (Where filter)
            # Example: filters = {"path": ["metadata", "source"], "operator": "Equal", "valueString": "news_api"}
            # This requires metadata fields to be top-level properties in Weaviate schema or careful pathing.
            # For "metadata_json" field, filtering is complex.
            # We might need to use GraphQL `where` filter with path for JSON fields if supported, or specific properties.
            # For now, this filter example is conceptual and might need adjustment based on actual schema.
            if filters:
                # This is a simplified filter. Real Weaviate 'where' filter is more structured.
                # Example:
                # where_filter = {
                #     "operator": "And",
                #     "operands": [
                #         {"path": ["source"], "operator": "Equal", "valueString": "some_source_value"}
                #     ]
                # }
                # query = query.with_where(where_filter)
                logger.warning(f"Weaviate filtering with generic dict is not fully implemented in this basic connector. Filter: {filters}")


            # Semantic search is done using near_vector
            query = query.with_near_vector({"vector": query_vector})
            query = query.with_limit(top_k)
            # Request _additional fields like score (certainty) and id
            query = query.with_additional(["certainty", "id", "vector"])


            response = query.do()

            results = []
            if response and f"Get" in response.get("data", {}) and class_name in response["data"]["Get"]:
                for item in response["data"]["Get"][class_name]:
                    item_id = item.get("_additional", {}).get("id")
                    certainty = item.get("_additional", {}).get("certainty") # Weaviate's score is certainty
                    vector = item.get("_additional", {}).get("vector")

                    payload = {"text": item.get("text")}
                    try:
                        metadata_str = item.get("metadata_json")
                        if metadata_str:
                            payload["metadata"] = json.loads(metadata_str)
                        else:
                            payload["metadata"] = {}
                    except json.JSONDecodeError:
                        payload["metadata"] = {"raw_metadata_json": metadata_str} # Store raw if not parsable

                    results.append(SearchResult(
                        id=item_id or str(uuid.uuid4()), # Fallback ID
                        score=certainty if certainty is not None else 0.0,
                        payload=payload,
                        vector=vector
                    ))
            return results
        except WeaviateQueryError as e:
            logger.error(f"Failed to search Weaviate class '{class_name}': {e.message}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during Weaviate search: {e}", exc_info=True)
            return []

    async def delete_documents(self, class_name: str, document_ids: List[Union[str, int, uuid.UUID]], **kwargs) -> bool:
        if not self.client:
            logger.error("Weaviate client not initialized. Cannot delete documents.")
            return False

        # Weaviate expects UUIDs as strings for deletion by ID.
        # This assumes document_ids are already valid UUID strings or can be cast.
        success_count = 0
        for doc_id_union in document_ids:
            doc_id_str = str(doc_id_union)
            try:
                # Validate if it's a UUID string, otherwise Weaviate might error
                uuid.UUID(doc_id_str) # Throws ValueError if not a valid UUID
                self.client.data_object.delete(
                    uuid=doc_id_str,
                    class_name=class_name,
                    consistency_level=weaviate.ConsistencyLevel.ONE # Or configurable
                )
                logger.debug(f"Successfully deleted document ID '{doc_id_str}' from class '{class_name}'.")
                success_count +=1
            except ValueError:
                logger.error(f"Invalid UUID format for document ID '{doc_id_str}'. Skipping deletion.")
            except Exception as e:
                logger.error(f"Failed to delete document ID '{doc_id_str}' from Weaviate class '{class_name}': {e}", exc_info=True)

        if success_count > 0:
             logger.info(f"Deletion process completed. Successfully deleted {success_count}/{len(document_ids)} documents.")
        return success_count > 0


    async def get_collection_info(self, class_name: str, **kwargs) -> Dict[str, Any]:
        if not self.client:
            logger.error("Weaviate client not initialized. Cannot get class info.")
            return {}
        try:
            schema = self.client.schema.get(class_name)
            # To get object count, you need to perform an aggregation query
            # This is a simplified way; a real count needs a GraphQL query.
            # result = self.client.query.aggregate(class_name).with_meta_count().do()
            # object_count = result["data"]["Aggregate"][class_name][0]["meta"]["count"]
            # For simplicity, returning schema info here. Count requires another call.
            logger.warning("Object count not implemented in this basic get_collection_info for Weaviate. Returning schema.")
            return {
                "name": class_name,
                "schema": schema, # Full schema dict
                "objects_count": "Not implemented in this basic version"
            }
        except Exception as e:
            logger.error(f"Failed to get info for Weaviate class '{class_name}': {e}", exc_info=True)
            return {}

    async def close(self):
        # Weaviate client does not have an explicit close() method in v3.
        # Connections are typically managed by the underlying HTTP library.
        logger.info("Weaviate client does not require explicit close.")
        self.client = None # Allow re-initialization

    async def health_check(self) -> bool:
        if not self.client:
            return False
        try:
            return self.client.is_ready() # Checks if Weaviate is live and ready
        except Exception:
            return False

# Example Usage (Conceptual - requires Weaviate instance)
if __name__ == '__main__':
    async def main():
        logging.basicConfig(level=logging.INFO)

        # Configure these based on your Weaviate setup
        WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY") # Optional

        logger.info(f"Attempting to connect to Weaviate at {WEAVIATE_URL}")
        store = WeaviateVectorStore(url=WEAVIATE_URL, api_key=WEAVIATE_API_KEY)

        if not await store.health_check():
            logger.error("Weaviate not healthy or not connected. Exiting example.")
            return

        class_name = "TestRagClass" # Must start with Uppercase
        vector_dim = 3 # Example

        await store.initialize(class_name, vector_dim, distance_metric="cosine")

        info = await store.get_collection_info(class_name)
        logger.info(f"Class info for '{class_name}': {info}")

        doc_id_1 = uuid.uuid4()
        doc_id_2 = uuid.uuid4()

        docs_to_add = [
            DocumentChunk(id=doc_id_1, text="Weaviate is a vector database.", vector=[0.1, 0.3, 0.5], metadata={"type": "db"}),
            DocumentChunk(id=doc_id_2, text="It supports GraphQL and REST APIs.", vector=[0.2, 0.4, 0.6], metadata={"type": "api"}),
        ]
        added = await store.add_documents(class_name, docs_to_add)
        logger.info(f"Added document UUIDs: {added}")

        if added:
            query_vec = [0.11, 0.31, 0.51] # Similar to doc1
            results = await store.search(class_name, query_vec, top_k=1)
            logger.info(f"Search results for {query_vec}:")
            for res in results:
                logger.info(f"  ID: {res.id}, Score (Certainty): {res.score:.4f}, Payload: {res.payload}")

        await store.close()

    import asyncio
    asyncio.run(main())
