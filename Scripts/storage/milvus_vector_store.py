import logging
from typing import List, Dict, Any, Optional, Union
import uuid

from pymilvus import (
    connections,
    utility,
    CollectionSchema,
    FieldSchema,
    DataType,
    Collection,
    MilvusException,
)

from .vector_store_base import VectorStoreBase, DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

class MilvusVectorStore(VectorStoreBase):
    def __init__(self, host: str = "localhost", port: str = "19530", alias: str = "default",
                 user: Optional[str] = None, password: Optional[str] = None, secure: bool = False, **kwargs):
        self.host = host
        self.port = str(port) # Milvus client expects port as string
        self.alias = alias
        self.user = user
        self.password = password
        self.secure = secure
        self._kwargs = kwargs # For other connection params like server_name for TLS
        self._initialize_client()

    def _initialize_client(self):
        try:
            # Check if a connection with this alias already exists
            existing_connections = utility.list_connections()
            if any(conn[0] == self.alias for conn in existing_connections):
                logger.info(f"Connection with alias '{self.alias}' already exists. Reusing.")
                # Optionally, disconnect and reconnect if parameters might have changed
                # utility.disconnect(self.alias)

            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                secure=self.secure,
                **self._kwargs
            )
            logger.info(f"Milvus client initialized and connected to {self.host}:{self.port} with alias '{self.alias}'.")
        except MilvusException as e:
            logger.error(f"Failed to initialize Milvus client: {e}", exc_info=True)
        except Exception as e: # Catch other potential errors e.g. network
            logger.error(f"An unexpected error occurred during Milvus client initialization: {e}", exc_info=True)


    async def initialize(self, collection_name: str, vector_size: int, distance_metric: str = "L2", **kwargs):
        """
        Initializes a Milvus collection.
        Distance Metrics for Milvus: "L2", "IP" (Inner Product), "HAMMING", "JACCARD", "TANIMOTO".
        Defaulting to "L2" (Euclidean) as it's common. Cosine similarity is often achieved with IP on normalized vectors.
        """
        try:
            self._ensure_connected()
            if utility.has_collection(collection_name, using=self.alias):
                logger.info(f"Collection '{collection_name}' already exists in Milvus.")
                # Optionally, load collection if not loaded, or verify schema
                # Collection(collection_name, using=self.alias).load()
                return

            # Define schema fields
            # Primary key field - Milvus requires an int64 primary key if auto_id is False.
            # If auto_id is True (default for recent versions), Milvus generates string UUIDs.
            # For compatibility with VectorStoreBase expecting various ID types, we'll use a string ID field
            # and let Milvus handle it if auto_id=True, or we manage conversion if auto_id=False.
            # Let's use auto_id=True for simplicity with string UUIDs from DocumentChunk.

            # Milvus typically requires a primary key field and a vector field.
            # Other fields are metadata.
            # For DocumentChunk: id (string), text (string), vector (float_vector), metadata (JSON string or individual fields)

            # String IDs are fine if auto_id=True. If we want to use our own string IDs, need to ensure they are unique.
            # Milvus primary key constraints: int64 or varchar, must be specified.
            # If we use DocumentChunk.id (which can be str(uuid.uuid4())), we need a varchar primary key.
            pk_field = FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, max_length=36) # Max UUID length
            vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size)
            text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=kwargs.get("text_max_length", 65535)) # Max length for VARCHAR
            metadata_json_field = FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=kwargs.get("metadata_max_length", 65535))

            schema = CollectionSchema(
                fields=[pk_field, vector_field, text_field, metadata_json_field],
                description=kwargs.get("description", f"Collection for {collection_name}"),
                auto_id=False, # We will provide our own string IDs from DocumentChunk
                primary_field="doc_id"
            )

            collection = Collection(collection_name, schema=schema, using=self.alias)
            logger.info(f"Milvus collection '{collection_name}' created with schema.")

            # Create an index for the vector field for efficient searching
            index_params = kwargs.get("index_params", {
                "metric_type": distance_metric.upper(), # L2, IP, COSINE (if supported directly by index)
                "index_type": "IVF_FLAT", # Or HNSW, etc.
                "params": {"nlist": 128}, # Example params for IVF_FLAT
            })
            # Note: For COSINE with some indexes, vectors should be normalized.
            # Milvus handles "COSINE" metric type for some indexes directly now.
            # If using "IP" for cosine similarity, ensure vectors are normalized before insertion.

            collection.create_index(field_name="vector", index_params=index_params)
            logger.info(f"Index created for vector field in '{collection_name}' with params: {index_params}")

            collection.load() # Load collection into memory for searching
            logger.info(f"Collection '{collection_name}' loaded.")

        except MilvusException as e:
            logger.error(f"Failed to initialize Milvus collection '{collection_name}': {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error initializing Milvus collection '{collection_name}': {e}", exc_info=True)

    async def add_documents(self, collection_name: str, documents: List[DocumentChunk], **kwargs) -> List[Union[str, int, uuid.UUID]]:
        self._ensure_connected()
        try:
            collection = Collection(collection_name, using=self.alias)
            # Ensure collection is loaded
            if not collection.has_index(): # Or check if loaded: utility.get_query_segment_info might be too low level
                 logger.info(f"Collection '{collection_name}' has no index or might not be loaded. Attempting to load...")
                 collection.load()


            insert_data = []
            doc_ids = []
            for doc in documents:
                doc_id_str = str(doc.id)
                insert_data.append([
                    doc_id_str,
                    doc.vector,
                    doc.text,
                    json.dumps(doc.metadata or {})
                ])
                doc_ids.append(doc.id)

            if not insert_data:
                return []

            # Data format for insert: [[pk_values], [vector_values], [text_values], [metadata_json_values]]
            # Or a list of lists/tuples, where each inner list/tuple corresponds to a row.
            # The order of fields in inner list must match schema: doc_id, vector, text, metadata_json

            # Milvus client expects list of lists for columns, or list of dicts for rows (newer SDK versions).
            # For clarity, using list of lists matching field order.
            # data_to_insert = [
            #     [d[0] for d in insert_data], # doc_ids
            #     [d[1] for d in insert_data], # vectors
            #     [d[2] for d in insert_data], # texts
            #     [d[3] for d in insert_data]  # metadata_jsons
            # ]
            # This column-oriented format is common but error-prone.
            # Row-oriented might be: entities = [{"doc_id": "id1", "vector": [], ...}, ...]
            # The `insert_data` list of lists (rows) format is generally accepted by `collection.insert()`.

            mutation_result = collection.insert(insert_data)
            collection.flush() # Ensure data is written to disk segment

            if mutation_result.insert_count == len(documents):
                logger.info(f"Successfully added {mutation_result.insert_count} documents to Milvus collection '{collection_name}'.")
                return doc_ids # Return original IDs
            else:
                logger.warning(f"Partial insert to Milvus: expected {len(documents)}, inserted {mutation_result.insert_count}. Errors: {mutation_result.err_handler.exceptions}")
                # Need to map which IDs failed if possible from mutation_result.primary_keys and errors
                return [doc_ids[i] for i, pk in enumerate(mutation_result.primary_keys) if pk] # Simplistic
        except MilvusException as e:
            logger.error(f"Failed to add documents to Milvus collection '{collection_name}': {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error adding documents to Milvus '{collection_name}': {e}", exc_info=True)
            return []


    async def search(self, collection_name: str, query_vector: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None, **kwargs) -> List[SearchResult]:
        self._ensure_connected()
        try:
            collection = Collection(collection_name, using=self.alias)
            if not collection.has_index(): # Or check if loaded
                 logger.info(f"Collection '{collection_name}' has no index or might not be loaded. Attempting to load...")
                 collection.load()

            # Milvus filter expression (expr)
            # Example: filters = {"metadata.source": "news"} -> expr = "metadata_json like '%\"source\": \"news\"%'" (fragile)
            # Or if metadata fields are actual schema fields: expr = "source == 'news'"
            # For JSON field, it's more complex: json_extract(metadata_json, '$.source') == 'news' (if Milvus supports JSON functions)
            # This part needs robust filter conversion.
            expr_filter = None
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    # Assuming metadata is stored as a JSON string in 'metadata_json' field
                    if key.startswith("metadata."):
                        # This is a very basic and potentially fragile way to filter JSON strings.
                        # Milvus might have better JSON support in newer versions or require specific syntax.
                        json_path_key = key.split("metadata.", 1)[1]
                        # Escape quotes in value if it's a string
                        if isinstance(value, str):
                            value_str = value.replace("'", "''").replace('"', '\\"')
                            filter_parts.append(f"metadata_json like '%\"{json_path_key}\": \"{value_str}\"%'")
                        else: # For numbers or booleans in JSON
                            filter_parts.append(f"metadata_json like '%\"{json_path_key}\": {json.dumps(value)}%'")
                    else: # Direct field
                        if isinstance(value, str):
                            filter_parts.append(f"{key} == '{value}'")
                        else:
                            filter_parts.append(f"{key} == {value}")
                if filter_parts:
                    expr_filter = " and ".join(filter_parts)
                logger.info(f"Using Milvus filter expression: {expr_filter}")

            search_params = kwargs.get("search_params", {
                "metric_type": kwargs.get("distance_metric", "L2").upper(), # Match index metric
                "params": {"nprobe": 10}, # Example for IVF_FLAT
            })

            # Output fields should include the ones we stored: doc_id, text, metadata_json
            output_fields = ["doc_id", "text", "metadata_json"]

            hits = collection.search(
                data=[query_vector], # Search with a list of vectors
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=expr_filter,
                output_fields=output_fields,
                consistency_level=kwargs.get("consistency_level", "Strong") # Or Bounded, Eventually
            )

            results = []
            # hits is a list of SearchResult objects (from Milvus client), one per query vector
            # Since we search with one query_vector, we take hits[0]
            for hit in hits[0]:
                doc_id = hit.entity.get("doc_id")
                text_content = hit.entity.get("text")
                metadata_json_str = hit.entity.get("metadata_json")
                metadata = {}
                try:
                    if metadata_json_str: metadata = json.loads(metadata_json_str)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse metadata_json for doc_id {doc_id}")
                    metadata = {"raw_metadata_json": metadata_json_str}

                results.append(SearchResult(
                    id=doc_id or str(uuid.uuid4()), # Fallback if ID not in output_fields
                    score=hit.distance, # Milvus returns distance, lower is better for L2/Hamming, higher for IP/Cosine
                    payload={"text": text_content, "metadata": metadata}
                    # vector=hit.entity.get("vector") # If vector field is requested in output_fields
                ))
            return results
        except MilvusException as e:
            logger.error(f"Failed to search Milvus collection '{collection_name}': {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching Milvus '{collection_name}': {e}", exc_info=True)
            return []

    async def delete_documents(self, collection_name: str, document_ids: List[Union[str, int, uuid.UUID]], **kwargs) -> bool:
        self._ensure_connected()
        try:
            collection = Collection(collection_name, using=self.alias)
            # Milvus delete expression uses 'in' for list of primary keys
            # Assuming doc_id is the primary key field name and is VARCHAR
            ids_str = [f"'{str(doc_id)}'" for doc_id in document_ids]
            expr = f"doc_id in [{','.join(ids_str)}]"

            delete_result = collection.delete(expr)
            collection.flush() # Ensure deletions are committed

            # delete_result might be a MutationResult, check its properties
            # For now, assume success if no exception. A more robust check would inspect delete_result.
            logger.info(f"Deletion attempt completed for {len(document_ids)} IDs in Milvus collection '{collection_name}'. Result: {delete_result}")
            return True # Or check delete_result.delete_count if available and accurate
        except MilvusException as e:
            logger.error(f"Failed to delete documents from Milvus collection '{collection_name}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting from Milvus '{collection_name}': {e}", exc_info=True)
            return False

    async def get_collection_info(self, collection_name: str, **kwargs) -> Dict[str, Any]:
        self._ensure_connected()
        try:
            if not utility.has_collection(collection_name, using=self.alias):
                return {"error": "Collection not found"}

            collection = Collection(collection_name, using=self.alias)
            # collection.load() # Ensure it's loaded to get num_entities, but this can be slow

            return {
                "name": collection.name,
                "description": collection.description,
                "schema": collection.schema, # Returns CollectionSchema object
                "num_entities": collection.num_entities, # Number of vectors/documents
                "primary_field": collection.primary_field.name,
                "indexes": collection.indexes, # List of Index objects
                "is_empty": collection.is_empty,
                # "partitions": collection.partitions, # List of Partition objects
                # "consistency_level": collection.consistency_level, # String representation
            }
        except MilvusException as e:
            logger.error(f"Failed to get info for Milvus collection '{collection_name}': {e}", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting info for Milvus '{collection_name}': {e}", exc_info=True)
            return {}

    async def close(self):
        try:
            # Connections are managed globally by alias in pymilvus
            utility.disconnect(self.alias)
            logger.info(f"Milvus connection alias '{self.alias}' disconnected.")
        except Exception as e:
            logger.error(f"Error disconnecting Milvus alias '{self.alias}': {e}", exc_info=True)

    def _ensure_connected(self):
        """Checks if connection for alias exists, attempts to reconnect if not."""
        try:
            # utility.has_connection(self.alias) is one way, or just try using it.
            # A simple check could be to list collections or a lightweight op.
            # For simplicity, we rely on operations failing and then potentially re-init.
            # A more robust check would be:
            if not any(conn[0] == self.alias for conn in utility.list_connections()):
                 logger.warning(f"No active Milvus connection for alias '{self.alias}'. Attempting to reconnect.")
                 self._initialize_client()
            # Or, if the client object itself stores state that can be checked:
            # if self.client and not self.client.is_connected(): self._initialize_client()
        except Exception as e:
            logger.warning(f"Error during Milvus connection check for alias '{self.alias}', attempting reconnect: {e}")
            self._initialize_client() # Try to re-establish

    async def health_check(self) -> bool:
        try:
            self._ensure_connected() # Make sure we try to connect
            # A basic health check could be to list collections or check server status
            # This is a lightweight operation.
            utility.list_collections(using=self.alias)
            return True
        except MilvusException: # Specific Milvus client/server errors
            return False
        except Exception: # Broader connectivity issues
            return False

# Example Usage (Conceptual - requires Milvus instance)
if __name__ == '__main__':
    async def main():
        logging.basicConfig(level=logging.INFO)

        # Configure these based on your Milvus setup
        MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
        MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

        logger.info(f"Attempting to connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        store = MilvusVectorStore(host=MILVUS_HOST, port=MILVUS_PORT)

        if not await store.health_check():
            logger.error("Milvus not healthy or not connected. Exiting example.")
            return

        collection_name = "TestRagMilvusCol"
        vector_dim = 3 # Example

        # Clean up old collection if it exists for test idempotency
        if utility.has_collection(collection_name, using=store.alias):
            logger.info(f"Dropping existing collection '{collection_name}' for test.")
            utility.drop_collection(collection_name, using=store.alias)
            time.sleep(1) # Give Milvus a moment

        await store.initialize(collection_name, vector_dim, distance_metric="L2")

        info = await store.get_collection_info(collection_name)
        logger.info(f"Collection info for '{collection_name}': {info}")

        doc_id_1_str = str(uuid.uuid4())
        doc_id_2_str = str(uuid.uuid4())

        docs_to_add = [
            DocumentChunk(id=doc_id_1_str, text="Milvus is a vector database for AI.", vector=[0.1, 0.2, 0.7], metadata={"category": "db"}),
            DocumentChunk(id=doc_id_2_str, text="It is open source and highly scalable.", vector=[0.3, 0.4, 0.8], metadata={"category": "tech"}),
        ]
        added = await store.add_documents(collection_name, docs_to_add)
        logger.info(f"Added document IDs: {added}") # These should be the original string IDs

        # Wait a bit for Milvus to index (especially if segments were just flushed)
        time.sleep(2)

        info_after_add = await store.get_collection_info(collection_name)
        logger.info(f"Collection info after add (num_entities): {info_after_add.get('num_entities')}")


        if added and info_after_add.get('num_entities', 0) > 0:
            query_vec = [0.11, 0.21, 0.71] # Similar to doc1
            results = await store.search(collection_name, query_vec, top_k=1)
            logger.info(f"Search results for {query_vec}:")
            for res in results:
                logger.info(f"  ID: {res.id}, Score (Distance): {res.score:.4f}, Payload: {res.payload}")
        else:
            logger.warning("No documents seem to be added or collection not ready for search.")

        await store.close()

    import asyncio
    asyncio.run(main())
