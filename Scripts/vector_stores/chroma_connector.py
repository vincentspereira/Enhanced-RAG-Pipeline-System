import os
import chromadb
from chromadb.config import Settings as ChromaSettings
import logging
from typing import List, Dict, Any, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

# Import config loader
try:
    from Scripts.utils.config_loader import get_config_value
except ImportError:
    import sys
    # This assumes chroma_connector.py is in Scripts/vector_stores/
    # and config_loader.py is in Scripts/utils/
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from utils.config_loader import get_config_value
    except ImportError as e:
        logger.error(f"Critical: Failed to import get_config_value for ChromaConnector. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None): # Basic fallback
            return os.getenv(env_var_name, default)

class ChromaConnector:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        path: Optional[str] = None, # For persistent local storage
        collection_name: str = "default_collection",
        embedding_function_name: str = "default" # Chroma can manage its own SentenceTransformer embeddings
    ):
        self.host = host or get_config_value("CHROMA_HOST", yaml_path="vector_store.chroma.host")
        self.port = port or get_config_value("CHROMA_PORT", yaml_path="vector_store.chroma.port")
        # Path for persistent client. If host/port are set, path is usually ignored for HttpClient.
        self.path = path or get_config_value("CHROMA_PATH", yaml_path="vector_store.chroma.path", default="./chroma_data")

        self.collection_name_default = collection_name or get_config_value(
            "CHROMA_COLLECTION_NAME",
            yaml_path="vector_store.chroma.collection_name",
            default="default_collection"
        )
        # Embedding function: Chroma can use its own, or you can pass pre-computed embeddings.
        # For simplicity, this stub will rely on Chroma's default or a named SentenceTransformer.
        self.embedding_function_name = embedding_function_name or get_config_value(
            "CHROMA_EMBEDDING_FUNCTION",
            yaml_path="vector_store.chroma.embedding_function",
            default="default" # "default" uses SentenceTransformer all-MiniLM-L6-v2
        )

        self.client = None
        self.collection = None
        self._initialized = False

        self._initialize_client()
        if self.client:
            self._get_or_create_collection(self.collection_name_default)

    def _initialize_client(self):
        try:
            if self.host and self.port:
                logger.info(f"Initializing ChromaDB HttpClient for host {self.host}:{self.port}")
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=ChromaSettings(anonymized_telemetry=False) # Optional: disable telemetry
                )
            elif self.path:
                logger.info(f"Initializing ChromaDB PersistentClient at path: {self.path}")
                # Ensure directory exists for persistent client
                if not os.path.exists(self.path):
                    os.makedirs(self.path, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=self.path,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            else: # Default to an in-memory client if no host/port or path is specified
                logger.info("Initializing ChromaDB in-memory client (ephemeral).")
                self.client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))

            # Test connection (HttpClient might raise error here if server not reachable)
            self.client.heartbeat() # Checks if the server is alive (for HttpClient)
            logger.info("ChromaDB client initialized and connected successfully.")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            self.client = None
            self._initialized = False

    def _get_or_create_collection(self, collection_name: str):
        if not self.client:
            logger.error("Chroma client not initialized. Cannot get or create collection.")
            return None

        try:
            if self.embedding_function_name and self.embedding_function_name != "default":
                # If using a specific sentence transformer model with Chroma
                from chromadb.utils import embedding_functions
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_function_name)
                self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=ef)
                logger.info(f"Using embedding function: {self.embedding_function_name}")
            else: # Use Chroma's default embedding function
                self.collection = self.client.get_or_create_collection(name=collection_name)
                logger.info(f"Using Chroma's default embedding function for collection '{collection_name}'.")

            logger.info(f"ChromaDB collection '{collection_name}' is ready.")
            return self.collection
        except Exception as e:
            logger.error(f"Failed to get or create ChromaDB collection '{collection_name}': {e}", exc_info=True)
            self.collection = None
            return None

    def switch_collection(self, collection_name: str, create_if_not_exists: bool = True):
        """Switches the active collection for subsequent operations."""
        if create_if_not_exists:
            return self._get_or_create_collection(collection_name)
        else:
            if not self.client:
                logger.error("Chroma client not initialized.")
                return None
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Switched to existing ChromaDB collection '{collection_name}'.")
                return self.collection
            except Exception as e: # chromadb.errors.CollectionNotFoundException or similar
                logger.error(f"Failed to get ChromaDB collection '{collection_name}': {e}")
                self.collection = None # Ensure current collection is None if switch failed
                return None


    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ):
        """
        Adds documents to the specified ChromaDB collection.
        If collection_name is None, uses the connector's current/default collection.
        Embeddings are generated by Chroma using its configured embedding function.
        """
        target_collection = self.collection
        if collection_name:
            target_collection = self.switch_collection(collection_name, create_if_not_exists=True)

        if not target_collection:
            logger.error(f"No valid collection to add documents to ('{collection_name or self.collection_name_default}').")
            return False

        if not ids:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
        elif len(ids) != len(documents):
            logger.error("Number of IDs must match number of documents.")
            return False

        try:
            target_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {len(documents)} documents to Chroma collection '{target_collection.name}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to Chroma collection '{target_collection.name}': {e}", exc_info=True)
            return False

    def query_collection(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None, # For pre-computed embeddings
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None, # Metadata filter
        where_document_filter: Optional[Dict[str, Any]] = None, # Document content filter ($contains, $not_contains)
        collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Queries the ChromaDB collection.
        Provide either query_texts (for Chroma to embed) or query_embeddings.
        """
        target_collection = self.collection
        if collection_name:
            target_collection = self.switch_collection(collection_name, create_if_not_exists=False)
            if not target_collection: # If collection does not exist and we are not creating it
                 logger.error(f"Collection '{collection_name}' not found for querying.")
                 return None


        if not target_collection:
            logger.error(f"No valid collection to query ('{collection_name or self.collection_name_default}').")
            return None

        if not query_texts and not query_embeddings:
            logger.error("Either query_texts or query_embeddings must be provided.")
            return None

        try:
            results = target_collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_filter,
                where_document=where_document_filter,
                # include=['metadatas', 'documents', 'distances'] # Default is all
            )
            logger.info(f"Query returned {len(results.get('ids', [[]])[0]) if results else 0} results from Chroma collection '{target_collection.name}'.")
            return results
        except Exception as e:
            logger.error(f"Failed to query Chroma collection '{target_collection.name}': {e}", exc_info=True)
            return None

    def delete_collection(self, collection_name: Optional[str] = None):
        name_to_delete = collection_name or self.collection_name_default
        if not self.client:
            logger.error("Chroma client not initialized.")
            return
        try:
            self.client.delete_collection(name=name_to_delete)
            logger.info(f"Chroma collection '{name_to_delete}' deleted.")
            if self.collection and self.collection.name == name_to_delete:
                self.collection = None # Reset current collection if it was the one deleted
        except Exception as e: # chromadb.errors.CollectionNotFoundException or similar
            logger.error(f"Failed to delete Chroma collection '{name_to_delete}': {e} (It might not exist).")

    def list_collections(self) -> List[Any]:
        if not self.client:
            logger.error("Chroma client not initialized.")
            return []
        return self.client.list_collections()


# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example 1: Using HttpClient (requires a running ChromaDB server)
    # Start ChromaDB server: docker run -p 8000:8000 chromadb/chroma
    # os.environ["CHROMA_HOST"] = "localhost"
    # os.environ["CHROMA_PORT"] = "8000"
    # os.environ["CHROMA_COLLECTION_NAME"] = "my_http_collection"
    # chroma_http_connector = ChromaConnector() # Uses env vars or defaults

    # Example 2: Using PersistentClient (stores data locally)
    logger.info("\n--- Testing ChromaDB PersistentClient ---")
    persistent_path = "./test_chroma_data_persistent"
    if os.path.exists(persistent_path): # Clean up from previous run
        import shutil
        shutil.rmtree(persistent_path)

    chroma_persistent_connector = ChromaConnector(path=persistent_path, collection_name="persistent_test_coll")

    if chroma_persistent_connector._initialized:
        logger.info(f"Available collections: {chroma_persistent_connector.list_collections()}")

        # Add documents
        docs_to_add = [
            "This is document 1 about apples.",
            "Document 2 discusses bananas and their properties.",
            "The third document is an article on oranges."
        ]
        metadatas_to_add = [
            {"source": "doc_source_1", "type": "fruit_fact"},
            {"source": "doc_source_2", "type": "fruit_fact"},
            {"source": "doc_source_3", "type": "news_article"}
        ]
        ids_to_add = ["id1", "id2", "id3"]

        chroma_persistent_connector.add_documents(docs_to_add, metadatas_to_add, ids_to_add)

        # Query documents
        query_results = chroma_persistent_connector.query_collection(query_texts=["information about apples"], n_results=1)
        if query_results and query_results.get('documents'):
            logger.info(f"Query for 'apples': {query_results['documents'][0]}")
            logger.info(f"Distances: {query_results['distances'][0]}")
            logger.info(f"Metadatas: {query_results['metadatas'][0]}")


        # Query with metadata filter
        query_filtered = chroma_persistent_connector.query_collection(
            query_texts=["fruit"],
            n_results=2,
            where_filter={"type": "fruit_fact"}
        )
        if query_filtered and query_filtered.get('documents'):
             logger.info(f"Query for 'fruit' with type 'fruit_fact': {len(query_filtered['documents'][0])} results.")
             for doc in query_filtered['documents'][0]:
                 logger.info(f" - {doc[:50]}...")

        # Clean up by deleting the collection
        # chroma_persistent_connector.delete_collection()
        # logger.info(f"Cleaned up collection. Remaining collections: {chroma_persistent_connector.list_collections()}")

        # Clean up the local storage path for persistent client
        if os.path.exists(persistent_path):
            import shutil
            shutil.rmtree(persistent_path)
            logger.info(f"Cleaned up persistent storage directory: {persistent_path}")
    else:
        logger.error("ChromaDB PersistentClient connector failed to initialize.")

    # Example 3: In-memory client
    logger.info("\n--- Testing ChromaDB In-Memory Client ---")
    # Unset env vars that might point to HTTP client to ensure in-memory is chosen by default if path also not set
    # if "CHROMA_HOST" in os.environ: del os.environ["CHROMA_HOST"]
    # if "CHROMA_PORT" in os.environ: del os.environ["CHROMA_PORT"]
    # if "CHROMA_PATH" in os.environ: del os.environ["CHROMA_PATH"] # Ensure path is not set from env

    chroma_mem_connector = ChromaConnector(collection_name="in_memory_test_coll", path=None, host=None, port=None) # Explicitly no path/host/port
    if chroma_mem_connector._initialized:
        chroma_mem_connector.add_documents(["Test document for in-memory Chroma"], ids=["mem_id1"])
        mem_results = chroma_mem_connector.query_collection(query_texts=["test document"], n_results=1)
        if mem_results and mem_results.get('documents'):
            logger.info(f"In-memory query result: {mem_results['documents'][0]}")
    else:
        logger.error("ChromaDB In-Memory connector failed to initialize.")
