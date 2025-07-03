import logging
from typing import List, Dict, Any, Optional, Union
import uuid
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from .vector_store_base import VectorStoreBase, DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

class ChromaVectorStore(VectorStoreBase):
    def __init__(self, path: Optional[str] = None, host: Optional[str] = None, port: Optional[int] = None,
                 use_ssl: bool = False, headers: Optional[Dict[str, str]] = None,
                 settings: Optional[ChromaSettings] = None, **kwargs):

        self.path = path
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.headers = headers
        self.client_settings = settings
        self._kwargs = kwargs # For other client settings if needed
        self.client: Optional[chromadb.HttpClient | chromadb.PersistentClient | chromadb.EphemeralClient] = None
        self._initialize_client()

    def _initialize_client(self):
        try:
            if self.host and self.port:
                protocol = "https" if self.use_ssl else "http"
                self.client = chromadb.HttpClient(
                    url=f"{protocol}://{self.host}:{self.port}",
                    settings=self.client_settings,
                    headers=self.headers,
                    **self._kwargs
                )
                logger.info(f"Chroma HttpClient initialized for {protocol}://{self.host}:{self.port}")
            elif self.path:
                self.client = chromadb.PersistentClient(
                    path=self.path,
                    settings=self.client_settings,
                    **self._kwargs
                )
                logger.info(f"Chroma PersistentClient initialized at path: {self.path}")
            else:
                self.client = chromadb.EphemeralClient(
                    settings=self.client_settings,
                    **self._kwargs
                )
                logger.info("Chroma EphemeralClient initialized (in-memory).")

            # Verify connection (heartbeat is a good way)
            self.client.heartbeat() # Raises exception on failure
            logger.info("Chroma client connection verified (heartbeat successful).")

        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}", exc_info=True)
            self.client = None

    async def initialize(self, collection_name: str, vector_size: int, distance_metric: str = "cosine", **kwargs):
        if not self.client:
            logger.error("Chroma client not initialized. Cannot create collection.")
            return

        # Chroma distance function mapping
        # Supported: "l2", "ip", "cosine"
        distance_map = {
            "cosine": "cosine",
            "l2": "l2",        # Euclidean
            "euclidean": "l2",
            "ip": "ip",        # Inner Product / Dot Product
            "dot": "ip"
        }
        chroma_distance = distance_map.get(distance_metric.lower(), "cosine")

        embedding_function = kwargs.get("embedding_function") # Optional: Chroma can manage embeddings

        try:
            # get_or_create_collection is idempotent
            collection: Collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": chroma_distance, **kwargs.get("collection_metadata", {})},
                embedding_function=embedding_function # Pass None if providing own embeddings
            )
            logger.info(f"Chroma collection '{collection.name}' ensured/created with distance: {chroma_distance}.")
            # Note: Chroma doesn't explicitly take vector_size at collection creation if you provide your own embeddings.
            # It infers it from the first batch of embeddings added.
            # If using Chroma's embedding functions, it's tied to the function's output.
        except Exception as e:
            logger.error(f"Failed to initialize Chroma collection '{collection_name}': {e}", exc_info=True)

    async def add_documents(self, collection_name: str, documents: List[DocumentChunk], **kwargs) -> List[Union[str, int, uuid.UUID]]:
        if not self.client:
            logger.error("Chroma client not initialized. Cannot add documents.")
            return []
        if not documents:
            return []

        try:
            collection = self.client.get_collection(name=collection_name)

            ids = [str(doc.id) for doc in documents]
            embeddings = [doc.vector for doc in documents]
            metadatas = [doc.metadata for doc in documents] # Chroma expects list of dicts for metadata
            contents = [doc.text for doc in documents] # Chroma calls this 'documents'

            # Chroma's add method can take embeddings directly
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=contents # This is the text content for Chroma
            )
            logger.info(f"Successfully added/updated {len(documents)} documents in Chroma collection '{collection_name}'.")
            return [doc.id for doc in documents] # Return original IDs
        except Exception as e:
            logger.error(f"Failed to add documents to Chroma collection '{collection_name}': {e}", exc_info=True)
            return []

    async def search(self, collection_name: str, query_vector: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None, **kwargs) -> List[SearchResult]:
        if not self.client:
            logger.error("Chroma client not initialized. Cannot perform search.")
            return []

        try:
            collection = self.client.get_collection(name=collection_name)

            # Chroma's where filter for metadata:
            # e.g., filters = {"source": "news_api"}
            # e.g., filters = {"year": {"$gte": 2022}}
            # This is passed as `where` argument.

            include_fields = ["metadatas", "documents", "distances"]
            if kwargs.get("with_vectors", False):
                include_fields.append("embeddings")

            query_results = collection.query(
                query_embeddings=[query_vector], # Must be a list of embeddings
                n_results=top_k,
                where=filters, # Pass filters directly
                include=include_fields
            )

            results = []
            # query_results structure for a single query vector:
            # {
            #   'ids': [['id1', 'id2']],
            #   'distances': [[0.1, 0.2]],
            #   'metadatas': [[{'meta1': val1}, {'meta2': val2}]],
            #   'documents': [['text1', 'text2']],
            #   'embeddings': [[emb1, emb2]] (if included)
            # }
            if not query_results or not query_results.get('ids') or not query_results['ids'][0]:
                return []

            num_results = len(query_results['ids'][0])
            for i in range(num_results):
                doc_id = query_results['ids'][0][i]
                distance = query_results['distances'][0][i] if query_results.get('distances') else 0.0
                metadata = query_results['metadatas'][0][i] if query_results.get('metadatas') else {}
                text_content = query_results['documents'][0][i] if query_results.get('documents') else None
                vector = query_results['embeddings'][0][i] if query_results.get('embeddings') else None

                results.append(SearchResult(
                    id=doc_id,
                    score=1.0 - distance if collection.metadata.get("hnsw:space") == "cosine" else -distance, # Higher score = better for cosine
                    payload={"text": text_content, "metadata": metadata},
                    vector=vector
                ))
            return results
        except Exception as e:
            logger.error(f"Failed to search Chroma collection '{collection_name}': {e}", exc_info=True)
            return []

    async def delete_documents(self, collection_name: str, document_ids: List[Union[str, int, uuid.UUID]], **kwargs) -> bool:
        if not self.client:
            logger.error("Chroma client not initialized. Cannot delete documents.")
            return False

        try:
            collection = self.client.get_collection(name=collection_name)
            ids_to_delete = [str(doc_id) for doc_id in document_ids]

            if not ids_to_delete: return True

            collection.delete(ids=ids_to_delete) # Can also use `where` filter
            logger.info(f"Deletion request sent for {len(ids_to_delete)} documents from Chroma collection '{collection_name}'.")
            return True # Chroma's delete doesn't return detailed status per ID easily, assumes success if no exception
        except Exception as e:
            logger.error(f"Failed to delete documents from Chroma collection '{collection_name}': {e}", exc_info=True)
            return False

    async def get_collection_info(self, collection_name: str, **kwargs) -> Dict[str, Any]:
        if not self.client:
            logger.error("Chroma client not initialized. Cannot get collection info.")
            return {}
        try:
            collection = self.client.get_collection(name=collection_name)
            return {
                "name": collection.name,
                "id": str(collection.id), # UUID
                "count": collection.count(),
                "metadata": collection.metadata, # Includes "hnsw:space"
                # "peek": collection.peek() # Returns a few items, might be large
            }
        except Exception as e:
            logger.error(f"Failed to get info for Chroma collection '{collection_name}': {e}", exc_info=True)
            return {}

    async def close(self):
        # ChromaDB client (HttpClient, PersistentClient, EphemeralClient)
        # doesn't have an explicit close() method. Connections are managed by underlying libraries.
        # For PersistentClient, changes are flushed on modification.
        logger.info("Chroma client does not require explicit close. For PersistentClient, ensure data is flushed if operations were pending (usually automatic).")
        self.client = None # Allow re-initialization

    async def health_check(self) -> bool:
        if not self.client:
            return False
        try:
            self.client.heartbeat() # Returns server nano time if healthy, raises exception otherwise
            return True
        except Exception:
            return False

# Example Usage (Conceptual)
if __name__ == '__main__':
    async def main():
        logging.basicConfig(level=logging.INFO)

        # Example for Ephemeral (in-memory) client
        logger.info("Testing Chroma EphemeralClient (in-memory)...")
        # chroma_store = ChromaVectorStore() # In-memory

        # Example for Persistent client (local disk storage)
        persistent_path = "./chroma_db_test_data"
        import shutil
        if os.path.exists(persistent_path): shutil.rmtree(persistent_path) # Clean start
        logger.info(f"Testing Chroma PersistentClient (path: {persistent_path})...")
        chroma_store = ChromaVectorStore(path=persistent_path)


        if not await chroma_store.health_check():
            logger.error("ChromaDB not healthy. Exiting example.")
            return

        collection_name = "my_rag_collection"
        vector_dim = 3 # Example size, Chroma infers this from data or embedding_function

        await chroma_store.initialize(collection_name, vector_dim, distance_metric="cosine")

        info = await chroma_store.get_collection_info(collection_name)
        logger.info(f"Collection info for '{collection_name}': {info}")

        doc_id_1 = str(uuid.uuid4())
        doc_id_2 = str(uuid.uuid4())

        docs_to_add = [
            DocumentChunk(id=doc_id_1, text="Chroma is a an AI-native open-source vector database.", vector=[0.5, 0.1, 0.2], metadata={"topic": "database"}),
            DocumentChunk(id=doc_id_2, text="It makes it easy to build LLM apps.", vector=[0.6, 0.2, 0.3], metadata={"topic": "llm_app"}),
        ]
        added_ids = await chroma_store.add_documents(collection_name, docs_to_add)
        logger.info(f"Added document IDs: {added_ids}")

        info_after_add = await chroma_store.get_collection_info(collection_name)
        logger.info(f"Collection info after add (count): {info_after_add.get('count')}")

        if added_ids and info_after_add.get('count',0) > 0:
            query_vec = [0.51, 0.11, 0.21] # Similar to doc1
            results = await chroma_store.search(collection_name, query_vec, top_k=1)
            logger.info(f"Search results for {query_vec}:")
            for res in results:
                logger.info(f"  ID: {res.id}, Score: {res.score:.4f}, Payload: {res.payload}")

        await chroma_store.close()
        if os.path.exists(persistent_path): shutil.rmtree(persistent_path) # Clean up persistent data

    import asyncio
    asyncio.run(main())
