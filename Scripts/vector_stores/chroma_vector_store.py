import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .base import VectorStore # Assuming base.py is in the same directory

# Assuming ChromaDBConfig is defined in config.manager:
# from ..config.manager import ChromaDBConfig

logger = logging.getLogger(__name__)

class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, config: Any): # config should be compatible with ChromaDBConfig
        self.mode = getattr(config, 'mode', 'memory').lower()
        self.collection_name = getattr(config, 'collection_name', 'rag_chroma_collection')

        client_settings = ChromaSettings()

        if self.mode == "local_persistent":
            persist_dir_str = getattr(config, 'persist_directory', 'chroma_db_persistence')
            # Ensure persist_directory is an absolute path or relative to a known base
            # For simplicity, assume it's a path string that chromadb can handle.
            # Creating the directory if it doesn't exist.
            self.persist_directory = Path(persist_dir_str)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"ChromaDB initializing in local persistent mode. Path: {self.persist_directory}")
            client_settings.is_persistent = True # This is how you enable persistence with chromadb.PersistentClient
            client_settings.persist_directory = str(self.persist_directory)
            self.client = chromadb.PersistentClient(path=str(self.persist_directory), settings=client_settings)

        elif self.mode == "remote_http":
            host = getattr(config, 'host', 'localhost')
            port = getattr(config, 'port', 8000)
            logger.info(f"ChromaDB initializing in remote HTTP mode. Host: {host}, Port: {port}")
            self.client = chromadb.HttpClient(host=host, port=port, settings=client_settings)

        elif self.mode == "memory":
            logger.info("ChromaDB initializing in in-memory mode.")
            self.client = chromadb.Client(settings=client_settings) # In-memory client
        else:
            raise ValueError(f"Unsupported ChromaDB mode: {self.mode}. Choose from 'memory', 'local_persistent', 'remote_http'.")

        try:
            # Get or create collection. ChromaDB's get_or_create_collection handles this.
            # We might want to specify metadata like {'hnsw:space': 'cosine'} for distance metric if needed,
            # or embedding function if not providing embeddings directly.
            # For this VectorStore interface, we assume embeddings are pre-computed.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
                # metadata={"hnsw:space": "cosine"} # Example for cosine distance
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' ensured/created.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection '{self.collection_name}': {e}", exc_info=True)
            raise

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """Add vectors with associated metadata to the ChromaDB store."""
        if not self.collection:
            logger.error("ChromaDB collection not initialized.")
            raise ConnectionError("ChromaDB collection not available.")

        if ids is None:
            # Chroma requires IDs. Generate if not provided.
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif len(ids) != len(vectors):
            raise ValueError("Length of ids must match length of vectors.")

        # Chroma expects embeddings as List[List[float]]
        embeddings_list = [v.tolist() for v in vectors]

        # Filter out None metadata, Chroma expects dicts or None for all
        processed_metadata = [m if m is not None else {} for m in metadata]


        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=processed_metadata # Ensure this is a list of dicts
            )
            logger.info(f"Added/updated {len(vectors)} vectors to ChromaDB collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to add vectors to ChromaDB: {e}", exc_info=True)
            raise

    def search(self, query_vector: np.ndarray, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in ChromaDB."""
        if not self.collection:
            logger.error("ChromaDB collection not initialized.")
            raise ConnectionError("ChromaDB collection not available.")

        query_embedding_list = query_vector.tolist()

        # ChromaDB's where filter for metadata:
        # Example: filters = {"source": "doc_x"}  -> where={"source": "doc_x"}
        # Example: filters = {"pages": {"$gte": 10}} -> where={"pages": {"$gte": 10}} (Chroma supports operators)
        where_filter = filters if filters else None

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding_list], # Expects a list of query embeddings
                n_results=top_k,
                where=where_filter,
                include=['metadatas', 'distances'] # Request distances (scores) and metadatas
            )

            # Process results into the expected format
            # Results['ids'][0], Results['distances'][0], Results['metadatas'][0] will be lists for the single query
            formatted_results = []
            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    # Chroma's distance can be squared L2, L2, IP, or cosine.
                    # For cosine, distance = 1 - similarity. Higher similarity = lower distance.
                    # The VectorStore interface implies higher score is better.
                    # If using cosine distance, score = 1 - distance. If L2, score might be -distance or 1/(1+distance).
                    # Assuming cosine for now, so score = 1 - distance.
                    # This needs to align with collection's distance metric.
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                    score = 1 - distance if distance is not None else 0.0 # Adjust if not cosine

                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "score": score,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    })
            logger.debug(f"ChromaDB search returned {len(formatted_results)} results for top_k={top_k}.")
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search in ChromaDB: {e}", exc_info=True)
            raise

    def delete_vectors(self, ids: List[str]):
        """Delete vectors by their IDs from ChromaDB."""
        if not self.collection:
            logger.error("ChromaDB collection not initialized.")
            raise ConnectionError("ChromaDB collection not available.")

        try:
            if not ids:
                logger.warning("No IDs provided for deletion in ChromaDB.")
                return

            self.collection.delete(ids=ids)
            logger.info(f"Attempted to delete {len(ids)} vectors from ChromaDB collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to delete vectors from ChromaDB: {e}", exc_info=True)
            raise

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    class MockChromaDBConfig:
        mode: str = "memory" # "memory", "local_persistent", "remote_http"
        # persist_directory: str = "temp_chroma_data" # For local_persistent
        # host: str = "localhost" # For remote_http
        # port: int = 8000        # For remote_http
        collection_name: str = "my_test_collection_chroma"

    test_chroma_config = MockChromaDBConfig()

    # Clean up persistent directory if it exists from a previous run for 'local_persistent'
    # if test_chroma_config.mode == "local_persistent":
    #     import shutil
    #     if Path(test_chroma_config.persist_directory).exists():
    #         shutil.rmtree(test_chroma_config.persist_directory)
    #         print(f"Cleaned up old persist directory: {test_chroma_config.persist_directory}")


    chroma_store = None
    try:
        chroma_store = ChromaVectorStore(config=test_chroma_config)
        print(f"ChromaDB store initialized in '{chroma_store.mode}' mode for collection '{chroma_store.collection_name}'.")

        # Test add_vectors
        print("\n--- Adding Vectors ---")
        num_vectors = 5
        vector_dim = 10 # Example dimension
        test_vectors = np.random.rand(num_vectors, vector_dim).astype(np.float32)
        test_metadata = [{"source": f"doc_{i}", "page": i+1, "type": "test"} for i in range(num_vectors)]
        test_ids = [f"id_{i}" for i in range(num_vectors)]

        chroma_store.add_vectors(vectors=test_vectors, metadata=test_metadata, ids=test_ids)
        print(f"Added {num_vectors} vectors.")

        # Give some time for additions to be processed, especially if async or network involved (though client is sync)
        import time
        time.sleep(0.1)

        # Verify count (Chroma's count() method)
        print(f"Collection count: {chroma_store.collection.count()}")


        # Test search
        print("\n--- Searching Vectors ---")
        query_vec = np.random.rand(1, vector_dim).astype(np.float32)[0]
        search_results = chroma_store.search(query_vector=query_vec, top_k=3, filters={"type": "test"})
        print(f"Search results for a random vector (top 3, type='test'):")
        for res in search_results:
            print(f"  ID: {res['id']}, Score: {res['score']:.4f}, Metadata: {res['metadata']}")

        search_results_no_filter = chroma_store.search(query_vector=query_vec, top_k=2)
        print(f"Search results for a random vector (top 2, no filter):")
        for res in search_results_no_filter:
            print(f"  ID: {res['id']}, Score: {res['score']:.4f}, Metadata: {res['metadata']}")


        # Test delete_vectors
        print("\n--- Deleting Vectors ---")
        ids_to_delete = [test_ids[0], test_ids[2]]
        chroma_store.delete_vectors(ids=ids_to_delete)
        print(f"Attempted to delete vectors with IDs: {ids_to_delete}")
        time.sleep(0.1)
        print(f"Collection count after delete: {chroma_store.collection.count()}")

        # Verify deletion by trying to get one of the deleted IDs (should not be found easily by search)
        # Or, if Chroma client allows fetching by ID, use that.
        # For now, just checking count and re-searching.
        remaining_search = chroma_store.search(query_vector=test_vectors[0], top_k=1) # Search for the deleted vector
        print(f"Search for deleted vector {test_ids[0]}: {remaining_search}")
        if not any(r['id'] == test_ids[0] for r in remaining_search):
             print(f"Vector {test_ids[0]} appears to be deleted or no longer a top match.")


    except Exception as e:
        print(f"An error occurred during ChromaDB example: {e}", exc_info=True)
    finally:
        # If using local_persistent mode and created a temp dir for testing, clean it up.
        # if test_chroma_config.mode == "local_persistent" and chroma_store and Path(chroma_store.persist_directory).exists():
        #     import shutil
        #     shutil.rmtree(chroma_store.persist_directory)
        #     print(f"Cleaned up persist directory: {chroma_store.persist_directory}")
        pass
