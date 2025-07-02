import faiss
import numpy as np
import logging
import os
from sentence_transformers import SentenceTransformer # For generating dummy embeddings

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure a directory for saving the index exists
INDEX_DIR = "data/faiss_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)
INDEX_FILE_PATH = os.path.join(INDEX_DIR, "my_faiss_index.idx")
ID_MAP_FILE_PATH = os.path.join(INDEX_DIR, "my_faiss_id_map.json") # To map FAISS indices to original document IDs

class FaissIndexManager:
    def __init__(self, embedding_dimension: int, index_path: str = INDEX_FILE_PATH, id_map_path: str = ID_MAP_FILE_PATH):
        self.embedding_dimension = embedding_dimension
        self.index_path = index_path
        self.id_map_path = id_map_path

        self.index = None
        self.id_map = {} # Maps sequential FAISS index to actual document ID
        self.next_internal_idx = 0 # Counter for sequential FAISS index

        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.id_map_path):
            try:
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                import json
                with open(self.id_map_path, 'r') as f:
                    # Convert string keys back to int for id_map
                    loaded_id_map_str_keys = json.load(f)
                    self.id_map = {int(k): v for k, v in loaded_id_map_str_keys.get("id_map", {}).items()}
                    self.next_internal_idx = loaded_id_map_str_keys.get("next_internal_idx", 0)

                # Sanity check
                if self.index.ntotal != len(self.id_map):
                    logger.warning(f"FAISS index size ({self.index.ntotal}) does not match ID map size ({len(self.id_map)}). May need rebuild.")
                    # Optionally, could trigger a rebuild or raise an error.
                    # For this example, we'll proceed but this indicates potential inconsistency.

                logger.info(f"FAISS index loaded with {self.index.ntotal} vectors. Next internal index: {self.next_internal_idx}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index or ID map: {e}. Creating a new one.", exc_info=True)
                self._create_new_index()
        else:
            logger.info("No existing FAISS index found. Creating a new one.")
            self._create_new_index()

    def _create_new_index(self):
        # Using IndexFlatL2 for simplicity. For larger datasets, consider more advanced indexes
        # like IndexIVFFlat, IndexHNSWFlat, etc., which require training.
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        # If you need to store original IDs alongside vectors (especially if vectors can be removed/updated)
        # FAISS also supports IndexIDMap which wraps another index:
        # self.index_flat = faiss.IndexFlatL2(self.embedding_dimension)
        # self.index = faiss.IndexIDMap(self.index_flat) # This allows adding vectors with custom integer IDs
        # However, managing custom IDs with IndexIDMap and our separate string ID map can be redundant.
        # The current approach uses a sequential FAISS index and a separate map for string IDs.

        self.id_map = {}
        self.next_internal_idx = 0
        logger.info(f"Created new FAISS IndexFlatL2 with dimension {self.embedding_dimension}")

    def add_embeddings(self, embeddings: np.ndarray, document_ids: List[str]):
        """
        Adds embeddings to the FAISS index.
        Args:
            embeddings (np.ndarray): A 2D numpy array of shape (n_samples, embedding_dimension).
            document_ids (List[str]): A list of document IDs corresponding to the embeddings.
                                      Must be the same length as the number of embeddings.
        """
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D numpy array.")
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dimension}, got {embeddings.shape[1]}")
        if len(document_ids) != embeddings.shape[0]:
            raise ValueError("Number of document IDs must match number of embeddings.")
        if not self.index:
            logger.error("FAISS index is not initialized.")
            return

        # FAISS expects float32
        embeddings = embeddings.astype('float32')

        # Store mapping from FAISS sequential index to actual document ID
        current_batch_indices = []
        for doc_id in document_ids:
            self.id_map[self.next_internal_idx] = doc_id
            current_batch_indices.append(self.next_internal_idx)
            self.next_internal_idx += 1

        # If using IndexIDMap, you'd add with the actual IDs (must be int64)
        # self.index.add_with_ids(embeddings, np.array(custom_integer_ids_for_faiss))
        self.index.add(embeddings)
        logger.info(f"Added {embeddings.shape[0]} embeddings. Total vectors in index: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the FAISS index for similar embeddings.
        Args:
            query_embedding (np.ndarray): A 1D numpy array of shape (embedding_dimension,).
            top_k (int): Number of nearest neighbors to retrieve.
        Returns:
            List of dictionaries, each containing 'document_id', 'distance', and 'faiss_index'.
        """
        if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1:
            # If it's (1, dim), reshape it.
            if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
                query_embedding = query_embedding.reshape(-1)
            else:
                raise ValueError("Query embedding must be a 1D numpy array or a 2D array with one row.")

        if query_embedding.shape[0] != self.embedding_dimension:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.embedding_dimension}, got {query_embedding.shape[0]}")
        if not self.index or self.index.ntotal == 0:
            logger.warning("FAISS index is not initialized or is empty.")
            return []

        query_embedding = query_embedding.astype('float32').reshape(1, -1) # Reshape to (1, dim) for search

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in range(len(indices[0])):
            faiss_idx = indices[0][i]
            dist = distances[0][i]
            if faiss_idx != -1: # -1 means no neighbor found (e.g., if k > ntotal)
                doc_id = self.id_map.get(faiss_idx)
                if doc_id: # Should always exist if faiss_idx is valid
                    results.append({
                        "document_id": doc_id,
                        "distance": float(dist),
                        "faiss_index": int(faiss_idx)
                    })
                else:
                    logger.warning(f"Could not find document ID for FAISS index {faiss_idx}")
        return results

    def save_index(self):
        if not self.index:
            logger.error("FAISS index is not initialized. Nothing to save.")
            return
        try:
            logger.info(f"Saving FAISS index to {self.index_path} and ID map to {self.id_map_path}")
            faiss.write_index(self.index, self.index_path)
            import json
            # Convert int keys in id_map to strings for JSON compatibility
            id_map_to_save = {str(k): v for k, v in self.id_map.items()}
            with open(self.id_map_path, 'w') as f:
                json.dump({"id_map": id_map_to_save, "next_internal_idx": self.next_internal_idx}, f)
            logger.info("FAISS index and ID map saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index or ID map: {e}", exc_info=True)

    def get_vector_by_faiss_id(self, faiss_id: int) -> Optional[np.ndarray]:
        """Reconstructs a vector from the index given its FAISS internal ID."""
        if not self.index or faiss_id < 0 or faiss_id >= self.index.ntotal:
            logger.warning(f"Invalid FAISS ID {faiss_id} or index not ready.")
            return None
        try:
            return self.index.reconstruct(faiss_id)
        except Exception as e: # reconstruction might not be supported by all index types or if not stored.
            logger.error(f"Failed to reconstruct vector for FAISS ID {faiss_id}: {e}")
            return None


if __name__ == "__main__":
    logger.info("--- FAISS Example Utility ---")

    # 0. Setup: Initialize an embedding model (e.g., SentenceTransformer)
    # This is just for generating dummy embeddings for the example.
    # In a real application, embeddings would come from your documents.
    try:
        embed_model_name = 'all-MiniLM-L6-v2' # ~384 dimensions
        logger.info(f"Loading SentenceTransformer model: {embed_model_name} for dummy embeddings...")
        sbert_model = SentenceTransformer(embed_model_name)
        EMBEDDING_DIM = sbert_model.get_sentence_embedding_dimension()
    except Exception as e:
        logger.error(f"Could not load SentenceTransformer model for example: {e}. Using random embeddings.")
        sbert_model = None
        EMBEDDING_DIM = 128 # Fallback dimension if model load fails

    # 1. Initialize FaissIndexManager
    # If an index file exists at data/faiss_indexes/my_faiss_index.idx, it will be loaded.
    # Otherwise, a new index is created.
    faiss_manager = FaissIndexManager(embedding_dimension=EMBEDDING_DIM)

    # 2. Prepare some dummy data and embeddings
    if faiss_manager.index.ntotal == 0: # Only add if index is empty
        logger.info("Index is empty. Adding dummy data...")
        dummy_documents = [
            "FAISS is a library for efficient similarity search.",
            "It was developed by Facebook AI Research (FAIR).",
            "FAISS supports GPU acceleration for faster searches.",
            "Embeddings are numerical representations of text or other data.",
            "Similarity search finds items with embeddings close to a query embedding."
        ]
        dummy_doc_ids = [f"doc_{i+1}" for i in range(len(dummy_documents))]

        if sbert_model:
            logger.info("Generating embeddings for dummy documents...")
            dummy_embeddings = sbert_model.encode(dummy_documents, convert_to_numpy=True)
        else: # Fallback to random embeddings
            logger.info(f"Generating random dummy embeddings of dimension {EMBEDDING_DIM}...")
            dummy_embeddings = np.random.rand(len(dummy_documents), EMBEDDING_DIM).astype('float32')

        # 3. Add embeddings to the index
        faiss_manager.add_embeddings(dummy_embeddings, dummy_doc_ids)
        faiss_manager.save_index() # Save after adding
    else:
        logger.info(f"Index already contains {faiss_manager.index.ntotal} vectors. Skipping dummy data addition.")


    # 4. Perform a search
    if sbert_model:
        query_text = "efficient similarity search for embeddings"
        logger.info(f"\nSearching for: '{query_text}'")
        query_embedding_np = sbert_model.encode([query_text], convert_to_numpy=True)[0]
    else:
        logger.info(f"\nSearching with a random query vector (dim={EMBEDDING_DIM})...")
        query_embedding_np = np.random.rand(EMBEDDING_DIM).astype('float32')

    search_results = faiss_manager.search(query_embedding_np, top_k=3)

    logger.info("\nSearch Results:")
    if search_results:
        for result in search_results:
            logger.info(f"  Document ID: {result['document_id']}, Distance: {result['distance']:.4f}, FAISS Index: {result['faiss_index']}")
            # Optionally, reconstruct and show the vector (if stored and reconstructible)
            # reconstructed_vec = faiss_manager.get_vector_by_faiss_id(result['faiss_index'])
            # if reconstructed_vec is not None:
            #     logger.info(f"    Reconstructed Vector (first 5 dims): {reconstructed_vec[:5]}")
    else:
        logger.info("  No results found or index is empty.")

    # 5. Save the index (optional, already saved after adding if new)
    # faiss_manager.save_index()

    logger.info("\nFAISS example finished.")
    logger.info(f"Index file is at: {faiss_manager.index_path}")
    logger.info(f"ID map file is at: {faiss_manager.id_map_path}")
