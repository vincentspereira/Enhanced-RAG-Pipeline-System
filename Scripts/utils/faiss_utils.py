import faiss
import numpy as np
import logging
from typing import Optional, Tuple, List
from pathlib import Path
import os

logger = logging.getLogger(__name__)

def build_faiss_index(embeddings: np.ndarray, index_type: str = "IndexFlatL2") -> Optional[faiss.Index]:
    """
    Builds a FAISS index from a given set of embeddings.

    Args:
        embeddings (np.ndarray): A 2D NumPy array of shape (num_vectors, dim)
                                 where each row is an embedding. Embeddings should be float32.
        index_type (str): The type of FAISS index to build.
                          Defaults to "IndexFlatL2". Other examples: "IndexFlatIP" (Inner Product),
                          "IndexIVFFlat" (requires training), etc.

    Returns:
        Optional[faiss.Index]: The built FAISS index, or None if an error occurs.
    """
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        logger.error("Embeddings must be a 2D NumPy array.")
        return None
    if embeddings.dtype != np.float32:
        logger.warning(f"FAISS typically expects float32 embeddings. Input dtype is {embeddings.dtype}. Casting to float32.")
        embeddings = embeddings.astype(np.float32)

    num_vectors, dim = embeddings.shape
    logger.info(f"Building FAISS index of type '{index_type}' with {num_vectors} vectors of dimension {dim}.")

    try:
        if index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(dim)
        elif index_type == "IndexFlatIP": # For cosine similarity, vectors should be normalized, and use Inner Product
            index = faiss.IndexFlatIP(dim)
        # Add more index types here as needed, e.g., IndexIVFFlat which requires training
        # elif index_type == "IndexIVFFlat":
        #     quantizer = faiss.IndexFlatL2(dim) # Example quantizer for IVFFlat
        #     nlist = int(np.sqrt(num_vectors)) # Heuristic for number of Voronoi cells
        #     index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        #     if not index.is_trained:
        #         index.train(embeddings) # Training is required for IVF indexes
        else:
            logger.error(f"Unsupported FAISS index type: {index_type}")
            return None

        index.add(embeddings)
        logger.info(f"Successfully built FAISS index. Total vectors in index: {index.ntotal}")
        return index
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}", exc_info=True)
        return None

def save_faiss_index(index: faiss.Index, file_path: Union[str, Path]):
    """
    Saves a FAISS index to a file.

    Args:
        index (faiss.Index): The FAISS index to save.
        file_path (Union[str, Path]): The path where the index will be saved.
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(file_path))
        logger.info(f"FAISS index saved to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save FAISS index to {file_path}: {e}", exc_info=True)
        raise

def load_faiss_index(file_path: Union[str, Path]) -> Optional[faiss.Index]:
    """
    Loads a FAISS index from a file.

    Args:
        file_path (Union[str, Path]): The path to the saved FAISS index file.

    Returns:
        Optional[faiss.Index]: The loaded FAISS index, or None if an error occurs.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"FAISS index file not found: {file_path}")
            return None
        index = faiss.read_index(str(file_path))
        logger.info(f"FAISS index loaded from: {file_path}. Total vectors: {index.ntotal}")
        return index
    except Exception as e:
        logger.error(f"Failed to load FAISS index from {file_path}: {e}", exc_info=True)
        return None

def search_faiss_index(index: faiss.Index, query_vectors: np.ndarray, top_k: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Performs a similarity search on the FAISS index.

    Args:
        index (faiss.Index): The FAISS index to search.
        query_vectors (np.ndarray): A 2D NumPy array of query vectors (num_queries, dim).
                                    Should be float32.
        top_k (int): The number of nearest neighbors to retrieve for each query.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: A tuple containing:
            - D (np.ndarray): Distances of the `top_k` neighbors for each query (shape: num_queries, top_k).
            - I (np.ndarray): Indices of the `top_k` neighbors for each query (shape: num_queries, top_k).
            Returns None if an error occurs.
    """
    if not isinstance(query_vectors, np.ndarray) or query_vectors.ndim != 2:
        logger.error("Query vectors must be a 2D NumPy array.")
        return None
    if query_vectors.dtype != np.float32:
        logger.warning(f"FAISS typically expects float32 query vectors. Input dtype is {query_vectors.dtype}. Casting.")
        query_vectors = query_vectors.astype(np.float32)

    if index.d != query_vectors.shape[1]:
        logger.error(f"Query vector dimension ({query_vectors.shape[1]}) does not match index dimension ({index.d}).")
        return None

    try:
        logger.debug(f"Searching FAISS index for {query_vectors.shape[0]} queries, top_k={top_k}.")
        # The search method returns distances (D) and indices (I)
        D, I = index.search(query_vectors, top_k)
        return D, I
    except Exception as e:
        logger.error(f"FAISS index search failed: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example Usage
    dim = 64  # Dimension of vectors
    num_db_vectors = 1000
    num_query_vectors = 5
    k = 4 # Number of nearest neighbors to find

    # Generate some random data
    db_embeddings = np.random.rand(num_db_vectors, dim).astype(np.float32)
    query_embeddings = np.random.rand(num_query_vectors, dim).astype(np.float32)

    print("--- Building FAISS Index (IndexFlatL2) ---")
    index_l2 = build_faiss_index(db_embeddings, index_type="IndexFlatL2")

    if index_l2:
        print(f"Index L2 built. Is trained: {index_l2.is_trained}, Total vectors: {index_l2.ntotal}")

        print("\n--- Searching FAISS Index (L2) ---")
        distances, indices = search_faiss_index(index_l2, query_embeddings, top_k=k)
        if distances is not None and indices is not None:
            for i in range(num_query_vectors):
                print(f"Query {i}:")
                for j in range(k):
                    print(f"  Neighbor {j+1}: Index={indices[i][j]}, Distance={distances[i][j]:.4f}")

        # Test save and load
        index_file_path = Path("temp_faiss_index.bin")
        print(f"\n--- Saving FAISS Index to {index_file_path} ---")
        save_faiss_index(index_l2, index_file_path)

        print(f"\n--- Loading FAISS Index from {index_file_path} ---")
        loaded_index = load_faiss_index(index_file_path)
        if loaded_index:
            print(f"Loaded index. Is trained: {loaded_index.is_trained}, Total vectors: {loaded_index.ntotal}")
            distances_loaded, indices_loaded = search_faiss_index(loaded_index, query_embeddings, top_k=k)
            assert np.array_equal(indices, indices_loaded), "Indices from loaded index do not match original!"
            assert np.allclose(distances, distances_loaded), "Distances from loaded index do not match original!"
            print("Search results from loaded index match original search results.")

        if index_file_path.exists():
            os.remove(index_file_path)
            print(f"Cleaned up temporary index file: {index_file_path}")

    # Example with Inner Product (requires normalized vectors for cosine similarity)
    print("\n--- Building FAISS Index (IndexFlatIP for Cosine Similarity) ---")
    # Normalize vectors for cosine similarity with IndexFlatIP
    faiss.normalize_L2(db_embeddings) # Normalize in-place
    query_embeddings_norm = query_embeddings.copy()
    faiss.normalize_L2(query_embeddings_norm)

    index_ip = build_faiss_index(db_embeddings, index_type="IndexFlatIP") # db_embeddings is now normalized
    if index_ip:
        print(f"Index IP built. Is trained: {index_ip.is_trained}, Total vectors: {index_ip.ntotal}")
        print("\n--- Searching FAISS Index (IP/Cosine) ---")
        # Search returns inner products. For normalized vectors, IP = cosine similarity. Higher is better.
        similarities, indices_ip = search_faiss_index(index_ip, query_embeddings_norm, top_k=k)
        if similarities is not None and indices_ip is not None:
            for i in range(num_query_vectors):
                print(f"Query {i}:")
                for j in range(k):
                    print(f"  Neighbor {j+1}: Index={indices_ip[i][j]}, Similarity={similarities[i][j]:.4f}")

    # Note: IndexIVFFlat example (requires training)
    # print("\n--- Building FAISS Index (IndexIVFFlat - requires training) ---")
    # index_ivf = build_faiss_index(db_embeddings, index_type="IndexIVFFlat")
    # if index_ivf:
    #     print(f"Index IVF built. Is trained: {index_ivf.is_trained}, Total vectors: {index_ivf.ntotal}")
    #     index_ivf.nprobe = 10 # Set nprobe for searching IVF indexes
    #     distances_ivf, indices_ivf = search_faiss_index(index_ivf, query_embeddings, top_k=k)
    #     if distances_ivf is not None:
    #         print(f"Search results from IVF index (first query): Indices={indices_ivf[0]}, Distances={distances_ivf[0]}")

    print("\nFAISS Utils Example Done.")
