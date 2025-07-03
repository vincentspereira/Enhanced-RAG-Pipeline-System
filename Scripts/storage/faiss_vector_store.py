import logging
from typing import List, Dict, Any, Optional, Union
import uuid
import numpy as np
import faiss
import os
import json
import pickle # For saving/loading Python objects like metadata list

from .vector_store_base import VectorStoreBase, DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

class FaissVectorStore(VectorStoreBase):
    def __init__(self, index_file_path: str, metadata_file_path: str, create_if_not_exists: bool = True):
        self.index_file_path = index_file_path
        self.metadata_file_path = metadata_file_path # To store texts and metadatas
        self.create_if_not_exists = create_if_not_exists

        self.index: Optional[faiss.Index] = None
        self.doc_id_to_index_pos: Dict[Union[str, int, uuid.UUID], int] = {} # Maps our doc ID to FAISS index position
        self.index_pos_to_doc_id: Dict[int, Union[str, int, uuid.UUID]] = {}
        self.document_texts: List[str] = [] # Stores text content, parallel to FAISS index
        self.document_metadatas: List[Dict[str, Any]] = [] # Stores metadata, parallel to FAISS index

        self.vector_size: Optional[int] = None
        self.distance_metric: Optional[str] = None # FAISS index type implies metric

        self._initialize_store()

    def _initialize_store(self):
        if os.path.exists(self.index_file_path) and os.path.exists(self.metadata_file_path):
            try:
                self.index = faiss.read_index(self.index_file_path)
                with open(self.metadata_file_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.doc_id_to_index_pos = saved_data.get('doc_id_to_index_pos', {})
                    self.index_pos_to_doc_id = saved_data.get('index_pos_to_doc_id', {})
                    self.document_texts = saved_data.get('document_texts', [])
                    self.document_metadatas = saved_data.get('document_metadatas', [])
                    self.vector_size = saved_data.get('vector_size')
                    self.distance_metric = saved_data.get('distance_metric') # Store the metric used at creation

                if self.index and self.vector_size:
                    logger.info(f"FAISS index loaded from {self.index_file_path} with {self.index.ntotal} vectors.")
                    logger.info(f"Metadata loaded from {self.metadata_file_path}.")
                else:
                    logger.warning(f"Loaded FAISS index from {self.index_file_path}, but it's invalid or vector_size missing. Re-initializing if allowed.")
                    if self.create_if_not_exists:
                        self._create_new_index_placeholder() # Create placeholder, actual size set in initialize()
                    else:
                        raise FileNotFoundError("FAISS index loaded but seems invalid and create_if_not_exists is False.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index/metadata: {e}. Re-initializing if allowed.", exc_info=True)
                if self.create_if_not_exists:
                    self._create_new_index_placeholder()
                else:
                    raise # Re-raise if not allowed to create
        elif self.create_if_not_exists:
            self._create_new_index_placeholder()
            logger.info("No existing FAISS index found. New placeholder index ready (actual initialization via initialize method).")
        else:
            raise FileNotFoundError(f"FAISS index file '{self.index_file_path}' or metadata file '{self.metadata_file_path}' not found and create_if_not_exists is False.")

    def _create_new_index_placeholder(self):
        """Creates a placeholder for the index; actual type/size set in initialize()."""
        self.index = None # Will be created in initialize()
        self.doc_id_to_index_pos = {}
        self.index_pos_to_doc_id = {}
        self.document_texts = []
        self.document_metadatas = []
        self.vector_size = None
        self.distance_metric = None


    def _save_store(self):
        if self.index is None:
            logger.warning("Attempted to save FAISS store, but index is None.")
            return
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.index_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.metadata_file_path), exist_ok=True)

            faiss.write_index(self.index, self.index_file_path)
            metadata_to_save = {
                'doc_id_to_index_pos': self.doc_id_to_index_pos,
                'index_pos_to_doc_id': self.index_pos_to_doc_id,
                'document_texts': self.document_texts,
                'document_metadatas': self.document_metadatas,
                'vector_size': self.vector_size,
                'distance_metric': self.distance_metric
            }
            with open(self.metadata_file_path, 'wb') as f:
                pickle.dump(metadata_to_save, f)
            logger.info(f"FAISS index saved to {self.index_file_path} and metadata to {self.metadata_file_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index/metadata: {e}", exc_info=True)


    async def initialize(self, collection_name: str, vector_size: int, distance_metric: str = "L2", **kwargs):
        # FAISS doesn't have named collections in the same way. The "collection" is the index file.
        # We use collection_name here for compatibility, but it primarily influences file paths if not set directly.
        # If index_file_path was generic, collection_name could be used to make it specific.
        # For this implementation, index_file_path is primary.

        if self.index is not None and self.index.d == vector_size:
            # Metric check is harder as FAISS index type implies metric.
            # For simplicity, if index exists and dimensions match, assume it's usable.
            logger.info(f"FAISS index already initialized with dimension {self.index.d}. Using existing.")
            self.vector_size = self.index.d # Ensure it's set
            # self.distance_metric might have been loaded
            return

        self.vector_size = vector_size
        self.distance_metric = distance_metric.upper() # Store metric name

        # FAISS distance metric mapping (faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT)
        # For Cosine similarity with FAISS, vectors should be normalized, and METRIC_INNER_PRODUCT used.
        # Or, use IndexFlatIP for dot product, IndexFlatL2 for Euclidean.

        index_type_str = kwargs.get("faiss_index_type", "IndexFlatL2") # e.g., "IndexFlatL2", "IndexFlatIP", "IndexHNSWFlat"

        try:
            if index_type_str == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(vector_size)
                self.distance_metric = "L2" # Override if specific index chosen
            elif index_type_str == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(vector_size)
                self.distance_metric = "IP" # Override
            elif index_type_str == "IndexHNSWFlat": # Example of a more complex index
                hnsw_m = kwargs.get("hnsw_m", 32) # Number of connections for HNSW
                # HNSW requires a base quantizer/index, e.g. IndexFlatL2 or IndexFlatIP
                if self.distance_metric == "IP":
                    base_index = faiss.IndexFlatIP(vector_size)
                else: # Default to L2 for HNSW if not IP
                    base_index = faiss.IndexFlatL2(vector_size)
                    self.distance_metric = "L2"
                self.index = faiss.IndexHNSWFlat(base_index, hnsw_m)
                # self.index.hnsw.efSearch = kwargs.get("hnsw_ef_search", 64) # Search time parameter
                # self.index.hnsw.efConstruction = kwargs.get("hnsw_ef_construction", 128) # Build time parameter
            else:
                logger.warning(f"Unsupported faiss_index_type '{index_type_str}'. Defaulting to IndexFlatL2.")
                self.index = faiss.IndexFlatL2(vector_size)
                self.distance_metric = "L2"

            # Reset data structures for new index
            self.doc_id_to_index_pos = {}
            self.index_pos_to_doc_id = {}
            self.document_texts = []
            self.document_metadatas = []

            logger.info(f"FAISS index '{index_type_str}' initialized for collection '{collection_name}' (path: {self.index_file_path}) with vector size {vector_size}, distance metric implied by index type (using: {self.distance_metric}).")
            self._save_store() # Save empty initialized store
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index for '{collection_name}': {e}", exc_info=True)
            self.index = None


    async def add_documents(self, collection_name: str, documents: List[DocumentChunk], **kwargs) -> List[Union[str, int, uuid.UUID]]:
        if self.index is None:
            logger.error("FAISS index not initialized. Cannot add documents.")
            # Try to initialize if vector_size is available from first doc and create_if_not_exists was true
            if self.create_if_not_exists and documents and documents[0].vector:
                logger.info("Attempting to initialize FAISS index on first add...")
                await self.initialize(collection_name, len(documents[0].vector))
                if self.index is None: return [] # Failed to initialize
            else:
                return []

        if not documents: return []

        vectors_to_add = []
        new_doc_ids = []

        for doc in documents:
            if len(doc.vector) != self.vector_size:
                logger.error(f"Vector dimension mismatch for doc ID {doc.id}. Expected {self.vector_size}, got {len(doc.vector)}. Skipping.")
                continue

            doc_id_str = str(doc.id)
            if doc_id_str in self.doc_id_to_index_pos:
                # Update existing document (FAISS requires remove then add for updates)
                # This is complex and can fragment the index. Simpler to just log or disallow updates for now.
                # For this basic version, we'll assume new IDs or allow overwriting if FAISS handles it (IndexIDMap).
                # If using IndexIDMap, `add_with_ids` can update if ID exists.
                # Standard FAISS indexes just append. We need to manage IDs externally.
                logger.warning(f"Document ID {doc_id_str} already exists. FAISS update logic not fully implemented in this basic version. Re-adding might lead to duplicates if not careful with ID management.")
                # To implement update: remove old vector (if possible, needs direct ID mapping), then add new.
                # Or, use IndexIDMap. For now, we'll just add, which means new entries for same logical doc if called again.

            vectors_to_add.append(doc.vector)

            # Store text and metadata corresponding to the order of vectors added
            current_index_pos = self.index.ntotal + (len(vectors_to_add) - 1) # Tentative position
            self.doc_id_to_index_pos[doc_id_str] = current_index_pos
            self.index_pos_to_doc_id[current_index_pos] = doc_id_str
            self.document_texts.append(doc.text)
            self.document_metadatas.append(doc.metadata or {})
            new_doc_ids.append(doc.id)

        if not vectors_to_add:
            return []

        try:
            vectors_np = np.array(vectors_to_add, dtype=np.float32)
            # Normalize vectors if using Inner Product for Cosine similarity
            if self.distance_metric == "IP" or (isinstance(self.index, faiss.IndexHNSWFlat) and self.index.metric_type == faiss.METRIC_INNER_PRODUCT):
                 faiss.normalize_L2(vectors_np)

            self.index.add(vectors_np)
            self._save_store() # Save after adding
            logger.info(f"Successfully added {len(vectors_np)} vectors to FAISS index. Total vectors: {self.index.ntotal}")
            return new_doc_ids
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS index: {e}", exc_info=True)
            # Rollback metadata additions for this failed batch
            # This is simplified; a robust rollback would be more complex.
            num_added_this_batch = len(vectors_to_add)
            self.document_texts = self.document_texts[:-num_added_this_batch]
            self.document_metadatas = self.document_metadatas[:-num_added_this_batch]
            for i in range(num_added_this_batch):
                pos_to_remove = self.index.ntotal -1 + num_added_this_batch - i # theoretical position before add
                doc_id_to_remove = self.index_pos_to_doc_id.pop(pos_to_remove, None)
                if doc_id_to_remove:
                    self.doc_id_to_index_pos.pop(doc_id_to_remove, None)
            return []


    async def search(self, collection_name: str, query_vector: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None, **kwargs) -> List[SearchResult]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index not initialized or is empty. Cannot perform search.")
            return []

        query_vector_np = np.array([query_vector], dtype=np.float32)
        # Normalize query vector if index uses Inner Product for Cosine similarity
        if self.distance_metric == "IP" or (isinstance(self.index, faiss.IndexHNSWFlat) and self.index.metric_type == faiss.METRIC_INNER_PRODUCT):
            faiss.normalize_L2(query_vector_np)

        try:
            # FAISS search returns distances (D) and indices (I)
            distances, indices = self.index.search(query_vector_np, top_k)

            results = []
            if len(indices[0]) > 0:
                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    if idx == -1: continue # -1 means no vector found for that position (e.g. if k > ntotal)

                    doc_id = self.index_pos_to_doc_id.get(idx)
                    if doc_id is None:
                        logger.warning(f"No document ID found for FAISS index position {idx}. Skipping.")
                        continue

                    text = self.document_texts[idx] if idx < len(self.document_texts) else None
                    metadata = self.document_metadatas[idx] if idx < len(self.document_metadatas) else {}

                    # Apply filters post-search (FAISS core doesn't support metadata filtering during search easily)
                    # This is inefficient for large result sets but a common pattern for basic FAISS.
                    # More advanced FAISS usage might involve IndexIDMap2 or custom filtering logic if IDs map to metadata.
                    if filters:
                        match = True
                        for key, value in filters.items():
                            # Assuming filters are on metadata. This is a very basic equality check.
                            # Example: key="source", value="news_A"
                            meta_path = key.split('.') # e.g. "category" or "details.year"
                            current_val = metadata
                            try:
                                for p_key in meta_path:
                                    current_val = current_val[p_key]
                                if current_val != value:
                                    match = False
                                    break
                            except (KeyError, TypeError): # Key not found or metadata not dict
                                match = False
                                break
                        if not match:
                            continue

                    # Score: FAISS returns distances. For L2, lower is better. For IP (cosine on normalized), higher is better.
                    score = float(distances[0][i])
                    if self.distance_metric == "IP": # Higher is better
                        final_score = score
                    else: # L2, lower is better. Convert to a "similarity" where higher is better (e.g. 1 / (1 + dist))
                        final_score = 1.0 / (1.0 + score) if score >= 0 else 0.0


                    results.append(SearchResult(
                        id=doc_id,
                        score=final_score,
                        payload={"text": text, "metadata": metadata}
                        # vector=self.index.reconstruct(idx).tolist() # If needed, but slow
                    ))
                    if len(results) >= top_k: # Ensure we don't exceed top_k after filtering
                        break
            return results
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}", exc_info=True)
            return []

    async def delete_documents(self, collection_name: str, document_ids: List[Union[str, int, uuid.UUID]], **kwargs) -> bool:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index not initialized or empty. Cannot delete.")
            return False

        # FAISS `remove_ids` requires an IDSelector.
        # This is non-trivial if we don't use IndexIDMap.
        # Standard FAISS indexes do not support efficient deletion of arbitrary vectors by their content ID.
        # Deletion often means rebuilding the index without the removed vectors.
        # For IndexFlat, one might be able to reconstruct a new index.
        # For HNSW or IVF, it's more complex.
        # This basic implementation will NOT support efficient deletion.
        logger.warning("FAISS deletion is complex and not fully supported in this basic version without IndexIDMap or rebuilding. Logical deletion can be implemented by marking metadata.")

        # Placeholder: if you were to implement logical deletion
        # num_deleted = 0
        # for doc_id_to_delete in document_ids:
        #     str_doc_id = str(doc_id_to_delete)
        #     if str_doc_id in self.doc_id_to_index_pos:
        #         idx_pos = self.doc_id_to_index_pos[str_doc_id]
        #         if 0 <= idx_pos < len(self.document_metadatas):
        #             self.document_metadatas[idx_pos]["_deleted"] = True # Mark as deleted
        #             num_deleted +=1
        # if num_deleted > 0: self._save_store()
        # return num_deleted > 0
        return False # Indicate not fully supported


    async def get_collection_info(self, collection_name: str, **kwargs) -> Dict[str, Any]:
        if self.index is None:
            return {"name": collection_name, "status": "not_initialized", "vectors_count": 0}
        return {
            "name": collection_name, # FAISS doesn't use collection names, this is for compatibility
            "path": self.index_file_path,
            "status": "initialized",
            "vectors_count": self.index.ntotal,
            "vector_dimension": self.index.d,
            "distance_metric_type": self.distance_metric, # The metric we aimed for
            "faiss_index_type": self.index.__class__.__name__,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True, # IndexFlat is always "trained"
        }

    async def close(self):
        # FAISS index objects are in memory. Saving is explicit via _save_store().
        # No explicit close method for the index object itself.
        logger.info("FAISS store is in-memory; ensure `_save_store()` is called for persistence if changes made.")
        self.index = None # Allow GC if no other refs

    async def health_check(self) -> bool:
        # For a file-based FAISS, health means the index is loaded and usable.
        return self.index is not None and self.vector_size is not None

# Example Usage (Conceptual)
if __name__ == '__main__':
    async def main():
        logging.basicConfig(level=logging.INFO)

        INDEX_PATH = "./faiss_test_index.bin"
        META_PATH = "./faiss_test_metadata.pkl"

        # Clean up previous test files
        if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
        if os.path.exists(META_PATH): os.remove(META_PATH)

        faiss_store = FaissVectorStore(index_file_path=INDEX_PATH, metadata_file_path=META_PATH)

        collection_name = "my_faiss_docs" # Used conceptually
        vector_dim = 4 # Example dimension

        await faiss_store.initialize(collection_name, vector_dim, distance_metric="L2", faiss_index_type="IndexFlatL2")

        info = await faiss_store.get_collection_info(collection_name)
        logger.info(f"Store info: {info}")

        doc_chunks = [
            DocumentChunk(id="faiss_doc1", text="Introduction to FAISS.", vector=[0.1, 0.2, 0.3, 0.4], metadata={"cat": "tech"}),
            DocumentChunk(id="faiss_doc2", text="Efficient similarity search.", vector=[0.5, 0.6, 0.7, 0.8], metadata={"cat": "ml"}),
            DocumentChunk(id="faiss_doc3", text="Another tech document.", vector=[0.15, 0.25, 0.35, 0.45], metadata={"cat": "tech"}),
        ]
        added_ids = await faiss_store.add_documents(collection_name, doc_chunks)
        logger.info(f"Added IDs: {added_ids}")

        info_after = await faiss_store.get_collection_info(collection_name)
        logger.info(f"Store info after add: {info_after}")

        if added_ids:
            query_v = [0.12, 0.22, 0.32, 0.42] # Close to doc1 and doc3
            search_res = await faiss_store.search(collection_name, query_v, top_k=2)
            logger.info(f"Search results for {query_v}:")
            for r in search_res:
                logger.info(f"  ID: {r.id}, Score: {r.score:.4f}, Text: {r.payload['text']}")

            # Test filter (post-search)
            search_res_filtered = await faiss_store.search(collection_name, query_v, top_k=2, filters={"cat": "ml"})
            logger.info(f"Search results for {query_v} (filtered cat=ml):")
            for r in search_res_filtered:
                logger.info(f"  ID: {r.id}, Score: {r.score:.4f}, Text: {r.payload['text']}")


        await faiss_store.close()

        # Test loading
        logger.info("--- Testing loading from file ---")
        faiss_store_loaded = FaissVectorStore(index_file_path=INDEX_PATH, metadata_file_path=META_PATH)
        if await faiss_store_loaded.health_check():
            info_loaded = await faiss_store_loaded.get_collection_info(collection_name)
            logger.info(f"Loaded store info: {info_loaded}")
            if info_loaded.get("vectors_count", 0) > 0:
                 query_v = [0.55, 0.65, 0.75, 0.85] # Close to doc2
                 search_res_loaded = await faiss_store_loaded.search(collection_name, query_v, top_k=1)
                 logger.info(f"Search results from loaded index for {query_v}:")
                 for r in search_res_loaded:
                     logger.info(f"  ID: {r.id}, Score: {r.score:.4f}, Text: {r.payload['text']}")
            await faiss_store_loaded.close()

        # Clean up
        if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
        if os.path.exists(META_PATH): os.remove(META_PATH)

    import asyncio
    asyncio.run(main())
