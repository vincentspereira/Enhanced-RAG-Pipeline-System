from typing import List, Dict, Set, Tuple, Optional
import hashlib
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DeduplicationConfig:
    min_similarity_threshold: float = 0.85
    near_duplicate_threshold: float = 0.95
    max_batch_size: int = 1000
    use_content_hash: bool = True
    use_semantic_dedup: bool = True
    max_workers: int = 4
    min_content_length: int = 50
    store_duplicates: bool = True
    ngram_range: Tuple[int, int] = (1, 3)

class DocumentDeduplicator:
    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self.config = config or DeduplicationConfig()
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.config.ngram_range,
            min_df=1,
            strip_accents='unicode'
        )
        self.duplicate_store = {}
        self.hash_cache = {}

    def compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of normalized content"""
        if content in self.hash_cache:
            return self.hash_cache[content]
        
        normalized = self._normalize_content(content)
        content_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        self.hash_cache[content] = content_hash
        return content_hash

    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistent hashing"""
        # Remove whitespace and convert to lowercase
        return ' '.join(content.lower().split())

    def find_duplicates(
        self,
        documents: List[Dict[str, str]],
        content_key: str = 'content',
        id_key: str = 'id'
    ) -> Dict[str, List[str]]:
        """Find duplicate documents in a collection"""
        if not documents:
            return {}

        # Filter out documents that are too short
        valid_docs = [
            doc for doc in documents
            if len(doc.get(content_key, '')) >= self.config.min_content_length
        ]

        if not valid_docs:
            return {}

        # Initialize results
        duplicates = {}
        exact_duplicates = set()

        # First pass: exact hash matching
        if self.config.use_content_hash:
            hash_groups = self._find_hash_duplicates(valid_docs, content_key, id_key)
            duplicates.update(hash_groups)
            exact_duplicates.update(
                doc_id 
                for group in hash_groups.values() 
                for doc_id in group
            )

        # Second pass: semantic similarity for remaining documents
        if self.config.use_semantic_dedup:
            remaining_docs = [
                doc for doc in valid_docs
                if doc[id_key] not in exact_duplicates
            ]
            
            if remaining_docs:
                semantic_groups = self._find_semantic_duplicates(
                    remaining_docs,
                    content_key,
                    id_key
                )
                duplicates.update(semantic_groups)

        # Store duplicates if configured
        if self.config.store_duplicates:
            self._store_duplicates(duplicates)

        return duplicates

    def _find_hash_duplicates(
        self,
        documents: List[Dict[str, str]],
        content_key: str,
        id_key: str
    ) -> Dict[str, List[str]]:
        """Find exact duplicates using content hashing"""
        hash_map = {}
        
        def process_doc(doc):
            content = doc.get(content_key, '')
            doc_id = doc.get(id_key)
            content_hash = self.compute_content_hash(content)
            return content_hash, doc_id

        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            results = list(executor.map(process_doc, documents))

        # Group by hash
        for content_hash, doc_id in results:
            if content_hash not in hash_map:
                hash_map[content_hash] = []
            hash_map[content_hash].append(doc_id)

        # Filter groups with duplicates
        return {
            hash_val: doc_ids
            for hash_val, doc_ids in hash_map.items()
            if len(doc_ids) > 1
        }

    def _find_semantic_duplicates(
        self,
        documents: List[Dict[str, str]],
        content_key: str,
        id_key: str
    ) -> Dict[str, List[str]]:
        """Find near-duplicates using semantic similarity"""
        contents = [doc.get(content_key, '') for doc in documents]
        doc_ids = [doc.get(id_key) for doc in documents]

        # Process in batches to handle large collections
        duplicates = {}
        for i in range(0, len(contents), self.config.max_batch_size):
            batch_contents = contents[i:i + self.config.max_batch_size]
            batch_ids = doc_ids[i:i + self.config.max_batch_size]
            
            # Compute TF-IDF vectors
            try:
                vectors = self.vectorizer.fit_transform(batch_contents)
            except Exception as e:
                logger.error(f"Error computing TF-IDF vectors: {e}")
                continue

            # Compute similarity matrix
            similarity_matrix = cosine_similarity(vectors)

            # Find similar documents
            for idx1 in range(len(batch_contents)):
                similar_docs = []
                for idx2 in range(idx1 + 1, len(batch_contents)):
                    similarity = similarity_matrix[idx1, idx2]
                    if similarity >= self.config.min_similarity_threshold:
                        similar_docs.append({
                            'id': batch_ids[idx2],
                            'similarity': float(similarity)
                        })

                if similar_docs:
                    doc_id = batch_ids[idx1]
                    duplicates[doc_id] = similar_docs

        return duplicates

    def _store_duplicates(self, duplicates: Dict[str, List[str]]):
        """Store duplicate information for later reference"""
        timestamp = datetime.now().isoformat()
        
        for primary_id, duplicate_info in duplicates.items():
            if primary_id not in self.duplicate_store:
                self.duplicate_store[primary_id] = []
            
            entry = {
                'timestamp': timestamp,
                'duplicates': duplicate_info
            }
            self.duplicate_store[primary_id].append(entry)

    def get_duplicate_history(self, document_id: str) -> List[Dict]:
        """Get the duplicate detection history for a document"""
        return self.duplicate_store.get(document_id, [])

    def export_duplicate_store(self, file_path: str):
        """Export duplicate store to a JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.duplicate_store, f, indent=2)

    def import_duplicate_store(self, file_path: str):
        """Import duplicate store from a JSON file"""
        with open(file_path, 'r') as f:
            self.duplicate_store = json.load(f)

    def clear_duplicate_store(self):
        """Clear the duplicate store"""
        self.duplicate_store = {}
        self.hash_cache = {}
