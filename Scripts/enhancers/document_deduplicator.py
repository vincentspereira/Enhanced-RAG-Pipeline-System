from typing import List, Dict, Any, Optional, Set, Tuple
import hashlib
from dataclasses import dataclass
import logging
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
from pathlib import Path
import json
import sqlite3
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DeduplicationConfig:
    min_similarity_threshold: float = 0.8
    use_content_hash: bool = True
    use_similarity_check: bool = True
    cache_vectors: bool = True
    batch_size: int = 1000
    store_duplicates_log: bool = True
    min_content_length: int = 50
    use_database: bool = True
    database_path: str = "deduplication_cache.db"

class DocumentDeduplicator:
    def __init__(self, config: DeduplicationConfig = None):
        self.config = config or DeduplicationConfig()
        self.vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            analyzer='word',
            stop_words='english'
        )
        
        if self.config.use_database:
            self._init_database()

    def _init_database(self):
        """Initialize SQLite database for caching document hashes and vectors"""
        try:
            self.conn = sqlite3.connect(self.config.database_path)
            cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_hashes (
                    doc_id TEXT PRIMARY KEY,
                    content_hash TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_vectors (
                    doc_id TEXT PRIMARY KEY,
                    vector BLOB,
                    created_at TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS duplicate_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_doc_id TEXT,
                    duplicate_doc_id TEXT,
                    similarity_score REAL,
                    detection_method TEXT,
                    detected_at TIMESTAMP
                )
            """)
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.config.use_database = False

    def compute_content_hash(self, content: str) -> str:
        """Compute a stable hash of the document content"""
        # Normalize content before hashing
        normalized_content = (
            content.lower()
            .replace('\r\n', '\n')
            .replace('\r', '\n')
            .strip()
        )
        return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()

    def compute_similarity(self, doc1: str, doc2: str) -> float:
        """Compute similarity between two documents"""
        if not doc1 or not doc2:
            return 0.0
            
        try:
            # For very short texts, use sequence matcher
            if len(doc1) < 1000 and len(doc2) < 1000:
                return SequenceMatcher(None, doc1, doc2).ratio()
            
            # For longer texts, use TF-IDF cosine similarity
            tfidf_matrix = self.vectorizer.fit_transform([doc1, doc2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def find_duplicates(self, documents: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Find duplicates in a list of documents
        Args:
            documents: List of tuples (doc_id, content)
        Returns:
            List of duplicate groups with similarity scores
        """
        if not documents:
            return []

        duplicates = []
        doc_hashes = {}
        doc_vectors = {}

        # First pass: compute hashes and identify exact duplicates
        for doc_id, content in documents:
            if len(content) < self.config.min_content_length:
                continue

            if self.config.use_content_hash:
                content_hash = self.compute_content_hash(content)
                if content_hash in doc_hashes:
                    duplicates.append({
                        'original_id': doc_hashes[content_hash],
                        'duplicate_id': doc_id,
                        'similarity': 1.0,
                        'method': 'exact_hash'
                    })
                    continue
                doc_hashes[content_hash] = doc_id

            # Compute and store document vector
            if self.config.use_similarity_check:
                try:
                    vector = self.vectorizer.fit_transform([content])
                    doc_vectors[doc_id] = vector
                except Exception as e:
                    logger.error(f"Error computing vector for {doc_id}: {e}")

        # Second pass: find similar documents
        if self.config.use_similarity_check and len(doc_vectors) > 1:
            doc_ids = list(doc_vectors.keys())
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    try:
                        similarity = cosine_similarity(
                            doc_vectors[doc_ids[i]], 
                            doc_vectors[doc_ids[j]]
                        )[0][0]
                        
                        if similarity >= self.config.min_similarity_threshold:
                            duplicates.append({
                                'original_id': doc_ids[i],
                                'duplicate_id': doc_ids[j],
                                'similarity': float(similarity),
                                'method': 'similarity'
                            })
                    except Exception as e:
                        logger.error(f"Error comparing documents {doc_ids[i]} and {doc_ids[j]}: {e}")

        # Store duplicate information if configured
        if self.config.store_duplicates_log and self.config.use_database:
            self._store_duplicates(duplicates)

        return duplicates

    def _store_duplicates(self, duplicates: List[Dict[str, Any]]):
        """Store duplicate detection results in the database"""
        if not self.config.use_database:
            return

        try:
            cursor = self.conn.cursor()
            for dup in duplicates:
                cursor.execute("""
                    INSERT INTO duplicate_logs 
                    (original_doc_id, duplicate_doc_id, similarity_score, 
                     detection_method, detected_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    dup['original_id'],
                    dup['duplicate_id'],
                    dup['similarity'],
                    dup['method'],
                    datetime.now().isoformat()
                ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to store duplicates log: {e}")

    def batch_process(self, documents: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Process documents in batches to find duplicates"""
        all_duplicates = []
        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i:i + self.config.batch_size]
            duplicates = self.find_duplicates(batch)
            all_duplicates.extend(duplicates)
        return all_duplicates

    def get_duplicate_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected duplicates"""
        if not self.config.use_database:
            return {}

        try:
            cursor = self.conn.cursor()
            
            # Get total number of duplicates
            cursor.execute("SELECT COUNT(*) FROM duplicate_logs")
            total_duplicates = cursor.fetchone()[0]
            
            # Get average similarity score
            cursor.execute("SELECT AVG(similarity_score) FROM duplicate_logs")
            avg_similarity = cursor.fetchone()[0]
            
            # Get detection method distribution
            cursor.execute("""
                SELECT detection_method, COUNT(*) 
                FROM duplicate_logs 
                GROUP BY detection_method
            """)
            method_distribution = dict(cursor.fetchall())
            
            # Get recent duplicates
            cursor.execute("""
                SELECT COUNT(*) 
                FROM duplicate_logs 
                WHERE detected_at >= datetime('now', '-1 day')
            """)
            recent_duplicates = cursor.fetchone()[0]
            
            return {
                "total_duplicates": total_duplicates,
                "average_similarity": avg_similarity,
                "method_distribution": method_distribution,
                "recent_duplicates": recent_duplicates
            }
        except Exception as e:
            logger.error(f"Error getting duplicate statistics: {e}")
            return {}

    def clear_cache(self):
        """Clear the database cache"""
        if not self.config.use_database:
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM document_hashes")
            cursor.execute("DELETE FROM document_vectors")
            cursor.execute("DELETE FROM duplicate_logs")
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'conn'):
            try:
                self.conn.close()
            except:
                pass
