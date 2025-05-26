from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass
import logging
from collections import defaultdict
import spacy
from scipy.spatial.distance import cosine
import json
import pickle
from datetime import datetime
import os

logger = logging.getLogger(__name__)

@dataclass
class RelevanceFeedbackConfig:
    positive_weight: float = 0.8
    negative_weight: float = 0.2
    min_feedback_samples: int = 3
    max_terms_per_feedback: int = 10
    feedback_history_size: int = 1000
    enable_persistent_learning: bool = True
    storage_path: str = "feedback_models"
    confidence_threshold: float = 0.6
    use_gpu: bool = True
    model_name: str = "all-MiniLM-L6-v2"

class RelevanceFeedback:
    def __init__(self, config: RelevanceFeedbackConfig = None):
        self.config = config or RelevanceFeedbackConfig()
        self._init_components()
        self.feedback_history = []
        self.term_weights = defaultdict(float)
        
        if self.config.enable_persistent_learning:
            self._load_feedback_history()

    def _init_components(self):
        """Initialize NLP components"""
        try:
            self.semantic_model = SentenceTransformer(self.config.model_name)
            if self.config.use_gpu and torch.cuda.is_available():
                self.semantic_model = self.semantic_model.to('cuda')
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            self.semantic_model = None

        try:
            self.nlp = spacy.load('en_core_web_md')
        except:
            logger.warning("Downloading spaCy model...")
            spacy.cli.download('en_core_web_md')
            self.nlp = spacy.load('en_core_web_md')

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            stop_words='english'
        )

    def _load_feedback_history(self):
        """Load feedback history from disk"""
        if not self.config.enable_persistent_learning:
            return

        try:
            os.makedirs(self.config.storage_path, exist_ok=True)
            history_file = os.path.join(self.config.storage_path, "feedback_history.pkl")
            weights_file = os.path.join(self.config.storage_path, "term_weights.json")
            
            if os.path.exists(history_file):
                with open(history_file, 'rb') as f:
                    self.feedback_history = pickle.load(f)
                    
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as f:
                    self.term_weights = defaultdict(float, json.load(f))
        except Exception as e:
            logger.error(f"Failed to load feedback history: {e}")

    def _save_feedback_history(self):
        """Save feedback history to disk"""
        if not self.config.enable_persistent_learning:
            return

        try:
            os.makedirs(self.config.storage_path, exist_ok=True)
            history_file = os.path.join(self.config.storage_path, "feedback_history.pkl")
            weights_file = os.path.join(self.config.storage_path, "term_weights.json")
            
            with open(history_file, 'wb') as f:
                pickle.dump(self.feedback_history[-self.config.feedback_history_size:], f)
                
            with open(weights_file, 'w') as f:
                json.dump(dict(self.term_weights), f)
        except Exception as e:
            logger.error(f"Failed to save feedback history: {e}")

    def process_feedback(
        self,
        query: str,
        positive_docs: List[str],
        negative_docs: List[str]
    ) -> Tuple[Set[str], Dict[str, float]]:
        """Process user feedback and extract relevant terms"""
        if (len(positive_docs) + len(negative_docs) < 
            self.config.min_feedback_samples):
            return set(), {}

        # Record feedback
        feedback_entry = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "positive_docs": positive_docs,
            "negative_docs": negative_docs
        }
        self.feedback_history.append(feedback_entry)
        
        # Extract terms from positive and negative documents
        positive_terms = self._extract_terms(positive_docs)
        negative_terms = self._extract_terms(negative_docs)
        
        # Update term weights
        updated_terms = set()
        term_scores = {}
        
        for term, count in positive_terms.items():
            weight = count * self.config.positive_weight
            self.term_weights[term] += weight
            updated_terms.add(term)
            term_scores[term] = self.term_weights[term]
            
        for term, count in negative_terms.items():
            weight = count * self.config.negative_weight
            self.term_weights[term] -= weight
            updated_terms.add(term)
            term_scores[term] = self.term_weights[term]
        
        # Save updated feedback history
        if self.config.enable_persistent_learning:
            self._save_feedback_history()
        
        return updated_terms, term_scores

    def _extract_terms(self, documents: List[str]) -> Dict[str, int]:
        """Extract important terms from documents"""
        terms = defaultdict(int)
        
        for doc in documents:
            # Process with spaCy
            processed_doc = self.nlp(doc)
            
            # Extract noun phrases and named entities
            for chunk in processed_doc.noun_chunks:
                if not chunk.root.is_stop:
                    terms[chunk.text.lower()] += 1
                    
            for ent in processed_doc.ents:
                terms[ent.text.lower()] += 1
                
            # Extract keywords using TF-IDF
            try:
                tfidf = self.vectorizer.fit_transform([doc])
                feature_names = self.vectorizer.get_feature_names_out()
                for idx, score in zip(
                    tfidf.indices,
                    tfidf.data
                ):
                    term = feature_names[idx]
                    terms[term] += score
            except:
                pass
        
        # Normalize and filter terms
        max_score = max(terms.values()) if terms else 1
        normalized_terms = {
            term: score/max_score 
            for term, score in terms.items()
        }
        
        # Sort and limit number of terms
        return dict(
            sorted(
                normalized_terms.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.config.max_terms_per_feedback]
        )

    def expand_query(self, query: str) -> Tuple[str, float]:
        """Expand query based on feedback history"""
        if not self.feedback_history:
            return query, 0.0

        # Find similar queries from history
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return query, 0.0

        similar_queries = []
        for entry in self.feedback_history:
            hist_query = entry["query"]
            hist_embedding = self._get_embedding(hist_query)
            if hist_embedding is not None:
                similarity = 1 - cosine(query_embedding, hist_embedding)
                if similarity > self.config.confidence_threshold:
                    similar_queries.append((entry, similarity))

        if not similar_queries:
            return query, 0.0

        # Sort by similarity
        similar_queries.sort(key=lambda x: x[1], reverse=True)
        
        # Collect relevant terms from similar queries
        expansion_terms = defaultdict(float)
        for entry, similarity in similar_queries:
            for doc in entry["positive_docs"]:
                terms = self._extract_terms([doc])
                for term, score in terms.items():
                    if self.term_weights[term] > 0:
                        expansion_terms[term] += score * similarity

        # Select top terms for expansion
        if expansion_terms:
            sorted_terms = sorted(
                expansion_terms.items(),
                key=lambda x: x[1] * self.term_weights[x[0]],
                reverse=True
            )
            expansion = " ".join(term for term, _ in sorted_terms[:5])
            confidence = sum(score for _, score in sorted_terms[:5]) / 5
            return f"{query} {expansion}", confidence
        
        return query, 0.0

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get semantic embedding for text"""
        if not self.semantic_model:
            return None

        try:
            with torch.no_grad():
                embedding = self.semantic_model.encode(
                    text,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            return embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about feedback history"""
        if not self.feedback_history:
            return {}

        total_entries = len(self.feedback_history)
        total_positive = sum(
            len(entry["positive_docs"])
            for entry in self.feedback_history
        )
        total_negative = sum(
            len(entry["negative_docs"])
            for entry in self.feedback_history
        )
        
        # Analyze term weights
        positive_terms = len([t for t, w in self.term_weights.items() if w > 0])
        negative_terms = len([t for t, w in self.term_weights.items() if w < 0])
        
        return {
            "total_feedback_entries": total_entries,
            "total_positive_samples": total_positive,
            "total_negative_samples": total_negative,
            "positive_terms": positive_terms,
            "negative_terms": negative_terms,
            "last_feedback": self.feedback_history[-1]["timestamp"]
            if self.feedback_history else None
        }

    def reset_feedback(self):
        """Reset feedback history and term weights"""
        self.feedback_history = []
        self.term_weights = defaultdict(float)
        if self.config.enable_persistent_learning:
            self._save_feedback_history()
