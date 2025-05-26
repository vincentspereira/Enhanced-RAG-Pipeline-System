from typing import List, Dict, Any
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import logging
import nltk

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    logging.warning(f"Could not download NLTK data: {e}")

logger = logging.getLogger(__name__)

class TextSearchHelper:
    def __init__(self):
        """Initialize text search helper with NLTK components."""
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception as e:
            logger.warning(f"Could not load stopwords: {e}. Using empty set.")
            self.stop_words = set()

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for search."""
        # Convert to lowercase and remove special characters
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens

    def calculate_tf(self, text: str) -> Dict[str, float]:
        """Calculate term frequency for text."""
        tokens = self.preprocess_text(text)
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        return {
            token: count / total_tokens
            for token, count in token_counts.items()
        }

    def calculate_idf(self, documents: List[str]) -> Dict[str, float]:
        """Calculate inverse document frequency for terms."""
        document_count = len(documents)
        term_doc_count = Counter()
        
        for doc in documents:
            # Count each term only once per document
            unique_terms = set(self.preprocess_text(doc))
            for term in unique_terms:
                term_doc_count[term] += 1
        
        return {
            term: math.log(document_count / (count + 1)) + 1
            for term, count in term_doc_count.items()
        }

    def calculate_tfidf(self, query: str, documents: List[str]) -> List[Dict[str, float]]:
        """Calculate TF-IDF scores for documents."""
        # Calculate IDF for all documents
        idf_scores = self.calculate_idf(documents)
        
        # Calculate TF-IDF for each document
        tfidf_scores = []
        for doc in documents:
            tf_scores = self.calculate_tf(doc)
            doc_tfidf = {
                term: tf * idf_scores.get(term, 0)
                for term, tf in tf_scores.items()
            }
            tfidf_scores.append(doc_tfidf)
        
        return tfidf_scores

    def calculate_keyword_similarity(self, query: str, document: str) -> float:
        """Calculate keyword-based similarity between query and document."""
        query_tokens = set(self.preprocess_text(query))
        doc_tokens = set(self.preprocess_text(document))
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_tokens & doc_tokens)
        union = len(query_tokens | doc_tokens)
        
        return intersection / union if union > 0 else 0.0

    def rank_documents(self, query: str, documents: List[str]) -> List[float]:
        """Rank documents based on keyword similarity."""
        scores = []
        for doc in documents:
            score = self.calculate_keyword_similarity(query, doc)
            scores.append(score)
        return scores

    def combine_scores(self, 
                      semantic_scores: List[float], 
                      keyword_scores: List[float], 
                      alpha: float = 0.7) -> List[float]:
        """Combine semantic and keyword-based scores."""
        if len(semantic_scores) != len(keyword_scores):
            raise ValueError("Score lists must have the same length")
        
        # Normalize scores to [0, 1] range
        max_semantic = max(semantic_scores) if semantic_scores else 1
        max_keyword = max(keyword_scores) if keyword_scores else 1
        
        normalized_semantic = [s / max_semantic for s in semantic_scores]
        normalized_keyword = [k / max_keyword for k in keyword_scores]
        
        # Combine scores with weighted average
        combined_scores = [
            alpha * s + (1 - alpha) * k
            for s, k in zip(normalized_semantic, normalized_keyword)
        ]
        
        return combined_scores
