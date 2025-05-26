from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from dataclasses import dataclass
import logging

@dataclass
class Category:
    """Represents a document category."""
    id: str
    name: str
    description: str
    keywords: List[str]
    parent_id: Optional[str] = None

@dataclass
class DocumentCategory:
    """Represents a document's categorization result."""
    document_id: str
    category_id: str
    confidence: float
    subcategories: List['DocumentCategory']

class DocumentCategorizer:
    """Automated document categorization system."""
    
    def __init__(self, embedding_dimension: int):
        self.categories: Dict[str, Category] = {}
        self.category_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_dimension = embedding_dimension
        self.logger = logging.getLogger(__name__)
        
        # Initialize clustering model
        self.clustering = KMeans(
            n_clusters=10,  # Will be adjusted based on data
            n_init='auto'
        )
    
    def add_category(self, category: Category):
        """Add a new category to the system."""
        self.categories[category.id] = category
        self.logger.info(f"Added category: {category.name}")
    
    def update_category_embeddings(self, 
                                 category_texts: Dict[str, List[str]], 
                                 get_embedding_fn):
        """Update category embeddings using example texts."""
        for category_id, texts in category_texts.items():
            if category_id not in self.categories:
                continue
                
            # Get embeddings for all texts
            embeddings = [get_embedding_fn(text) for text in texts]
            
            # Use mean embedding as category center
            self.category_embeddings[category_id] = np.mean(embeddings, axis=0)
        
        self.logger.info("Updated category embeddings")
    
    def categorize_document(self, 
                          document_id: str,
                          document_embedding: np.ndarray,
                          min_confidence: float = 0.5) -> Optional[DocumentCategory]:
        """Categorize a document based on its embedding."""
        if not self.categories or not self.category_embeddings:
            return None
        
        # Calculate similarity to each category
        similarities = {}
        document_embedding = normalize(document_embedding.reshape(1, -1))
        
        for category_id, category_embedding in self.category_embeddings.items():
            category_embedding = normalize(category_embedding.reshape(1, -1))
            similarity = np.dot(document_embedding, category_embedding.T)[0][0]
            similarities[category_id] = similarity
        
        # Get best matching category
        best_category_id = max(similarities.items(), key=lambda x: x[1])
        
        if similarities[best_category_id] < min_confidence:
            return None
        
        # Get subcategories
        subcategories = []
        for category in self.categories.values():
            if category.parent_id == best_category_id:
                sub_similarity = similarities.get(category.id, 0)
                if sub_similarity >= min_confidence:
                    subcategories.append(DocumentCategory(
                        document_id=document_id,
                        category_id=category.id,
                        confidence=sub_similarity,
                        subcategories=[]
                    ))
        
        return DocumentCategory(
            document_id=document_id,
            category_id=best_category_id,
            confidence=similarities[best_category_id],
            subcategories=subcategories
        )
    
    def discover_categories(self, 
                          document_embeddings: List[np.ndarray],
                          document_texts: List[str],
                          n_clusters: Optional[int] = None):
        """Discover categories automatically using clustering."""
        if not document_embeddings:
            return
        
        # Normalize embeddings
        X = normalize(np.array(document_embeddings))
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            n_clusters = min(10, len(document_embeddings) // 5)
        
        # Update clustering model
        self.clustering = KMeans(n_clusters=n_clusters, n_init='auto')
        
        # Fit clustering model
        self.clustering.fit(X)
        
        # Create categories from clusters
        cluster_texts = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(self.clustering.labels_):
            cluster_texts[label].append(document_texts[idx])
        
        # Create categories
        for cluster_id, texts in cluster_texts.items():
            # Use most common words as keywords
            keywords = self._extract_keywords(texts)
            
            category = Category(
                id=f"cluster_{cluster_id}",
                name=f"Category {cluster_id + 1}",
                description=f"Automatically discovered category {cluster_id + 1}",
                keywords=keywords
            )
            self.add_category(category)
            
            # Use cluster center as category embedding
            self.category_embeddings[category.id] = self.clustering.cluster_centers_[cluster_id]
        
        self.logger.info(f"Discovered {n_clusters} categories through clustering")
    
    def _extract_keywords(self, texts: List[str], top_k: int = 5) -> List[str]:
        """Extract keywords from a list of texts using simple frequency analysis."""
        # This is a simple implementation - could be enhanced with TF-IDF, etc.
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top-k most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_k]]
