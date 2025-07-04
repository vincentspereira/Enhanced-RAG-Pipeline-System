from typing import List, Dict, Any, Callable, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

@dataclass
class RelevanceFeedback:
    """Structure to store relevance feedback."""
    query_id: str
    document_id: str
    is_relevant: bool
    feedback_source: str  # e.g., "user", "automatic"
    confidence: float

class ActiveLearner:
    """Active learning system for relevance feedback."""
    
    def __init__(self, embedding_dimension: Optional[int] = None): # Made embedding_dimension optional
        self.relevance_history: List[RelevanceFeedback] = []
        self.positive_embeddings: List[np.ndarray] = []
        self.negative_embeddings: List[np.ndarray] = []
        self.embedding_dimension = embedding_dimension
        self.interaction_data: List[Dict[str, Any]] = [] # For continuous learning

    def add_interaction_data(self, query: str, context: str, response: str, domain: Optional[str]=None, feedback_score: Optional[float]=None):
        """Adds interaction data for continuous learning."""
        self.interaction_data.append({
            "query": query,
            "context": context,
            "response": response,
            "domain": domain,
            "feedback_score": feedback_score, # Could be explicit (e.g., thumbs up/down) or implicit
            "timestamp": "" # Add timestamp, e.g. datetime.now().isoformat()
        })

    def get_training_data_for_finetuning(self, min_samples: int = 100) -> Optional[List[Dict[str,Any]]]:
        """Retrieves collected interaction data suitable for fine-tuning."""
        if len(self.interaction_data) >= min_samples:
            # Potentially filter or process data further
            return self.interaction_data
        return None

    def add_feedback(self, feedback: RelevanceFeedback, embedding: Optional[np.ndarray] = None): # Made embedding optional
        """Add new relevance feedback and its embedding."""
        self.relevance_history.append(feedback)
        
        if embedding is not None: # Only add if embedding is provided
            if feedback.is_relevant:
                self.positive_embeddings.append(embedding)
            else:
                self.negative_embeddings.append(embedding)
    
    def select_samples_for_feedback(self,
                                  candidate_embeddings: List[np.ndarray], # candidate_embeddings might not always be available
                                  candidate_items: Optional[List[Any]] = None, # Raw items if embeddings not available
                                  n_samples: int = 5) -> List[int]:
        """Select most informative samples for feedback using uncertainty sampling."""
        if not self.positive_embeddings and not self.negative_embeddings:
            # If no feedback yet, select random samples
            return np.random.choice(
                len(candidate_embeddings), 
                size=min(n_samples, len(candidate_embeddings)), 
                replace=False
            ).tolist()
        
        # Calculate uncertainty scores
        uncertainties = []
        for emb in candidate_embeddings:
            pos_sim = max([cosine_similarity([emb], [pos])[0][0] 
                         for pos in self.positive_embeddings]) if self.positive_embeddings else 0
            neg_sim = max([cosine_similarity([emb], [neg])[0][0] 
                         for neg in self.negative_embeddings]) if self.negative_embeddings else 0
            
            # High uncertainty when similar to both positive and negative examples
            uncertainty = abs(pos_sim - neg_sim)
            uncertainties.append(uncertainty)
        
        # Return indices of most uncertain samples
        return np.argsort(uncertainties)[-n_samples:].tolist()
    
    def get_relevance_score(self, embedding: np.ndarray) -> float:
        """Calculate relevance score based on similarity to feedback embeddings."""
        if not self.positive_embeddings:
            return 0.5  # Neutral score if no feedback
        
        # Calculate similarity to positive and negative examples
        pos_sim = np.mean([cosine_similarity([embedding], [pos])[0][0] 
                          for pos in self.positive_embeddings])
        neg_sim = np.mean([cosine_similarity([embedding], [neg])[0][0] 
                          for neg in self.negative_embeddings]) if self.negative_embeddings else 0
        
        # Normalize to [0, 1]
        return (pos_sim - neg_sim + 1) / 2
    
    def rerank_results(self, 
                      results: List[Dict[str, Any]], 
                      embeddings: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Rerank search results based on relevance feedback."""
        if not self.positive_embeddings:
            return results
        
        # Calculate relevance scores
        scores = [self.get_relevance_score(emb) for emb in embeddings]
        
        # Combine original scores with relevance scores
        for result, score in zip(results, scores):
            result["original_score"] = result["score"]
            result["relevance_score"] = score
            result["score"] = (result["score"] + score) / 2
        
        # Sort by combined score
        return sorted(results, key=lambda x: x["score"], reverse=True)
