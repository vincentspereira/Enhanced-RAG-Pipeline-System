import pytest
from unittest.mock import Mock, patch
from Scripts.active_learning.learner import (
    ActiveLearner,
    RelevanceFeedback
)
import numpy as np

@pytest.fixture
def sample_embeddings():
    return np.random.randn(5, 768)  # 5 documents, 768 dimensions

@pytest.fixture
def active_learner():
    return ActiveLearner(embedding_dimension=768)

class TestActiveLearner:
    def test_initialization(self):
        """Test learner initialization"""
        learner = ActiveLearner(embedding_dimension=768)
        
        assert learner.embedding_dimension == 768
        assert len(learner.relevance_history) == 0
        assert len(learner.positive_embeddings) == 0
        assert len(learner.negative_embeddings) == 0
    
    def test_add_feedback(self, active_learner, sample_embeddings):
        """Test adding relevance feedback"""
        feedback = RelevanceFeedback(
            query_id="q1",
            document_id="d1",
            is_relevant=True,
            feedback_source="user",
            confidence=1.0
        )
        
        active_learner.add_feedback(feedback, sample_embeddings[0])
        
        assert len(active_learner.relevance_history) == 1
        assert len(active_learner.positive_embeddings) == 1
        assert len(active_learner.negative_embeddings) == 0
    
    def test_select_samples_no_feedback(self, active_learner, sample_embeddings):
        """Test sample selection without prior feedback"""
        selected = active_learner.select_samples_for_feedback(
            sample_embeddings.tolist(),
            n_samples=3
        )
        
        assert len(selected) == 3
        assert all(0 <= idx < len(sample_embeddings) for idx in selected)
    
    def test_select_samples_with_feedback(self, active_learner, sample_embeddings):
        """Test sample selection with existing feedback"""
        # Add some feedback first
        feedback_pos = RelevanceFeedback(
            query_id="q1",
            document_id="d1",
            is_relevant=True,
            feedback_source="user",
            confidence=1.0
        )
        feedback_neg = RelevanceFeedback(
            query_id="q1",
            document_id="d2",
            is_relevant=False,
            feedback_source="user",
            confidence=1.0
        )
        
        active_learner.add_feedback(feedback_pos, sample_embeddings[0])
        active_learner.add_feedback(feedback_neg, sample_embeddings[1])
        
        selected = active_learner.select_samples_for_feedback(
            sample_embeddings[2:].tolist(),
            n_samples=2
        )
        
        assert len(selected) == 2
        assert all(idx < len(sample_embeddings[2:]) for idx in selected)
    
    def test_get_relevance_score(self, active_learner, sample_embeddings):
        """Test relevance score calculation"""
        # Add some feedback
        feedback_pos = RelevanceFeedback(
            query_id="q1",
            document_id="d1",
            is_relevant=True,
            feedback_source="user",
            confidence=1.0
        )
        
        active_learner.add_feedback(feedback_pos, sample_embeddings[0])
        
        # Calculate score for a new embedding
        score = active_learner.get_relevance_score(sample_embeddings[1])
        
        assert 0 <= score <= 1
    
    def test_rerank_results(self, active_learner, sample_embeddings):
        """Test search result reranking"""
        # Create some test results
        results = [
            {"id": f"doc{i}", "score": 0.9 - i*0.1, "content": f"text{i}"}
            for i in range(3)
        ]
        
        # Add some feedback
        feedback = RelevanceFeedback(
            query_id="q1",
            document_id="d1",
            is_relevant=True,
            feedback_source="user",
            confidence=1.0
        )
        active_learner.add_feedback(feedback, sample_embeddings[0])
        
        # Rerank results
        reranked = active_learner.rerank_results(
            results,
            sample_embeddings[:3]
        )
        
        assert len(reranked) == len(results)
        assert all("relevance_score" in r for r in reranked)
        assert all("original_score" in r for r in reranked)
