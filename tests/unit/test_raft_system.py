import pytest
from unittest.mock import MagicMock, patch
import os

# Ensure Scripts directory is in path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Scripts')))

from Scripts.raft_system import RAFTSystem, DummyModelRegistry, DummyLLMService, DummyActiveLearner
# Assuming ModelVersion can be imported or is available via RAFTSystem instance's ModelVersion_class
# from Scripts.models.registry import ModelVersion

@pytest.fixture
def raft_config_path(tmp_path):
    # Create a dummy raft_config.yaml for testing
    config_content = """
default_base_llm: "mock-base-llm"
default_embedding_model: "mock-embedding-model"
intelligent_routing: true
continuous_learning: true
reinforcement_learning_integration: true
rl_reward_weights:
  user_feedback: 0.6
  response_quality_heuristic: 0.2
  response_time_penalty: -0.1
  context_relevance: 0.2
rl_response_time_penalty_threshold_ms: 3000
"""
    config_file = tmp_path / "raft_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)

@pytest.fixture
def basic_raft_system(raft_config_path):
    # Using sandbox_test_mode=True to use dummy components and avoid real model loading
    # We also need to mock the ModelVersion class used internally if it's from the real registry
    with patch('Scripts.raft_system.ModelRegistry', DummyModelRegistry), \
         patch('Scripts.raft_system.LLMService', DummyLLMService), \
         patch('Scripts.raft_system.HuggingFaceLLM', DummyLLMService), \
         patch('Scripts.raft_system.ActiveLearner', DummyActiveLearner), \
         patch('Scripts.raft_system.EmbeddingTrainer'), \
         patch('Scripts.raft_system.KnowledgeGraph'):
        # The ModelVersion class used by RAFTSystem is self.ModelVersion_class
        # In sandbox mode, this is already DummyModelVersion.
        system = RAFTSystem(config_path=raft_config_path, sandbox_test_mode=True)
        return system

class TestRAFTSystemRewardCalculation:

    def test_calculate_reward_positive_feedback(self, basic_raft_system: RAFTSystem):
        reward = basic_raft_system._calculate_reward(
            query="test query",
            response="good long response relevant to query",
            domain="test_domain",
            feedback_score=1.0, # Positive feedback
            response_time_ms=1000,
            retrieved_context_relevance=0.8
        )
        # Expected: 1.0*0.6 (feedback) + (0.5+0.3)*0.2 (quality) + 0 (time_penalty) + 0.8*0.2 (context)
        # = 0.6 + 0.16 + 0.16 = 0.92
        assert reward == pytest.approx(0.92)

    def test_calculate_reward_negative_feedback(self, basic_raft_system: RAFTSystem):
        reward = basic_raft_system._calculate_reward(
            query="test query",
            response="bad short", # Low quality score
            domain="test_domain",
            feedback_score=-1.0, # Negative feedback
            response_time_ms=6000, # High response time -> penalty
            retrieved_context_relevance=0.2 # Low context relevance
        )
        # Expected: -1.0*0.6 (feedback) + 0.5*0.2 (quality - only len > 20) + (-(6000-3000)/3000)*0.1 (time) + 0.2*0.2 (context)
        # = -0.6 + 0.1 - 0.1 + 0.04 = -0.56
        assert reward == pytest.approx(-0.56)

    def test_calculate_reward_no_feedback_good_response(self, basic_raft_system: RAFTSystem):
        reward = basic_raft_system._calculate_reward(
            query="test query",
            response="good long response relevant to query",
            domain="test_domain",
            feedback_score=None,
            response_time_ms=500,
            retrieved_context_relevance=0.9
        )
        # Expected: 0 (feedback) + (0.5+0.3)*0.2 (quality) + 0 (time) + 0.9*0.2 (context)
        # = 0.16 + 0.18 = 0.34
        assert reward == pytest.approx(0.34)

    def test_calculate_reward_only_time_penalty(self, basic_raft_system: RAFTSystem):
        reward = basic_raft_system._calculate_reward(
            query="test query",
            response="short", # low quality score
            domain="test_domain",
            feedback_score=None,
            response_time_ms=9000, # penalty = (9000-3000)/3000 = 2. Clipped to -1. So -1 * 0.1 = -0.1
            retrieved_context_relevance=None
        )
        # Expected: 0 (feedback) + 0 (quality) -0.1 (time) + 0 (context)
        # = -0.1
        assert reward == pytest.approx(-0.1)

    def test_calculate_reward_all_none_or_neutral(self, basic_raft_system: RAFTSystem):
        reward = basic_raft_system._calculate_reward(
            query="query",
            response="neutral response of decent length", # quality heuristic might give some points
            domain="generic",
            feedback_score=None,
            response_time_ms=1000, # No penalty
            retrieved_context_relevance=None
        )
        # Expected: 0 (feedback) + (0.5+0.0)*0.2 (quality, assuming 'query' not in 'neutral response...') + 0 (time) + 0 (context)
        # = 0.1
        assert reward == pytest.approx(0.1)

# TODO: Add tests for integrate_reinforcement_learning_step to check how it uses the reward
# and how it updates model_performance_analytics (or calls an RL agent).
# TODO: Add tests for the fine-tuning methods and agentic_rag_generate if not covered elsewhere.
```
