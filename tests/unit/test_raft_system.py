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


class TestRAFTSystemRoutingAndEnsembling:

    @pytest.fixture
    def raft_system_for_routing(self, raft_config_path):
        # System with specific config for ensembling tests
        config_content = """
default_base_llm: "model_default"
intelligent_routing: true
enable_ensembling: true
ensembling_score_threshold: 0.6
max_ensemble_models: 2
rl_reward_weights: {} # Keep it simple for these tests
"""
        config_file = Path(raft_config_path).parent / "raft_config_routing.yaml"
        config_file.write_text(config_content)

        with patch('Scripts.raft_system.ModelRegistry', DummyModelRegistry), \
             patch('Scripts.raft_system.LLMService', DummyLLMService), \
             patch('Scripts.raft_system.HuggingFaceLLM', DummyLLMService), \
             patch('Scripts.raft_system.ActiveLearner', DummyActiveLearner):
            system = RAFTSystem(config_path=str(config_file), sandbox_test_mode=True)
            # Register some mock models
            system.global_llm_registry.register_model("model_A", DummyLLMService())
            system.global_llm_registry.register_model("model_B", DummyLLMService())
            system.global_llm_registry.register_model("model_C", DummyLLMService())
            system.global_llm_registry.register_model("model_default", DummyLLMService())
            return system

    def test_intelligent_route_model_returns_single_model(self, raft_system_for_routing: RAFTSystem):
        system = raft_system_for_routing
        system.config["enable_ensembling"] = False # Disable ensembling for this test

        # Mock performance analytics to give model_A a high score
        system.model_performance_analytics = {"model_A": {"rl_preference_score": 0.9}}

        result = system._intelligent_route_model(query="test query", domain="model_A")
        assert isinstance(result, str)
        assert result == "model_A"

    def test_intelligent_route_model_returns_ensemble_list(self, raft_system_for_routing: RAFTSystem):
        system = raft_system_for_routing
        # Ensure ensembling is enabled (it is by default in the fixture's config)

        # Mock performance analytics for scores
        system.model_performance_analytics = {
            "model_A": {"rl_preference_score": 0.9}, # Score: 1.0 (domain) + 0.9 * 0.5 = 1.45
            "model_B": {"rl_preference_score": 0.8}, # Score: 0.0 (no domain) + 0.8 * 0.5 = 0.4 -> too low for threshold 0.6
            "model_C": {"rl_preference_score": 0.5}  # Score: 0.0 + 0.5 * 0.5 = 0.25 -> too low
        }
        # To make model_B also be part of ensemble, its raw score needs to be >= 0.6
        # Let's adjust rl_preference_score for B
        system.model_performance_analytics["model_B"]["rl_preference_score"] = 1.3 # Score: 0.0 + 1.3 * 0.5 = 0.65

        result = system._intelligent_route_model(query="test query", domain="model_A")

        assert isinstance(result, list)
        assert len(result) == 2 # model_A and model_B (max_ensemble_models = 2 in config)
        result_names = [r['name'] for r in result]
        assert "model_A" in result_names
        assert "model_B" in result_names
        # Check scores are also present
        assert all("score" in r for r in result)


    @pytest.mark.asyncio
    async def test_agentic_rag_generate_single_model_path(self, raft_system_for_routing: RAFTSystem):
        system = raft_system_for_routing
        system.config["enable_ensembling"] = False # Ensure single model path

        # Mock the router to return a single model
        system._intelligent_route_model = MagicMock(return_value="model_A")

        # Mock the LLM instance that will be fetched from registry
        mock_llm_A_instance = MagicMock(spec=DummyLLMService)
        mock_llm_A_instance.generate = MagicMock(return_value="Response from Model A")
        system.global_llm_registry.register_model("model_A", mock_llm_A_instance)

        # Mock retrieve_documents
        system.retrieve_documents = MagicMock(return_value=[{"text": "context doc 1"}])

        response = await system.agentic_rag_generate(query="test query for single model")

        system._intelligent_route_model.assert_called_once()
        mock_llm_A_instance.generate.assert_called_once()
        assert response == "Response from Model A"

    @pytest.mark.asyncio
    async def test_agentic_rag_generate_ensemble_path(self, raft_system_for_routing: RAFTSystem):
        system = raft_system_for_routing
        # Ensembling is enabled by default in this fixture's config

        # Mock the router to return multiple models
        ensemble_candidates = [
            {"name": "model_A", "score": 0.9},
            {"name": "model_B", "score": 0.8}
        ]
        system._intelligent_route_model = MagicMock(return_value=ensemble_candidates)

        # Mock LLM instances
        mock_llm_A_instance = MagicMock(spec=DummyLLMService)
        mock_llm_A_instance.generate = MagicMock(return_value="Response from Model A")
        system.global_llm_registry.register_model("model_A", mock_llm_A_instance)

        mock_llm_B_instance = MagicMock(spec=DummyLLMService)
        mock_llm_B_instance.generate = MagicMock(return_value="Response from Model B")
        system.global_llm_registry.register_model("model_B", mock_llm_B_instance)

        system.retrieve_documents = MagicMock(return_value=[{"text": "context doc 1"}])

        response = await system.agentic_rag_generate(query="test query for ensemble")

        system._intelligent_route_model.assert_called_once()
        mock_llm_A_instance.generate.assert_called_once()
        mock_llm_B_instance.generate.assert_called_once()

        expected_response = "Response from model_A:\nResponse from Model A\n\n---\n\nResponse from model_B:\nResponse from Model B"
        assert response == expected_response

        # Also check RL step action representation
        # This requires access to the last call to integrate_reinforcement_learning_step
        # For simplicity, we'll assume track_experiment was called by it.
        # This is more of an integration check.
        # system.track_experiment = MagicMock()
        # ... call agentic_rag_generate ...
        # last_experiment_action = system.track_experiment.call_args[1]['params']['action']
        # assert last_experiment_action['selected_model_name'] == "ensemble"
        # assert "model_A" in last_experiment_action['ensembled_models']


# TODO: Add tests for integrate_reinforcement_learning_step to check how it uses the reward
# and how it updates model_performance_analytics (or calls an RL agent).
# TODO: Add tests for the fine-tuning methods if not covered elsewhere.
```
