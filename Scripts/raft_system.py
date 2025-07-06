"""
Advanced Agentic RAFT (Retrieval-Augmented Fine-Tuning) System.

This module implements the core RAFT mechanism, integrating Agentic RAG
with fine-tuning capabilities for domain-specific LLM optimization.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import tempfile # Added for execute_raft_cycle
from datetime import datetime # Added for model versioning and RAFT cycle timestamp

import tempfile # Added for execute_raft_cycle
from datetime import datetime # Added for model versioning and RAFT cycle timestamp

# Conditionally import heavy libraries
torch = None
yaml = None
wandb = None # Added for wandb integration
AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments = None, None, None, None
SentenceTransformer = None

# Dynamically import ModelVersion for type hinting if possible, actual class used will be self.ModelVersion_class
try:
    from Scripts.models.registry import ModelVersion
except ImportError:
    ModelVersion = DummyModelVersion # Fallback for type hinting if module not found early

# from Scripts.models.registry import ModelRegistry # Deferred
# Make EmbeddingTrainer and EmbeddingDataset optional for sandbox mode if they import heavy deps
# For now, assume they are light enough or handle conditional imports internally
# from Scripts.models.training.pipeline import EmbeddingTrainer, EmbeddingDataset # Deferred
# from Scripts.llm.service import LLMService, HuggingFaceLLM, llm_registry as global_llm_registry # Deferred
# from Scripts.active_learning.learner import ActiveLearner # Deferred
# from Scripts.knowledge_graph.graph import KnowledgeGraph # Deferred

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Lightweight Placeholder Classes for Sandbox Mode ---
class DummyLLMService: # Placeholder for LLMService
    def generate(self, prompt: str, **kwargs) -> str: return f"Dummy LLM response for {prompt}"
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]: return [f"Dummy LLM response for {p}" for p in prompts]

class DummyEmbeddingTrainer: # Placeholder for EmbeddingTrainer
    def __init__(self, *args, **kwargs): logger.debug("DummyEmbeddingTrainer initialized")
    def train(self, *args, **kwargs): logger.debug("DummyEmbeddingTrainer train called"); return "dummy_model_path"

class DummyActiveLearner: # Placeholder for ActiveLearner
    def __init__(self, *args, **kwargs): logger.debug("DummyActiveLearner initialized")
    def add_interaction_data(self, *args, **kwargs): logger.debug("DummyActiveLearner add_interaction_data called")

class DummyKnowledgeGraph: # Placeholder for KnowledgeGraph
    def __init__(self, *args, **kwargs): logger.debug("DummyKnowledgeGraph initialized")

class DummyModelRegistry: # Placeholder for ModelRegistry
    def __init__(self, *args, **kwargs): logger.debug("DummyModelRegistry initialized")
    def register_model(self, *args, **kwargs): logger.debug("DummyModelRegistry register_model called") # Matches ModelRegistry method
    def get_model_version(self, *args, **kwargs): logger.debug("DummyModelRegistry get_model_version called"); return None # Matches ModelRegistry method
    def list_models(self, *args, **kwargs): logger.debug("DummyModelRegistry list_models called"); return [] # Matches ModelRegistry method

class DummyModelVersion: # Placeholder for ModelVersion
    def __init__(self, name: str, version: str, path: str, metadata: Optional[Dict] = None, description: Optional[str] = None):
        self.name = name
        self.version = version
        self.path = path
        self.metadata = metadata or {}
        self.description = description
        logger.debug(f"DummyModelVersion {name} v{version} initialized.")

class DummyLLMRegistry: # Placeholder for llm_registry
    def __init__(self): self._models = {}
    def register_model(self, name, model): self._models[name] = model; logger.debug(f"DummyLLMRegistry registered {name}")
    def get_model(self, name): return self._models.get(name)
    def list_models(self): return list(self._models.keys())

# Use dummy registries in sandbox mode
_global_dummy_llm_registry = DummyLLMRegistry()
_global_dummy_model_registry = DummyModelRegistry()

# --- End Placeholder Classes ---

# Pre-Tuned Domain Models (example, replace with actual model names or paths)
# Reduced for sandbox testing
PRE_TUNED_DOMAINS = {
    "r_and_d": "models/r_and_d_llm", # Keep one for testing initialization
    # "legal_indian": "models/legal_indian_llm",
    # "legal_uk": "models/legal_uk_llm",
    # "legal_us": "models/legal_us_llm",
    # "finance": "models/finance_llm",
    # "uk_social_care": "models/uk_social_care_llm",
    # "uk_healthcare": "models/uk_healthcare_llm",
    # "education": "models/education_llm",
    # "customer_support": "models/customer_support_llm",
    # "technical_documentation": "models/technical_documentation_llm",
    # "creative_writing": "models/creative_writing_llm",
    # "machine_learning_data_analysis": "models/ml_data_analysis_llm",
}

DEFAULT_BASE_LLM = "gpt2" # Example base model for fine-tuning if a domain model isn't found
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class RAFTSystem:
    """
    Implements the Advanced Agentic RAFT System.
    """

    def __init__(
        self,
        model_registry_path: str = "data/model_registry.json",
        config_path: str = "config/raft_config.yaml",
        custom_finetuning_output_dir: str = "data/custom_llm_finetuned",
        embedding_finetuning_output_dir: str = "data/custom_embeddings_finetuned",
        use_wandb: bool = False,
        wandb_project: str = "raft-system",
        **kwargs # Added to accept other keyword arguments like sandbox_test_mode
    ):
        self.model_registry = ModelRegistry() # Uses the global instance for now
        self.global_llm_registry = global_llm_registry
        # self.config is loaded later, after sandbox_test_mode is set
        self.custom_finetuning_output_dir = custom_finetuning_output_dir
        os.makedirs(self.custom_finetuning_output_dir, exist_ok=True)

        self.sandbox_test_mode = kwargs.get("sandbox_test_mode", False)
        logger.info(f"RAFTSystem initialized. Sandbox mode: {self.sandbox_test_mode}")

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.current_wandb_run = None # To store the active run object

        if not self.sandbox_test_mode:
            global torch, yaml, wandb, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, SentenceTransformer
            # Dynamically import heavy libraries only when not in sandbox mode
            import torch
            import yaml as pyyaml_module
            yaml = pyyaml_module
            if self.use_wandb:
                try:
                    import wandb as wb_module
                    wandb = wb_module
                    logger.info("Weights & Biases library loaded.")
                except ImportError:
                    logger.warning("wandb library not found, but use_wandb is True. Disabling W&B.")
                    self.use_wandb = False # Disable if not found
            from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM, AutoTokenizer as HfAutoTokenizer, Trainer as HfTrainer, TrainingArguments as HfTrainingArguments
            AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments = HfAutoModelForCausalLM, HfAutoTokenizer, HfTrainer, HfTrainingArguments
            from sentence_transformers import SentenceTransformer as STrans
            SentenceTransformer = STrans

            # Dynamically import local heavy modules
            from Scripts.models.training.pipeline import EmbeddingTrainer
            from Scripts.llm.service import LLMService, HuggingFaceLLM, llm_registry as global_llm_registry
            from Scripts.active_learning.learner import ActiveLearner
            from Scripts.knowledge_graph.graph import KnowledgeGraph
            from Scripts.models.registry import ModelRegistry as RealModelRegistry, ModelVersion as RealModelVersion

            self.model_registry = RealModelRegistry()
            self.ModelVersion_class = RealModelVersion
            self.global_llm_registry = global_llm_registry
            self.embedding_trainer = EmbeddingTrainer(
                output_dir=embedding_finetuning_output_dir,
                model_registry=self.model_registry,
                use_wandb=self.use_wandb, # Pass the potentially updated use_wandb
                wandb_project=self.wandb_project
            )
            self.active_learner = ActiveLearner()
            self.knowledge_graph = KnowledgeGraph()
            LLMService_class = LLMService
            HuggingFaceLLM_class = HuggingFaceLLM
        else: # Sandbox mode
            self.model_registry = _global_dummy_model_registry
            self.ModelVersion_class = DummyModelVersion
            self.global_llm_registry = _global_dummy_llm_registry
            self.embedding_trainer = DummyEmbeddingTrainer()
            self.active_learner = DummyActiveLearner()
            self.knowledge_graph = DummyKnowledgeGraph()
            LLMService_class = DummyLLMService
            HuggingFaceLLM_class = DummyLLMService
            logger.info("[Sandbox Mode] Using dummy/placeholder components for RAFTSystem.")
            self.use_wandb = False # Ensure wandb is off in sandbox

        self.LLMService_class = LLMService_class
        self.HuggingFaceLLM_class = HuggingFaceLLM_class
        self.config = self._load_config(config_path)
        self._initialize_pre_tuned_models()
        self.model_performance_analytics = {}
        self.experiment_tracking = {} # This is the local dict, wandb is separate

        if self.use_wandb and wandb: # Check wandb module itself also
            # Initialize a default run or manage runs per operation
            # For now, let's not start a run here, but in specific methods like execute_raft_cycle
            logger.info(f"W&B enabled. Project: {self.wandb_project}. Run will be initialized per operation.")


    def _ensure_libs_loaded(self):
        """Ensures heavy libraries are loaded if not in sandbox mode and not already loaded."""
        if not self.sandbox_test_mode:
            global torch, yaml, wandb, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, SentenceTransformer
            if torch is None or AutoModelForCausalLM is None or SentenceTransformer is None or yaml is None:
                import torch
                import yaml as pyyaml_module
                yaml = pyyaml_module
                from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM, \
                                         AutoTokenizer as HfAutoTokenizer, \
                                         Trainer as HfTrainer, \
                                         TrainingArguments as HfTrainingArguments
                AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments = HfAutoModelForCausalLM, HfAutoTokenizer, HfTrainer, HfTrainingArguments
                from sentence_transformers import SentenceTransformer as STrans
                SentenceTransformer = STrans
                logger.info("Core heavy libraries dynamically loaded.")

            if self.use_wandb and wandb is None: # Check if wandb needs to be loaded
                try:
                    import wandb as wb_module
                    wandb = wb_module
                    logger.info("Weights & Biases library dynamically loaded.")
                except ImportError:
                    logger.warning("wandb library not found during _ensure_libs_loaded, but use_wandb is True. Disabling W&B for this session.")
                    self.use_wandb = False


    def _load_config(self, config_path: str) -> Dict:
        """Loads RAFT system configuration."""
        self._ensure_libs_loaded() # yaml needed here
        if yaml is None and not self.sandbox_test_mode: # Should have been loaded by ensure
             raise ImportError("YAML library not loaded despite ensure_libs_loaded call.")
        if self.sandbox_test_mode and yaml is None: # If in sandbox, yaml might not be loaded
            logger.warning(f"Sandbox mode: YAML not loaded. Using default config as {config_path} cannot be read.")
            return {
                "default_base_llm": DEFAULT_BASE_LLM,
                "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
                "intelligent_routing": True,
                "continuous_learning": True,
                "reinforcement_learning_integration": True,
            }
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) # Now yaml refers to the imported module
            logger.info(f"RAFT configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return { # Default config
                "default_base_llm": DEFAULT_BASE_LLM,
                "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
                "intelligent_routing": True,
                "continuous_learning": True,
                "reinforcement_learning_integration": True, # Placeholder
            }

    def _initialize_pre_tuned_models(self):
        """Initializes pre-tuned domain models and registers them."""
        if self.sandbox_test_mode:
            logger.info(f"[Sandbox Mode] Skipping actual pre-tuned model loading. Registering dummy placeholders if needed.")
            for domain in PRE_TUNED_DOMAINS.keys():
                if not self.global_llm_registry.get_model(domain): # Check if dummy already registered
                    class DummyLLM(LLMService):
                        def generate(self, prompt: str, **kwargs) -> str: return f"Dummy response for {prompt} from domain {domain}"
                        def batch_generate(self, prompts: List[str], **kwargs) -> List[str]: return [f"Dummy response for {p} from domain {domain}" for p in prompts]
                    self.global_llm_registry.register_model(domain, DummyLLM())
            return

        # This part only runs if not in sandbox_test_mode
        self._ensure_libs_loaded()
        for domain, model_path_or_name in PRE_TUNED_DOMAINS.items():
            try:
                if os.path.exists(model_path_or_name):
                    llm = HuggingFaceLLM(model_name=model_path_or_name)
                else:
                    llm = HuggingFaceLLM(model_name=model_path_or_name)
                self.global_llm_registry.register_model(domain, llm)
                logger.info(f"Initialized and registered pre-tuned model for domain: {domain}")
            except Exception as e:
                logger.error(f"Failed to load pre-tuned model for {domain} ({model_path_or_name}): {e}. Using default base LLM as fallback.")
                if DEFAULT_BASE_LLM not in self.global_llm_registry.list_models():
                     # Ensure default is loaded once
                    default_llm_instance = HuggingFaceLLM(DEFAULT_BASE_LLM)
                    self.global_llm_registry.register_model(DEFAULT_BASE_LLM, default_llm_instance)
                self.global_llm_registry.register_model(domain, self.global_llm_registry.get_model(DEFAULT_BASE_LLM))


    def _get_embedding_model(self, model_name_or_path: Optional[str] = None) -> Optional[Any]: # Return Any for Dummy
        """Retrieves or loads an embedding model. Returns a dummy in sandbox_test_mode."""
        if self.sandbox_test_mode:
            model_key = model_name_or_path or self.config.get("default_embedding_model", DEFAULT_EMBEDDING_MODEL)
            logger.info(f"[Sandbox Mode] Returning dummy embedding model for: {model_key}")
            if not hasattr(self, '_embedding_models_cache'): self._embedding_models_cache = {}
            if model_key not in self._embedding_models_cache:
                class DummySentenceTransformer:
                    def encode(self, query): logger.debug(f"DummyEmbedding: encode('{query}')"); return [0.1] * 384 # e.g. all-MiniLM-L6-v2 dim
                self._embedding_models_cache[model_key] = DummySentenceTransformer()
            return self._embedding_models_cache[model_key]

        # This part only runs if not in sandbox_test_mode
        self._ensure_libs_loaded()
        model_key = model_name_or_path or self.config.get("default_embedding_model", DEFAULT_EMBEDDING_MODEL)
        if not hasattr(self, '_embedding_models_cache'):
            self._embedding_models_cache = {}

        if model_key not in self._embedding_models_cache:
            logger.info(f"Loading embedding model: {model_key}")
            try:
                self._embedding_models_cache[model_key] = SentenceTransformer(model_key)
            except Exception as e:
                logger.error(f"Failed to load embedding model {model_key}: {e}. Falling back to default.")
                default_emb_model = self.config.get("default_embedding_model", DEFAULT_EMBEDDING_MODEL)
                if model_key != default_emb_model:
                    return self._get_embedding_model(default_emb_model)
                else:
                    raise RuntimeError(f"Could not load any embedding model, including default {default_emb_model}.")
        return self._embedding_models_cache[model_key]

    def retrieve_documents(self, query: str, domain: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant documents using hybrid search (placeholder for now).
        This should integrate with the Advanced Hybrid Search module.
        """
        logger.info(f"Retrieving documents for query: '{query}' in domain: {domain}")
        # Placeholder: Simple semantic search using the default/domain embedding model
        # This will be replaced by a call to the dedicated Hybrid Search module
        embedding_model = self._get_embedding_model() # Later, select based on domain or query
        query_embedding = embedding_model.encode(query)

        # Placeholder for document store and actual search logic
        # Retrieved_docs should be a list of dicts, e.g., [{"text": "doc text", "metadata": {...}}, ...]
        retrieved_docs = [{"text": f"Placeholder document for query '{query}' - doc {i+1}", "metadata": {"source": "placeholder_db"}} for i in range(top_k)]
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        return retrieved_docs

    def agentic_rag_generate(
        self,
        query: str,
        domain: Optional[str] = None,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        llm_options: Optional[Dict] = None
    ) -> str:
        """
        Generates a response using the RAG pattern with an agentically selected LLM.
        """
        llm_options = llm_options or {}
        if retrieved_docs is None:
            retrieved_docs = self.retrieve_documents(query, domain)

        context = "\n".join([doc["text"] for doc in retrieved_docs])
        prompt = f"Based on the following context:\n{context}\n\nAnswer the question: {query}"

        routed_model_info = self._intelligent_route_model(query, domain, complexity=None) # Assuming complexity is not used yet or passed as None

        final_response = ""
        generation_params = {
            "max_length": llm_options.get("max_length", 250),
            "temperature": llm_options.get("temperature", 0.7)
        }
        action_representation = {"generation_params_used": generation_params}

        if isinstance(routed_model_info, list) and self.config.get("enable_ensembling", False):
            # Ensembling path
            logger.info(f"Ensembling responses from models: {[m['name'] for m in routed_model_info]}")
            ensemble_responses = []
            actual_models_used_for_ensemble = []

            for model_candidate in routed_model_info: # Iterates list of {'name': str, 'score': float}
                model_name = model_candidate['name']
                actual_models_used_for_ensemble.append(model_name)
                llm_instance = self.global_llm_registry.get_model(model_name)
                if not llm_instance:
                    logger.error(f"Could not find LLM instance for '{model_name}' during ensembling. Skipping this model.")
                    continue

                try:
                    model_response = llm_instance.generate(prompt, **generation_params)
                    ensemble_responses.append({"model_name": model_name, "response": model_response})
                except Exception as e_gen:
                    logger.error(f"Error generating response from model '{model_name}' during ensembling: {e_gen}")
                    ensemble_responses.append({"model_name": model_name, "response": f"[Error generating from {model_name}]"})

            # Simple Concatenation with Attribution
            if ensemble_responses:
                final_response_parts = [f"Response from {er['model_name']}:\n{er['response']}" for er in ensemble_responses]
                final_response = "\n\n---\n\n".join(final_response_parts)
            else:
                final_response = "[No responses generated by ensemble models]"

            action_representation["selected_model_name"] = "ensemble"
            action_representation["ensembled_models"] = actual_models_used_for_ensemble
            action_representation["ensemble_strategy"] = "concatenation_with_attribution"

        else: # Single model path (either ensembling disabled or routing returned single string)
            selected_llm_name = routed_model_info if isinstance(routed_model_info, str) else routed_model_info[0]['name'] # If list but len 1
            logger.info(f"Using single LLM: {selected_llm_name} for generation.")
            llm_instance = self.global_llm_registry.get_model(selected_llm_name)

            if not llm_instance:
                logger.error(f"LLM instance for '{selected_llm_name}' not found. Cannot generate response.")
                final_response = "[Error: Selected LLM not found]"
            else:
                final_response = llm_instance.generate(prompt, **generation_params)

            action_representation["selected_model_name"] = selected_llm_name

        # Continuous learning & RL step
        if self.config.get("continuous_learning", True):
            user_feedback_score_placeholder = None # TODO: Obtain actual feedback
            self.active_learner.add_interaction_data(query, context, final_response, domain=domain, feedback_score=user_feedback_score_placeholder)

            if self.config.get("reinforcement_learning_integration", True):
                state_representation = {
                    "query_length": len(query), "domain": domain, "context_length": len(context)
                }
                # TODO: Add other state features: query_type, complexity_estimate etc.

                response_time_ms_placeholder = None # TODO: Measure actual response time
                retrieved_context_relevance_placeholder = None # TODO: Get from search/retrieval step

                reward = self._calculate_reward(
                    query=query, response=final_response, domain=domain,
                    feedback_score=user_feedback_score_placeholder,
                    response_time_ms=response_time_ms_placeholder,
                    retrieved_context_relevance=retrieved_context_relevance_placeholder
                )
                next_state_representation = {"response_length": len(final_response)}

                self.integrate_reinforcement_learning_step(
                    state=state_representation, action=action_representation,
                    reward=reward, next_state=next_state_representation
                )
        return final_response

    def _intelligent_route_model(self, query: str, domain: Optional[str] = None, complexity: Optional[str] = None) -> Union[str, List[Dict[str, Any]]]:
        """
        Intelligently selects the best LLM based on query, domain, complexity, and performance.
        Can return a single model name or a list of models for ensembling if enabled.
        """
        # Config for ensembling
        enable_ensembling = self.config.get("enable_ensembling", False)
        ensembling_score_threshold = self.config.get("ensembling_score_threshold", 0.7) # Min score for a model to be considered for ensembling
        max_ensemble_models = self.config.get("max_ensemble_models", 3)

        if not self.config.get("intelligent_routing", True): # Default to True if not specified
            selected_model = domain or self.config.get("default_base_llm", DEFAULT_BASE_LLM)
            logger.info(f"Intelligent routing disabled. Selected model: {selected_model}")
            return selected_model # Return single model name

        # --- Candidate Gathering (as before) ---
        candidate_models_names = []
        if domain and domain in self.global_llm_registry.list_models():
            candidate_models_names.append(domain)

        all_registered_llms = self.global_llm_registry.list_models()
        for m_name in all_registered_llms:
            if m_name not in candidate_models_names:
                candidate_models_names.append(m_name)

        if not candidate_models_names:
            logger.warning("Intelligent routing: No models in global_llm_registry. Falling back to DEFAULT_BASE_LLM.")
            # Ensure DEFAULT_BASE_LLM is loadable (simplified from previous version)
            if DEFAULT_BASE_LLM not in self.global_llm_registry.list_models():
                 self.global_llm_registry.register_model(DEFAULT_BASE_LLM, self.LLMService_class()) # Register dummy or real
            return DEFAULT_BASE_LLM

        # --- Scoring Candidates (incorporating query analysis placeholders) ---
        # TODO: Implement actual query feature extraction (length, keywords, intent, topic)
        query_features = {"length": len(query), "keywords": query.lower().split()[:5]} # Basic example

        scored_candidates = []
        for model_name_cand in candidate_models_names:
            score = 0.0
            # Factor 1: Domain match
            if model_name_cand == domain:
                score += 1.0  # Base score for direct domain match

            # Factor 2: Query feature match (conceptual)
            # e.g., if model_name_cand is good for "question_answering" and query is a question.
            # For now, this is a placeholder.
            # if self._model_suits_query_features(model_name_cand, query_features):
            #    score += 0.5

            # Factor 3: RL preference score
            if model_name_cand in self.model_performance_analytics:
                rl_pref = self.model_performance_analytics[model_name_cand].get("rl_preference_score", 0.0)
                score += rl_pref * self.config.get("rl_preference_weight_in_routing", 0.5)
                logger.debug(f"Routing: Model '{model_name_cand}' RL pref score: {rl_pref:.3f}, weighted: {rl_pref * 0.5:.3f}")

            # Factor 4: Historical performance (placeholder)
            # historical_perf = self.model_performance_analytics.get(model_name_cand, {}).get("avg_accuracy_for_query_type_X", 0.0)
            # score += historical_perf * 0.3

            scored_candidates.append({"name": model_name_cand, "score": score})

        sorted_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)

        if not sorted_candidates:
            selected_model_name = self.config.get("default_base_llm", DEFAULT_BASE_LLM)
            logger.warning(f"Intelligent routing: No candidates after scoring, falling back to default: {selected_model_name}")
            return selected_model_name

        # --- Decide on single model vs. ensemble ---
        if enable_ensembling:
            top_candidate_score = sorted_candidates[0]["score"]
            ensemble_candidates = [
                cand for cand in sorted_candidates
                if cand["score"] >= ensembling_score_threshold and \
                   cand["score"] >= top_candidate_score * 0.8 # Also consider models close to the best
            ][:max_ensemble_models]

            if len(ensemble_candidates) > 1:
                logger.info(f"Intelligent routing selected models for ensembling: {[c['name'] for c in ensemble_candidates]} with scores {[c['score'] for c in ensemble_candidates]}")
                return ensemble_candidates # Return list of dicts {'name': ..., 'score': ...}
            else:
                # Not enough good candidates for ensembling, fall back to single best
                selected_model_name = sorted_candidates[0]["name"]
                logger.info(f"Ensembling enabled, but only one strong candidate. Selected model: '{selected_model_name}' score {sorted_candidates[0]['score']:.3f}")
                return selected_model_name
        else: # Ensembling not enabled
            selected_model_name = sorted_candidates[0]["name"]
            logger.info(f"Intelligent routing selected model: '{selected_model_name}' score {sorted_candidates[0]['score']:.3f} (Ensembling disabled)")
            return selected_model_name

        # Fallback in case logic above doesn't return (should not happen)
        # return self.config.get("default_base_llm", DEFAULT_BASE_LLM)

    def fine_tune_llm_on_custom_data(
        self,
        dataset_path: str, # Path to custom dataset (e.g., CSV, JSON lines with "text" field)
        model_name: str,   # Name for the fine-tuned model
        base_model_name_or_path: Optional[str] = None,
        training_args_override: Optional[Dict] = None,
        domain_for_base: Optional[str] = None # If base_model_name is None, use this domain's model
    ) -> ModelVersion:
        """
        Fine-tunes an LLM on a custom dataset.
        """
        logger.info(f"Starting fine-tuning for model '{model_name}' using dataset: {dataset_path}")

        # Determine base model
        if base_model_name_or_path:
            _base_model_name = base_model_name_or_path
        elif domain_for_base and domain_for_base in self.global_llm_registry.list_models():
            # This assumes the registered model is a path or HuggingFace name
            # For HuggingFaceLLM, the .model attribute holds the actual transformers model
            # but for fine-tuning, we need the name/path.
            # This part needs careful handling of how models are stored/accessed by the registry.
            # For now, we'll assume PRE_TUNED_DOMAINS stores paths/names directly usable.
             _base_model_name = PRE_TUNED_DOMAINS.get(domain_for_base, self.config.get("default_base_llm", DEFAULT_BASE_LLM))
        else:
            _base_model_name = self.config.get("default_base_llm", DEFAULT_BASE_LLM)
        logger.info(f"Using base model: {_base_model_name}")

        # Load tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(_base_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(_base_model_name)
        except Exception as e:
            logger.error(f"Error loading base model '{_base_model_name}' for fine-tuning: {e}")
            raise

        # Load and prepare dataset (simple example: assumes text file, one doc per line)
        # TODO: Add automated data quality assessment here
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            if not texts:
                raise ValueError("Dataset is empty or not in the expected format.")
            logger.info(f"Loaded {len(texts)} texts from {dataset_path} for fine-tuning.")
        except Exception as e:
            logger.error(f"Error loading or processing dataset {dataset_path}: {e}")
            raise

        # This is a simplified dataset preparation. For Causal LM, you'd typically format as prompt/completion pairs
        # or just continuous text. Here, we'll use a simple TextDataset equivalent.
        class SimpleTextDataset(torch.utils.data.Dataset):
            def __init__(self, tokenizer, texts, block_size=128):
                self.examples = []
                for text in texts:
                    tokenized_text = tokenizer.encode(text, add_special_tokens=True)
                    # Create chunks of block_size
                    for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                        self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, i):
                return torch.tensor(self.examples[i], dtype=torch.long)

        train_dataset = SimpleTextDataset(tokenizer, texts) # Use all data for training for simplicity

        # Define training arguments
        output_model_dir = os.path.join(self.custom_finetuning_output_dir, model_name)
        os.makedirs(output_model_dir, exist_ok=True)

        default_training_args = {
            "output_dir": output_model_dir,
            "num_train_epochs": 1, # Keep low for example
            "per_device_train_batch_size": 1, # Keep low for example
            "save_steps": 10_000, # Placeholder
            "save_total_limit": 2,
            "logging_dir": os.path.join(output_model_dir, "logs"),
            "fp16": torch.cuda.is_available(), # Use FP16 if GPU is available
        }
        if training_args_override:
            default_training_args.update(training_args_override)

        training_args = TrainingArguments(**default_training_args)

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )

        # Start fine-tuning
        logger.info("Starting LLM fine-tuning process...")
        training_start_time = datetime.now() # Record start time
        trainer.train()
        trainer.save_model(output_model_dir)
        tokenizer.save_pretrained(output_model_dir)
        logger.info(f"Fine-tuned LLM saved to {output_model_dir}")

        # Register the fine-tuned model
        # TODO: Implement proper model versioning and metadata (metrics, etc.)
        # For now, we'll just add it to the global LLM registry for use
        self.global_llm_registry.register_model(model_name, HuggingFaceLLM(model_name=output_model_dir))
        logger.info(f"Fine-tuned model '{model_name}' registered.")

        training_end_time = datetime.now()
        # Ensure training_start_time is defined; if not, this indicates an issue or prior step skip.
        training_duration_seconds = None
        if 'training_start_time' in locals(): # Check if training_start_time was set
            training_duration_seconds = (training_end_time - training_start_time).total_seconds()
        else:
            logger.warning("training_start_time not found in local scope for fine_tune_llm_on_custom_data. Duration will be None.")

        # TODO: Add actual evaluation metrics post-training if a quick eval step is feasible
        # For now, using loss from trainer state if available
        final_loss = None
        if hasattr(trainer, 'state') and trainer.state.log_history:
            # Find the last entry that contains 'loss' or 'train_loss'
            for log_entry in reversed(trainer.state.log_history):
                if 'loss' in log_entry:
                    final_loss = log_entry['loss']
                    break
                elif 'train_loss' in log_entry: # Sometimes it's train_loss
                    final_loss = log_entry['train_loss']
                    break
        eval_metrics_data = {"final_training_loss": final_loss} # More descriptive key

        model_version_metadata = {
            "base_model": _base_model_name,
            "training_dataset": os.path.basename(dataset_path),
            "training_args_used": default_training_args, # The actual args passed to TrainingArguments
            "training_duration_seconds": training_duration_seconds,
            "eval_metrics": eval_metrics_data,
            "model_type": "llm", # Explicitly set model type
            "fine_tuning_method": "huggingface_trainer_api" # More specific
        }

        # Generate a version string, e.g., based on timestamp or an incrementing number
        # This could be enhanced later with git commit hash or other versioning schemes
        version_str = f"1.0.0-{training_end_time.strftime('%Y%m%d%H%M%S')}"

        # Use self.ModelVersion_class which could be DummyModelVersion or RealModelVersion
        model_version_obj = self.ModelVersion_class(
            name=model_name, # The name for this specific fine-tuned variant
            version=version_str,
            path=output_model_dir, # Path where the model is saved
            metadata=model_version_metadata,
            description=f"Custom fine-tuned LLM '{model_name}' (v{version_str}) based on '{_base_model_name}' using dataset '{os.path.basename(dataset_path)}'."
        )

        # Register with the main model registry (self.model_registry)
        # This registry instance should handle the ModelVersionTracker interaction.
        self.model_registry.register_model(model_version_obj) # ModelRegistry.register_model expects a ModelVersion object
        logger.info(f"Fine-tuned LLM '{model_name}' (version: {version_str}) registered with ModelRegistry.")

        # Also, update the global_llm_registry if this model is to be immediately usable by that name
        # This assumes HuggingFaceLLM can be loaded from the output_model_dir
        if not self.sandbox_test_mode : # Avoid loading real models in sandbox
             self.global_llm_registry.register_model(model_name, self.HuggingFaceLLM_class(model_name=output_model_dir))
             logger.info(f"Fine-tuned model '{model_name}' also made available in global_llm_registry.")
        else:
             logger.info(f"[Sandbox Mode] Fine-tuned model '{model_name}' registered in ModelRegistry (dummy), not loaded into global_llm_registry.")

        # Track this fine-tuning event as an experiment
        self.track_experiment(
            experiment_name=f"llm_finetune_{model_name}_{version_str}",
            params={
                "model_name": model_name,
                "version": version_str,
                "base_model": _base_model_name,
                "dataset": os.path.basename(dataset_path),
                "training_args": default_training_args,
            },
            metrics={
                "final_training_loss": final_loss,
                "training_duration_seconds": training_duration_seconds,
                # TODO: Add more comprehensive evaluation metrics here after an eval step
            },
            experiment_type="llm_fine_tuning"
        )

        return model_version_obj # Return the created ModelVersion object


    async def execute_raft_cycle(self, query: str, domain: Optional[str] = None, search_system: Optional[Any] = None, wandb_run: Optional[Any] = None) -> Dict[str, Optional[str]]: # search_system is AdvancedSearchSystem, wandb_run for W&B
        """
        Executes a single RAFT cycle: Retrieve, Augment, Fine-tune, Generate.
        Returns a dictionary with 'model_path' and 'response'.
        """
        if self.sandbox_test_mode:
            logger.info("[Sandbox Mode] execute_raft_cycle is conceptual only.")
            return {"model_path": "sandbox_dummy_model_path", "response": f"Sandbox dummy response to: {query}"}

        self._ensure_libs_loaded()
        if not hasattr(self, 'LLMService_class') or not hasattr(self, 'HuggingFaceLLM_class'):
             logger.error("LLMService or HuggingFaceLLM class not available. Aborting RAFT cycle.")
             return {"model_path": None, "response": "Error: Core LLM classes not loaded."}

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        logger.info(f"RAFT Cycle {timestamp} initiated for query: '{query}' in domain: '{domain}'")

        base_model_for_finetuning = self._intelligent_route_model(query, domain) # Determine upfront for tracking
        finetuned_model_details = None # Path or name of the fine-tuned model if successful

        # 1. Retrieve
        retrieved_docs: List[Any] = []
        if search_system is None:
            logger.warning("RAFT Cycle: No search_system provided. Using placeholder retrieval.")
            _placeholder_docs = self.retrieve_documents(query, domain, top_k=5)
            retrieved_docs = [{"id": f"placeholder_{i}", "content": d.get("text"), "score": 1.0, "source_type": "placeholder"} for i,d in enumerate(_placeholder_docs)]
        else:
            try:
                retrieved_docs = await search_system.search(query, top_k=5, use_reranker=True) # Use reranker
                logger.info(f"RAFT Cycle: Retrieved {len(retrieved_docs)} documents via AdvancedSearchSystem.")
            except Exception as e:
                logger.error(f"RAFT Cycle: Error during document retrieval: {e}", exc_info=True)

        if not retrieved_docs:
            logger.warning(f"RAFT Cycle: No documents retrieved for query '{query}'. Generating with existing model.")
            self._track_raft_experiment(timestamp, query, domain, base_model_for_finetuning, None, "skipped_finetuning_no_docs", "No docs retrieved")
            return await self._generate_with_existing_model(query, domain, retrieved_docs) # Make helper async

        # 2. Augment Data
        generated_training_examples = []
        augmentation_llm_name = self.config.get("data_augmentation_llm", self.config.get("default_base_llm", DEFAULT_BASE_LLM))
        augmentation_llm = self.global_llm_registry.get_model(augmentation_llm_name)

        if not augmentation_llm:
            logger.error(f"RAFT Cycle: Data augmentation LLM '{augmentation_llm_name}' not found. Falling back to simple text concatenation.")
            augmented_data_text = "\n\n".join([doc.content for doc in retrieved_docs if doc.content and doc.content.strip()])
        else:
            logger.info(f"RAFT Cycle: Using LLM '{augmentation_llm_name}' for data augmentation.")
            for i, doc_chunk in enumerate(retrieved_docs):
                if not doc_chunk.content or not doc_chunk.content.strip(): continue

                prompt_for_augmentation = f"Context: {doc_chunk.content}\n\nBased *only* on the context above, answer the following question concisely: {query}\n\nAnswer:"
                try:
                    generated_response = augmentation_llm.generate(prompt_for_augmentation, max_length=150)
                    if generated_response and generated_response.strip():
                        training_example_text = f"Context: {doc_chunk.content}\nQuestion: {query}\nAnswer: {generated_response.strip()}"
                        generated_training_examples.append(training_example_text)
                    else:
                        logger.warning(f"RAFT Cycle: Augmentation LLM produced empty response for chunk {i}. Using raw chunk.")
                        generated_training_examples.append(doc_chunk.content)
                except Exception as e_aug:
                    logger.error(f"RAFT Cycle: Error during LLM-based data augmentation for chunk {i}: {e_aug}", exc_info=True)
                    generated_training_examples.append(doc_chunk.content) # Fallback to raw content

        augmented_data_text = "\n\n".join(generated_training_examples)

        if not augmented_data_text.strip():
            logger.warning(f"RAFT Cycle: No usable content after augmentation for query '{query}'. Generating with existing model.")
            self._track_raft_experiment(timestamp, query, domain, base_model_for_finetuning, None, "skipped_finetuning_no_aug_data", "No data after augmentation")
            return await self._generate_with_existing_model(query, domain, retrieved_docs)

        # 3. Fine-tune
        temp_dataset_path = None
        generation_llm_name = base_model_for_finetuning # Default to base if FT fails

        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as tmp_file:
                tmp_file.write(augmented_data_text)
                temp_dataset_path = tmp_file.name
            logger.info(f"RAFT Cycle: Augmented dataset for fine-tuning saved to {temp_dataset_path}")

            finetuned_model_name_prefix = f"{domain}_raft_{timestamp}" if domain else f"general_raft_{timestamp}"

            logger.info(f"RAFT Cycle: Starting fine-tuning of '{base_model_for_finetuning}' to create '{finetuned_model_name_prefix}-variant'.")

            model_version_obj = self.fine_tune_llm_on_custom_data(
                dataset_path=temp_dataset_path,
                model_name=finetuned_model_name_prefix,
                base_model_name_or_path=base_model_for_finetuning,
                training_args_override={"report_to": "wandb" if self.use_wandb else "none"} # Pass W&B config
            )
            finetuned_model_details = model_version_obj.path if hasattr(model_version_obj, 'path') else finetuned_model_name_prefix
            generation_llm_name = finetuned_model_name_prefix
            logger.info(f"RAFT Cycle: Fine-tuning complete. New model path/name: {finetuned_model_details}")

        except Exception as e_ft:
            logger.error(f"RAFT Cycle: Error during fine-tuning for query '{query}': {e_ft}", exc_info=True)
            finetuned_model_details = f"failed_ft_fallback_to_{generation_llm_name}" # Stays as base_model_for_finetuning
        finally:
            if temp_dataset_path and os.path.exists(temp_dataset_path):
                os.remove(temp_dataset_path)
                logger.debug(f"RAFT Cycle: Cleaned up temporary dataset {temp_dataset_path}")

        # 4. Generate
        generation_llm = self.global_llm_registry.get_model(generation_llm_name)
        if not generation_llm: # Should not happen if fine_tune_llm_on_custom_data registers correctly or fallback works
            logger.critical(f"RAFT Cycle: CRITICAL - Could not load LLM '{generation_llm_name}' for generation.")
            self._track_raft_experiment(timestamp, query, domain, base_model_for_finetuning, finetuned_model_details, "failed", "LLM loading critical failure")
            return {"model_path": finetuned_model_details, "response": "Error: Critical LLM loading failure."}

        rag_context = "\n".join([doc.content for doc in retrieved_docs if doc.content])
        prompt = f"Based on the following context:\n{rag_context}\n\nAnswer the question: {query}"

        logger.info(f"RAFT Cycle: Generating response using model '{generation_llm_name}'.")
        response = generation_llm.generate(prompt)

        self._track_raft_experiment(timestamp, query, domain, base_model_for_finetuning, finetuned_model_details, "success", response)
        logger.info(f"RAFT Cycle for query '{query}' complete. Response generated with model '{generation_llm_name}'.")
        return {"model_path": finetuned_model_details, "response": response}

    async def _generate_with_existing_model(self, query: str, domain: Optional[str], retrieved_docs: List[Any]) -> Dict[str, str]:
        """Helper to generate response using an existing model when fine-tuning is skipped."""
        generation_llm_name = self._intelligent_route_model(query, domain)
        if not self.global_llm_registry.get_model(generation_llm_name): # Check if model key exists
            logger.error(f"Intelligent routing selected non-existent model: {generation_llm_name}. Falling back.")
            generation_llm_name = self.config.get("default_base_llm", DEFAULT_BASE_LLM)

        generation_llm = self.global_llm_registry.get_model(generation_llm_name) # Get the actual model object
        if not generation_llm:
            logger.error(f"Could not load LLM instance for {generation_llm_name}. Aborting.")
            return {"model_path": f"error_model_not_found:{generation_llm_name}", "response": "Error: Could not load LLM for generation."}

        rag_context = "\n".join([doc.content for doc in retrieved_docs if doc.content])
        prompt = f"Based on the following context (if any):\n{rag_context}\n\nAnswer the question: {query}"
        response = generation_llm.generate(prompt)
        logger.info(f"RAFT Cycle (skipped fine-tuning): Used existing model '{generation_llm_name}' for generation.")
        return {"model_path": f"existing_model:{generation_llm_name}", "response": response}

    def _track_raft_experiment(self, timestamp:str, query:str, domain:Optional[str], base_model:str, ft_model_details:Optional[str], status:str, response_or_error:str):
        """Helper to track RAFT experiment details."""
        # Ensure all potentially referenced variables are defined or have defaults
        params = {
            "query": query,
            "domain": domain,
            "base_model_for_finetuning": base_model if base_model else 'N/A',
            "finetuned_model_details": ft_model_details if ft_model_details else 'N/A'
        }
        metrics = {
            "status": status,
            "response_length": len(response_or_error) if status == "success" else 0,
            "outcome_message": response_or_error if status != "success" else "N/A"
        }
        self.track_experiment(
            experiment_name=f"raft_cycle_{domain or 'general'}_{timestamp}",
            params=params,
            metrics=metrics
        )

    def fine_tune_embedding_model_on_custom_data(
        self,
        texts: List[str], # List of texts for unsupervised fine-tuning
        model_name: str,  # Name for the fine-tuned embedding model
        base_model_name: Optional[str] = None, # e.g., "sentence-transformers/all-MiniLM-L6-v2"
        training_args_override: Optional[Dict] = None,
    ):
        """Fine-tunes an embedding model on custom text data."""
        logger.info(f"Starting embedding model fine-tuning for '{model_name}'")
        _base_embedding_model = base_model_name or self.config.get("default_embedding_model", DEFAULT_EMBEDDING_MODEL)

        self.embedding_trainer.base_model = _base_embedding_model
        self.embedding_trainer.prepare_data(texts=texts) # Uses default test_size

        default_training_config = {
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_epochs": 1, # Keep low for example
        }
        if training_args_override:
            default_training_config.update(training_args_override)
        self.embedding_trainer.configure_training(**default_training_config)

        # The model_description is important for the registry
        description = f"Custom fine-tuned embedding model '{model_name}' based on {_base_embedding_model}."

        training_start_time = datetime.now()
        best_model_path = self.embedding_trainer.train(model_name=model_name, model_description=description)
        training_duration_seconds = (datetime.now() - training_start_time).total_seconds()

        logger.info(f"Fine-tuned embedding model saved to {best_model_path}. Registered as '{model_name}'.")

        # Retrieve the ModelVersion object created by EmbeddingTrainer via ModelRegistry
        # This assumes EmbeddingTrainer registers it and we can fetch it.
        # ModelRegistry.get_model_version needs model_name and model_type.
        # EmbeddingTrainer should register with model_type="embedding".
        model_version_obj = self.model_registry.get_model_version(model_name=model_name, model_type="embedding", version_str="latest") # Assuming "latest" gets the one just trained

        version_str_logged = model_version_obj.version if model_version_obj else "unknown"
        eval_metrics_logged = model_version_obj.metadata.get("eval_metrics", {}) if model_version_obj else {}

        self.track_experiment(
            experiment_name=f"embedding_finetune_{model_name}_{version_str_logged}",
            params={
                "model_name": model_name,
                "version": version_str_logged,
                "base_model": _base_embedding_model,
                "num_texts": len(texts),
                "training_args": default_training_config, # Config used for the trainer
            },
            metrics={
                "eval_metrics": eval_metrics_logged, # Metrics from EmbeddingTrainer's eval
                "training_duration_seconds": training_duration_seconds,
            },
            experiment_type="embedding_fine_tuning"
        )
        return best_model_path


    # --- RAFT Enhancements ---
    def track_experiment(self, experiment_name: str, params: Dict, metrics: Dict, experiment_type: str = "raft_cycle"):
        """
        Tracks experiments for RAFT, logging to W&B if enabled, otherwise to a local dictionary.
        Manages a persistent W&B run for the RAFTSystem instance if W&B is active.
        """
        timestamp_str = datetime.now().isoformat()

        if self.use_wandb and wandb:
            if self.current_wandb_run is None:
                try:
                    self.current_wandb_run = wandb.init(
                        project=self.wandb_project,
                        name=f"RAFTSystemRun-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        config=self.config, # Log base RAFT config
                        reinit=True # Allow reinit if a run was ended prematurely
                    )
                    logger.info(f"Initialized W&B run: {self.current_wandb_run.name} (ID: {self.current_wandb_run.id})")
                except Exception as e:
                    logger.error(f"Failed to initialize W&B run: {e}. Falling back to local tracking for this session.")
                    self.use_wandb = False # Disable for session to avoid repeated errors

            if self.use_wandb and self.current_wandb_run: # Check again in case init failed
                # Log experiment as a distinct step or nested structure within the run
                log_data = {
                    f"experiments/{experiment_name}/params": params,
                    f"experiments/{experiment_name}/metrics": metrics,
                    f"experiments/{experiment_name}/timestamp": timestamp_str,
                    f"experiments/{experiment_name}/type": experiment_type,
                }
                # Flatten metrics and params for easier W&B table/charting if simple
                flat_log_data = {}
                for k, v in params.items():
                    flat_log_data[f"{experiment_name}_param_{k}"] = v
                for k, v in metrics.items():
                    flat_log_data[f"{experiment_name}_metric_{k}"] = v
                flat_log_data[f"{experiment_name}_timestamp"] = timestamp_str
                flat_log_data[f"{experiment_name}_type"] = experiment_type

                self.current_wandb_run.log(flat_log_data) # Log flattened data
                logger.info(f"Logged experiment '{experiment_name}' to W&B run '{self.current_wandb_run.name}'.")


        # Always log to local experiment_tracking as a fallback or for quick inspection
        if experiment_name not in self.experiment_tracking:
            self.experiment_tracking[experiment_name] = []

        self.experiment_tracking[experiment_name].append({
            "params": params,
            "metrics": metrics,
            "timestamp": timestamp_str,
            "type": experiment_type,
            "logged_to_wandb": self.use_wandb and bool(self.current_wandb_run)
        })
        logger.info(f"Tracked experiment locally: {experiment_name}")

    def close_wandb_run(self):
        """Closes the current W&B run if active."""
        if self.use_wandb and wandb and self.current_wandb_run:
            self.current_wandb_run.finish()
            logger.info(f"Closed W&B run: {self.current_wandb_run.name}")
            self.current_wandb_run = None


    def update_model_performance(self, model_name: str, query_type: str, metrics: Dict):
        """Updates performance analytics for a given model."""
        if model_name not in self.model_performance_analytics:
            self.model_performance_analytics[model_name] = {}
        if query_type not in self.model_performance_analytics[model_name]:
            self.model_performance_analytics[model_name][query_type] = []
        self.model_performance_analytics[model_name][query_type].append(metrics)
        logger.info(f"Updated performance for model {model_name} on query type {query_type}")

    # --- Reinforcement Learning Integration ---
    def _calculate_reward(self,
                          query: str,
                          response: str,
                          domain: Optional[str],
                          feedback_score: Optional[float] = None, # e.g., -1.0 to 1.0 from user
                          response_time_ms: Optional[float] = None,
                          retrieved_context_relevance: Optional[float] = None) -> float: # 0.0 to 1.0
        """
        Calculates a reward score for a given query-response pair, considering multiple factors.
        The reward should be designed to guide the RL agent towards desired outcomes.
        """
        # Define weights for different reward components (configurable)
        reward_weights = self.config.get("rl_reward_weights", {
            "user_feedback": 0.5,
            "response_quality_heuristic": 0.2,
            "response_time_penalty": -0.1, # Negative weight for penalty
            "context_relevance": 0.2
        })

        total_reward = 0.0

        # Component 1: User Feedback
        if feedback_score is not None:
            total_reward += feedback_score * reward_weights.get("user_feedback", 0.5)

        # Component 2: Response Quality Heuristic (example)
        # More advanced: use a separate model to score response quality, or ROUGE vs. retrieved context.
        quality_heuristic_score = 0.0
        if response and len(response) > 20: # Basic check for non-trivial response
            quality_heuristic_score += 0.5
        if response and query.lower().split()[0] in response.lower(): # Very basic relevance check
             quality_heuristic_score += 0.3
        total_reward += min(quality_heuristic_score, 1.0) * reward_weights.get("response_quality_heuristic", 0.2)

        # Component 3: Response Time Penalty (example)
        # Penalize if response time is too high (e.g., > 5 seconds)
        if response_time_ms is not None:
            time_penalty_threshold_ms = self.config.get("rl_response_time_penalty_threshold_ms", 5000)
            if response_time_ms > time_penalty_threshold_ms:
                # Penalty could be scaled based on how much it exceeds threshold
                penalty = (response_time_ms - time_penalty_threshold_ms) / time_penalty_threshold_ms
                total_reward += max(-1.0, -penalty) * abs(reward_weights.get("response_time_penalty", -0.1)) # Ensure penalty is negative

        # Component 4: Retrieved Context Relevance (if available)
        if retrieved_context_relevance is not None: # Assuming a score from 0 to 1
            total_reward += retrieved_context_relevance * reward_weights.get("context_relevance", 0.2)

        # Normalize reward to a typical range if needed, e.g., [-1, 1] or [0, 1]
        # The current sum could exceed these; specific RL algos have different reward scale preferences.
        # For now, let's assume the RL agent can handle varied reward scales.
        # Clipping can be useful:
        # total_reward = max(-1.0, min(1.0, total_reward))

        # TODO: If 'response' is an ensembled response, future reward components might specifically
        #       evaluate ensemble quality (e.g., coherence of combined text, diversity, cost of N calls).
        #       The current heuristics (length, keyword match) will apply to the final string.

        logger.debug(f"RL Reward Calculation: UserFeedback={feedback_score}, QualityHeuristic={quality_heuristic_score}, "
                     f"ResponseTimeMs={response_time_ms}, ContextRelevance={retrieved_context_relevance} -> TotalReward={total_reward:.3f}")
        return total_reward

    def integrate_reinforcement_learning_step(
        self,
        state: Dict[str, Any], # Represents the state before an action was taken
        action: Dict[str, Any], # Represents the action taken (e.g., model selected, fine-tuning params)
        reward: float,          # The calculated reward for the state-action pair
        next_state: Optional[Dict[str, Any]] = None # The state after the action (if applicable for the RL algo)
    ):
        """
        Processes a single experience tuple (state, action, reward, next_state) for RL.
        This method would typically:
        1. Store this experience in a replay buffer.
        2. Trigger an update of the RL agent's policy/value function.
        """
        if not self.config.get("reinforcement_learning_integration"):
            return

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        experiment_name = f"rl_experience_{action.get('selected_model', 'unknown_model')}_{timestamp}"

        logger.info(f"RL Step ({experiment_name}): Processing experience. Reward: {reward:.3f}")
        logger.debug(f"RL State: {state}")
        logger.debug(f"RL Action: {action}")
        logger.debug(f"RL Next State: {next_state}")

        # 1. Store experience (conceptual - would go into a ReplayBuffer)
        # self.rl_replay_buffer.add(state, action, reward, next_state, done_flag)
        # For now, we can log it or store in a simple list for demonstration.
        if not hasattr(self, 'rl_experiences'): self.rl_experiences = []
        self.rl_experiences.append({
            "state": state, "action": action, "reward": reward,
            "next_state": next_state, "timestamp": timestamp
        })

        # 2. Trigger RL Agent Update (conceptual)
        # if hasattr(self, 'rl_agent') and self.rl_agent.is_ready_to_train(len(self.rl_experiences)):
        #     training_loss = self.rl_agent.train_step(self.rl_experiences) # or samples from buffer
        #     logger.info(f"RL Agent training step performed. Loss: {training_loss}")
        #     self.track_experiment("rl_agent_training", {"batch_size": self.rl_agent.batch_size}, {"loss": training_loss}, "rl_training")

        # Simplified heuristic update (as before, but using the passed reward directly)
        selected_model_name = action.get("selected_model_name")
        if selected_model_name:
            if selected_model_name not in self.model_performance_analytics:
                self.model_performance_analytics[selected_model_name] = {"rl_preference_score": 0.0, "rl_updates": 0, "cumulative_reward": 0.0}

            analytics = self.model_performance_analytics[selected_model_name]
            current_pref = analytics.get("rl_preference_score", 0.0)
            num_updates = analytics.get("rl_updates", 0)
            cumulative_reward = analytics.get("cumulative_reward", 0.0)

            learning_rate = self.config.get("rl_heuristic_learning_rate", 0.01)
            # Update preference based on whether reward is positive or negative
            # This simple heuristic assumes reward is centered around 0 for good/bad.
            # If reward is [0,1], then (reward - 0.5) is a common adjustment.
            new_pref = current_pref + learning_rate * reward

            analytics["rl_preference_score"] = new_pref
            analytics["rl_updates"] = num_updates + 1
            analytics["cumulative_reward"] = cumulative_reward + reward
            avg_reward = analytics["cumulative_reward"] / analytics["rl_updates"]

            logger.info(f"RL Heuristic Update: Model '{selected_model_name}' preference -> {new_pref:.3f} (Avg Reward: {avg_reward:.3f} over {analytics['rl_updates']} updates).")

        # Log RL step details more comprehensively
        self.track_experiment(
            experiment_name=experiment_name,
            params={
                "state_summary": str(state)[:200], # Summary to avoid overly long logs
                "action": action,
            },
            metrics={
                "reward": reward,
                "current_rl_preference_score": new_pref if selected_model_name else None,
                "rl_updates_for_model": analytics.get("rl_updates") if selected_model_name and analytics else None,
            },
            experiment_type="rl_experience_step"
        )

    # Placeholder for the old method name if it's called elsewhere, can be removed if not.
    def integrate_reinforcement_learning(self):
        """Placeholder for general RL integration. Specific updates are done via `integrate_reinforcement_learning_step`."""
        if self.config.get("reinforcement_learning_integration"):
            logger.info("Reinforcement learning integration is enabled. Call 'integrate_reinforcement_learning_step' after generation and feedback.")
        else:
            logger.info("Reinforcement learning integration is disabled in config.")


if __name__ == "__main__":
    # Example Usage (Conceptual - requires data and proper model paths)
    logger.info("Initializing RAFT System...")
    # Create dummy config for example
    if not os.path.exists("config"): os.makedirs("config")
    with open("config/raft_config.yaml", "w") as f:
        yaml.dump({
            "default_base_llm": "gpt2", # Small model for testing
            "default_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "intelligent_routing": True,
            "continuous_learning": True,
            "reinforcement_learning_integration": False,
        }, f)

    # Pass sandbox_test_mode=True for quick initialization without actual model loading
    raft_sys = RAFTSystem(config_path="config/raft_config.yaml", sandbox_test_mode=True)
    logger.info("RAFT System Initialized (Sandbox Mode).")

    # 1. Agentic RAG
    # sample_query = "What are the recent advancements in AI-driven drug discovery?"
    # sample_domain = "r_and_d" # research and development
    # logger.info(f"\n--- Testing Agentic RAG for domain: {sample_domain} ---")
    # response = raft_sys.agentic_rag_generate(query=sample_query, domain=sample_domain)
    # logger.info(f"Query: {sample_query}\nResponse: {response}")

    # 2. Custom Embedding Fine-Tuning
    # logger.info("\n--- Testing Custom Embedding Fine-Tuning ---")
    # # Create dummy data for embedding fine-tuning
    # if not os.path.exists("data/custom_data"): os.makedirs("data/custom_data")
    # dummy_embedding_texts = [
    #     "This is a document about sustainable energy solutions.",
    #     "Exploring the future of solar power technology.",
    #     "Wind turbines and their impact on the environment.",
    #     "Geothermal energy as a viable alternative source.",
    #     "Innovations in battery storage for renewable energy."
    # ] * 20 # Make it a bit larger for training

    # custom_embedding_model_name = "my_custom_energy_embeddings"
    # try:
    #     raft_sys.fine_tune_embedding_model_on_custom_data(
    #         texts=dummy_embedding_texts,
    #         model_name=custom_embedding_model_name,
    #         base_model_name="sentence-transformers/all-MiniLM-L6-v2", # Specify base for clarity
    #         training_args_override={"num_epochs": 1} # Minimal epochs for example
    #     )
    #     logger.info(f"Successfully fine-tuned and registered embedding model: {custom_embedding_model_name}")
    #     # Now, this model could theoretically be used by retrieve_documents if logic is added
    #     # to select it from the model_registry.
    # except Exception as e:
    #     logger.error(f"Error during embedding fine-tuning example: {e}")


    # # 3. Custom LLM Fine-Tuning
    # logger.info("\n--- Testing Custom LLM Fine-Tuning ---")
    # # Create dummy data for LLM fine-tuning
    # dummy_llm_finetune_data = os.path.join("data/custom_data", "llm_finetune_corpus.txt")
    # with open(dummy_llm_finetune_data, "w", encoding="utf-8") as f:
    #     for i in range(50): # Small dataset for example
    #         f.write(f"This is example sentence number {i} for fine-tuning a language model. It talks about custom topics.\n")
    #         f.write(f"Another example sentence about specific domain knowledge for sentence {i}.\n")

    # custom_llm_model_name = "my_custom_domain_llm"
    # try:
    #     # Ensure the base model for fine-tuning (e.g., 'gpt2') is small enough for local tests
    #     # or use a domain-specific one if already "downloaded" or available.
    #     # For this example, we'll assume 'gpt2' is the base.
    #     # If PRE_TUNED_DOMAINS had a 'default_base_llm' key pointing to 'gpt2', it would also work.
    #     raft_sys.fine_tune_llm_on_custom_data(
    #         dataset_path=dummy_llm_finetune_data,
    #         model_name=custom_llm_model_name,
    #         base_model_name_or_path="gpt2", # Explicitly use gpt2 as base
    #         training_args_override={
    #             "num_train_epochs": 1,
    #             "per_device_train_batch_size": 1,
    #             "logging_steps": 10,
    #             "save_steps": 50 # Save more frequently for small dataset
    #         }
    #     )
    #     logger.info(f"Successfully fine-tuned and registered LLM: {custom_llm_model_name}")

    #     # Test generation with the newly fine-tuned model
    #     if custom_llm_model_name in raft_sys.global_llm_registry.list_models():
    #         logger.info(f"\n--- Testing newly fine-tuned LLM: {custom_llm_model_name} ---")
    #         tuned_response = raft_sys.agentic_rag_generate(
    #             query="Tell me about custom topics.",
    #             domain=custom_llm_model_name # Use the name of the fine-tuned model as 'domain'
    #         )
    #         logger.info(f"Query: Tell me about custom topics.\nResponse from {custom_llm_model_name}: {tuned_response}")
    #     else:
    #         logger.warning(f"Fine-tuned LLM {custom_llm_model_name} not found in registry after tuning.")

    # except Exception as e:
    #     logger.error(f"Error during LLM fine-tuning example: {e}")

    logger.info("\nRAFT System example run complete (fine-tuning sections commented out).")
