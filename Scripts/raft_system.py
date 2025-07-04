"""
Advanced Agentic RAFT (Retrieval-Augmented Fine-Tuning) System.

This module implements the core RAFT mechanism, integrating Agentic RAG
with fine-tuning capabilities for domain-specific LLM optimization.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

# Conditionally import heavy libraries
torch = None
yaml = None
AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments = None, None, None, None
SentenceTransformer = None

# from Scripts.models.registry import ModelRegistry, ModelVersion # Deferred
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
        use_wandb: bool = False, # TODO: Integrate wandb from training pipeline
        wandb_project: str = "raft-system"
    ):
        self.model_registry = ModelRegistry() # Uses the global instance for now
        self.global_llm_registry = global_llm_registry
        self.config = self._load_config(config_path)
        self.custom_finetuning_output_dir = custom_finetuning_output_dir
        os.makedirs(self.custom_finetuning_output_dir, exist_ok=True)
        self.sandbox_test_mode = kwargs.get("sandbox_test_mode", False)

        if not self.sandbox_test_mode:
            global torch, yaml, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, SentenceTransformer
            # Dynamically import heavy libraries only when not in sandbox mode
            import torch
            import yaml as pyyaml_module
            yaml = pyyaml_module
            from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM, AutoTokenizer as HfAutoTokenizer, Trainer as HfTrainer, TrainingArguments as HfTrainingArguments
            AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments = HfAutoModelForCausalLM, HfAutoTokenizer, HfTrainer, HfTrainingArguments
            from sentence_transformers import SentenceTransformer as STrans
            SentenceTransformer = STrans

            # Dynamically import local heavy modules
            from Scripts.models.training.pipeline import EmbeddingTrainer
            from Scripts.llm.service import LLMService, HuggingFaceLLM, llm_registry as global_llm_registry
            from Scripts.active_learning.learner import ActiveLearner
            from Scripts.knowledge_graph.graph import KnowledgeGraph
            from Scripts.models.registry import ModelRegistry as RealModelRegistry, ModelVersion as RealModelVersion # Alias to avoid clash

            self.model_registry = RealModelRegistry()
            self.ModelVersion_class = RealModelVersion
            self.global_llm_registry = global_llm_registry
            self.embedding_trainer = EmbeddingTrainer( # Real EmbeddingTrainer
                output_dir=embedding_finetuning_output_dir,
                model_registry=self.model_registry,
                use_wandb=use_wandb,
                wandb_project=wandb_project
            )
            self.active_learner = ActiveLearner() # Real ActiveLearner
            self.knowledge_graph = KnowledgeGraph() # Real KnowledgeGraph
            LLMService_class = LLMService
            HuggingFaceLLM_class = HuggingFaceLLM
        else: # Sandbox mode
            self.model_registry = _global_dummy_model_registry
            self.ModelVersion_class = DummyModelVersion
            self.global_llm_registry = _global_dummy_llm_registry
            self.embedding_trainer = DummyEmbeddingTrainer() # Dummy
            self.active_learner = DummyActiveLearner() # Dummy
            self.knowledge_graph = DummyKnowledgeGraph() # Dummy
            LLMService_class = DummyLLMService
            HuggingFaceLLM_class = DummyLLMService # Use DummyLLMService as placeholder for HuggingFaceLLM type
            logger.info("[Sandbox Mode] Using dummy/placeholder components for RAFTSystem.")

        self.LLMService_class = LLMService_class
        self.HuggingFaceLLM_class = HuggingFaceLLM_class
        self.config = self._load_config(config_path)
        self._initialize_pre_tuned_models() # Uses self.HuggingFaceLLM_class if not sandbox
        self.model_performance_analytics = {}
        self.experiment_tracking = {}

    def _ensure_libs_loaded(self):
        """Ensures heavy libraries are loaded if not in sandbox mode and not already loaded."""
        if not self.sandbox_test_mode and (torch is None or AutoModelForCausalLM is None or SentenceTransformer is None or yaml is None):
            global torch, yaml, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, SentenceTransformer
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
            logger.info("Heavy libraries dynamically loaded.")


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

        selected_llm = self._intelligent_route_model(query, domain)
        logger.info(f"Using LLM: {selected_llm} for generation.")
        llm_instance = self.global_llm_registry.get_model(selected_llm)

        generation_params = {
            "max_length": llm_options.get("max_length", 250), # Example parameters
            "temperature": llm_options.get("temperature", 0.7)
        }
        response = llm_instance.generate(prompt, **generation_params)

        # Continuous learning: collect feedback (simplified)
        if self.config.get("continuous_learning"):
            self.active_learner.add_interaction_data(query, context, response, domain=domain) # TODO: Add feedback mechanism

        return response

    def _intelligent_route_model(self, query: str, domain: Optional[str] = None, complexity: Optional[str] = None) -> str:
        """
        Intelligently selects the best LLM based on query, domain, complexity, and performance.
        """
        if not self.config.get("intelligent_routing"):
            return domain or self.config.get("default_base_llm", DEFAULT_BASE_LLM)

        # Basic routing: prefer domain-specific model if available
        if domain and domain in self.global_llm_registry.list_models():
            logger.info(f"Routing to domain-specific model: {domain}")
            return domain

        # TODO: Implement more sophisticated routing based on:
        # 1. Query analysis (e.g., using a small model to classify query type)
        # 2. Complexity assessment
        # 3. Model performance analytics (self.model_performance_analytics)
        # 4. Dynamic model ensembling (if multiple models are suitable)

        default_model = self.config.get("default_base_llm", DEFAULT_BASE_LLM)
        logger.info(f"Routing to default model: {default_model}")
        return default_model

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
        trainer.train()
        trainer.save_model(output_model_dir)
        tokenizer.save_pretrained(output_model_dir)
        logger.info(f"Fine-tuned LLM saved to {output_model_dir}")

        # Register the fine-tuned model
        # TODO: Implement proper model versioning and metadata (metrics, etc.)
        # For now, we'll just add it to the global LLM registry for use
        self.global_llm_registry.register_model(model_name, HuggingFaceLLM(model_name=output_model_dir))
        logger.info(f"Fine-tuned model '{model_name}' registered.")

        # Create a dummy ModelVersion for now, this needs to be more robust
        # This should ideally come from the model registry's own mechanisms
        model_version_info = {
            "base_model": _base_model_name,
            "training_date": "", # Placeholder
            "eval_metrics": {} # Placeholder
        }
        model_version = ModelVersion(
            name=model_name,
            version="1.0.0-custom",
            path=output_model_dir,
            metadata=model_version_info,
            description=f"Custom fine-tuned LLM based on {_base_model_name} using {os.path.basename(dataset_path)}"
        )
        # self.model_registry.register_model(model_version) # This was for embedding models, need similar for LLMs
        return model_version # Placeholder return


    async def execute_raft_cycle(self, query: str, domain: Optional[str] = None, search_system: Optional[Any] = None) -> Dict[str, Optional[str]]: # search_system is AdvancedSearchSystem
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
        best_model_path = self.embedding_trainer.train(model_name=model_name, model_description=description)
        logger.info(f"Fine-tuned embedding model saved to {best_model_path}. Registered as '{model_name}'.")
        return best_model_path


    # --- RAFT Enhancements ---
    def track_experiment(self, experiment_name: str, params: Dict, metrics: Dict):
        """Tracks experiments for RAFT."""
        # TODO: Integrate with a proper experiment tracking tool (e.g., MLflow, W&B)
        self.experiment_tracking[experiment_name] = {"params": params, "metrics": metrics, "timestamp": ""} # Add timestamp
        logger.info(f"Tracked experiment: {experiment_name}")

    def update_model_performance(self, model_name: str, query_type: str, metrics: Dict):
        """Updates performance analytics for a given model."""
        if model_name not in self.model_performance_analytics:
            self.model_performance_analytics[model_name] = {}
        if query_type not in self.model_performance_analytics[model_name]:
            self.model_performance_analytics[model_name][query_type] = []
        self.model_performance_analytics[model_name][query_type].append(metrics)
        logger.info(f"Updated performance for model {model_name} on query type {query_type}")

    # --- Placeholder for future integration ---
    def integrate_reinforcement_learning(self):
        """Placeholder for RL integration for model optimization."""
        if self.config.get("reinforcement_learning_integration"):
            logger.info("Reinforcement learning integration is enabled (placeholder).")
            # TODO: Implement RL components (e.g., reward functions, policy updates)
            # This could involve using feedback from active_learner or other sources.
            pass


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
