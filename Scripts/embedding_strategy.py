"""
Embedding Strategy Manager for Adaptive Model Selection and other advanced strategies.
"""
import logging
from typing import Dict, Any, Optional, List
from Scripts.embeddings import EmbeddingProvider, create_embedding_provider

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingModelMeta:
    provider_name: str # e.g., "openai", "cohere", "huggingface_local"
    model_name: str    # e.g., "text-embedding-3-large", "embed-english-v3.0", "sentence-transformers/all-MiniLM-L6-v2"
    dim: int
    languages: List[str] = field(default_factory=lambda: ["en"])
    domain: Optional[str] = None # e.g., "finance", "legal"
    modality: str = "text" # "text", "image", "audio", "multimodal"
    # Add other relevant metadata: cost, speed, etc.
    provider_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


class EmbeddingStrategyManager:
    def __init__(self, model_configs: List[EmbeddingModelMeta]):
        self.model_configs = model_configs
        self.providers: Dict[str, EmbeddingProvider] = {} # Cache for initialized providers
        self._validate_configs()

    def _validate_configs(self):
        if not self.model_configs:
            raise ValueError("At least one embedding model configuration must be provided.")
        for config in self.model_configs:
            if not all([config.provider_name, config.model_name, config.dim]):
                raise ValueError(f"Invalid model config, missing essential fields: {config}")

    def _get_provider_key(self, meta: EmbeddingModelMeta) -> str:
        # Create a unique key for caching based on provider and its specific model,
        # as one provider (e.g. OpenAI) can serve multiple models.
        return f"{meta.provider_name}_{meta.model_name}"

    def get_embedding_provider(self, strategy_params: Dict[str, Any]) -> EmbeddingProvider:
        """
        Selects and initializes an embedding provider based on strategy parameters.

        Args:
            strategy_params: A dictionary containing parameters to guide selection, e.g.,
                             {"language": "en", "domain": "finance", "modality": "text", "prefer_accuracy": True}
                             Can also directly specify "provider_name" and "model_name".
        Returns:
            An initialized EmbeddingProvider.
        """

        # Direct specification takes precedence
        if "provider_name" in strategy_params and "model_name" in strategy_params:
            provider_name = strategy_params["provider_name"]
            model_name = strategy_params["model_name"]
            selected_config = next((mc for mc in self.model_configs if mc.provider_name == provider_name and mc.model_name == model_name), None)
            if not selected_config:
                raise ValueError(f"No matching model configuration found for provider '{provider_name}' and model '{model_name}'.")
        else:
            # Adaptive selection logic (placeholder for now, needs more sophistication)
            # This should filter self.model_configs based on language, domain, modality, etc.
            # and then rank them based on preference (e.g., accuracy, cost, speed).

            # Simple example: filter by language and modality, then pick the first match.
            lang = strategy_params.get("language", "en")
            modality = strategy_params.get("modality", "text")
            domain = strategy_params.get("domain")

            possible_configs = [
                mc for mc in self.model_configs
                if lang in mc.languages and mc.modality == modality
            ]

            if domain: # If domain is specified, try to find a domain-specific model
                domain_specific_configs = [pc for pc in possible_configs if pc.domain == domain]
                if domain_specific_configs:
                    possible_configs = domain_specific_configs

            if not possible_configs:
                logger.warning(f"No model found for lang='{lang}', modality='{modality}', domain='{domain}'. Falling back to first available.")
                selected_config = self.model_configs[0]
            else:
                selected_config = possible_configs[0] # Simplistic: pick the first suitable
                logger.info(f"Selected model based on strategy: {selected_config.provider_name}/{selected_config.model_name}")

        provider_key = self._get_provider_key(selected_config)
        if provider_key not in self.providers:
            logger.info(f"Initializing embedding provider: {selected_config.provider_name} with model {selected_config.model_name}")
            # Merge general provider_kwargs with any specific ones for this call
            provider_init_kwargs = selected_config.provider_kwargs.copy()
            provider_init_kwargs.update(strategy_params.get("provider_kwargs", {}))

            self.providers[provider_key] = create_embedding_provider(
                provider=selected_config.provider_name,
                model=selected_config.model_name,
                **provider_init_kwargs
            )
            # Ensure dimension matches if not already set by create_embedding_provider
            # This is more of a check, as the provider should know its dim.
            actual_dim = self.providers[provider_key].get_embedding_dim()
            if actual_dim != selected_config.dim:
                logger.warning(f"Dimension mismatch for {provider_key}: config says {selected_config.dim}, provider says {actual_dim}. Using provider's.")
                selected_config.dim = actual_dim # Update our meta to match reality

        return self.providers[provider_key]

    # --- Placeholders for other embedding strategies ---

    def get_multimodal_embedding_provider(self, strategy_params: Dict[str, Any]) -> Optional[EmbeddingProvider]:
        """Selects a provider suitable for multimodal embeddings."""
        logger.warning("Multi-modal embedding strategy is a placeholder.")
        # Similar to get_embedding_provider, but filters for modality="multimodal" or "image", "audio"
        # and would likely use specialized multimodal embedding models (e.g., CLIP based).
        params = strategy_params.copy()
        params["modality"] = "multimodal" # or "image", "audio" depending on need
        try:
            return self.get_embedding_provider(params)
        except ValueError: # If no direct multimodal, try to find one by other means or return None
            logger.error("No suitable multimodal provider found with current configs.")
            return None


    def get_compressed_embeddings(self, texts: List[str], provider: EmbeddingProvider, target_dim: Optional[int] = None) -> np.ndarray:
        """Generates embeddings and then applies compression."""
        logger.warning("Embedding compression is a placeholder.")
        # embeddings = await provider.generate_embeddings(texts) # This needs to be async if provider is
        # embeddings_np = np.array(embeddings)
        # 1. Generate full embeddings
        # 2. Apply compression technique (e.g., PCA, Matryoshka Embeddings, quantization)
        # return compressed_embeddings_np
        raise NotImplementedError("Embedding compression not implemented.")

    def get_fine_tuned_embedding_provider(self, model_id: str, base_provider_info: EmbeddingModelMeta) -> EmbeddingProvider:
        """
        Loads a fine-tuned embedding model.
        The 'model_id' could be a path to a local fine-tuned SentenceTransformer model,
        or an identifier in a model registry.
        """
        logger.warning(f"Fine-tuned embedding model loading for '{model_id}' is a placeholder.")
        # This would involve:
        # 1. Looking up model_id in a registry (e.g., the main ModelRegistry from RAFT system)
        # 2. Determining its type (e.g. local SentenceTransformer path)
        # 3. Instantiating an EmbeddingProvider, possibly a generic HuggingFace one,
        #    pointing to this local model path.
        # For now, create a new provider instance based on base_provider_info but override model name

        provider_key = f"{base_provider_info.provider_name}_finetuned_{model_id.replace('/', '_')}"
        if provider_key not in self.providers:
            logger.info(f"Initializing fine-tuned provider: {provider_key} (path/id: {model_id})")

            # This assumes fine-tuned models are SentenceTransformer compatible
            # and can be loaded by a generic HuggingFace local provider.
            # We might need a specific "local_hf" provider type in create_embedding_provider.

            # Example: if base_provider_info.provider_name was 'huggingface_local'
            # or if we add such a provider type that just takes a path.
            try:
                self.providers[provider_key] = create_embedding_provider(
                    provider=base_provider_info.provider_name, # Or a specific "local_finetuned_transformer" provider
                    model=model_id, # This would be the path to the fine-tuned model
                    **(base_provider_info.provider_kwargs or {})
                )
            except ValueError as e: # If provider type doesn't support direct path model like this
                 logger.error(f"Could not create provider for fine-tuned model {model_id} based on {base_provider_info.provider_name}: {e}")
                 raise # Or return a default
        return self.providers[provider_key]

# Example Usage:
# model_metadata = [
#     EmbeddingModelMeta(provider_name="openai", model_name="text-embedding-3-large", dim=3072, languages=["en","multi"], domain="general", provider_kwargs={"api_key":"..."}),
#     EmbeddingModelMeta(provider_name="cohere", model_name="embed-english-light-v3.0", dim=384, languages=["en"], domain="general", provider_kwargs={"api_key":"..."}),
#     EmbeddingModelMeta(provider_name="ollama", model_name="snowflake-arctic-embed", dim=1536, languages=["en"], domain="general"),
#     EmbeddingModelMeta(provider_name="huggingface_local", model_name="sentence-transformers/all-MiniLM-L6-v2", dim=384, languages=["en"]),
#     EmbeddingModelMeta(provider_name="huggingface_local", model_name="path/to/my_finetuned_model", dim=768, languages=["en"], domain="finance"),
# ]
# strategy_manager = EmbeddingStrategyManager(model_configs=model_metadata)
# general_provider = strategy_manager.get_embedding_provider({"language": "en"})
# finance_provider = strategy_manager.get_embedding_provider({"language": "en", "domain": "finance"})

```
