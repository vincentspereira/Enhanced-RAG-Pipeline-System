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
        """
        Selects a provider suitable for multimodal embeddings (text, image, potentially audio).
        Prioritizes models explicitly marked as 'multimodal' or 'image'.
        """
        logger.info(f"Attempting to get multimodal embedding provider with params: {strategy_params}")

        # Attempt to find a model explicitly configured for "multimodal" or "image"
        # This assumes 'clip' provider is registered with such modality in EmbeddingModelMeta

        # Try for "multimodal" first
        multimodal_params = strategy_params.copy()
        multimodal_params["modality"] = "multimodal"

        selected_config = next((
            mc for mc in self.model_configs
            if mc.modality == "multimodal" and
               (strategy_params.get("language", "en") in mc.languages if "language" in strategy_params else True) and
               (strategy_params.get("provider_name") == mc.provider_name if "provider_name" in strategy_params else True) and
               (strategy_params.get("model_name") == mc.model_name if "model_name" in strategy_params else True)
        ), None)

        if not selected_config:
            # Try for "image" modality as a fallback if direct "multimodal" not found or specified
            image_params = strategy_params.copy()
            image_params["modality"] = "image"
            selected_config = next((
                mc for mc in self.model_configs
                if mc.modality == "image" and
                   (strategy_params.get("language", "en") in mc.languages if "language" in strategy_params else True) and
                   (strategy_params.get("provider_name") == mc.provider_name if "provider_name" in strategy_params else True) and
                   (strategy_params.get("model_name") == mc.model_name if "model_name" in strategy_params else True)
            ), None)

        if selected_config:
            logger.info(f"Selected multimodal/image provider: {selected_config.provider_name} with model {selected_config.model_name}")
            # Use the main get_embedding_provider to handle caching and instantiation
            # Pass specific provider_name and model_name to ensure the correct one is chosen
            return self.get_embedding_provider({
                "provider_name": selected_config.provider_name,
                "model_name": selected_config.model_name,
                "modality": selected_config.modality, # Pass modality for clarity
                "provider_kwargs": strategy_params.get("provider_kwargs", selected_config.provider_kwargs)
            })
        else:
            logger.error(f"No suitable multimodal or image embedding provider found for strategy: {strategy_params} with current model_configs.")
            return None


    async def get_compressed_embeddings(self,
                                        texts: List[str],
                                        base_provider_strategy_params: Dict[str, Any],
                                        compression_technique: str = "scalar_quantization_int8",
                                        **compression_kwargs) -> List[Any]: # Return type matches _apply_compression
        """
        Generates full embeddings using a specified base provider and then applies compression.
        Returns a list of compressed data structures (e.g., (quantized_vector, params) tuples for scalar quantization).
        """
        if not texts:
            return []

        # 1. Get the base embedding provider
        base_provider = self.get_embedding_provider(base_provider_strategy_params)

        # 2. Generate full float embeddings
        # EmbeddingProvider.generate_embeddings is async
        float_embeddings = await base_provider.generate_embeddings(texts)

        if not float_embeddings:
            logger.warning(f"Base provider {base_provider.__class__.__name__} returned no embeddings for compression. Input texts count: {len(texts)}")
            return []

        # 3. Apply compression
        # _apply_compression is also async
        compressed_data = await self._apply_compression(
            float_embeddings,
            technique=compression_technique,
            **compression_kwargs
        )

        logger.info(f"Successfully generated and compressed {len(texts)} texts using technique '{compression_technique}'.")
        return compressed_data

    async def _apply_compression(self, embeddings: List[List[float]], technique: str = "scalar_quantization_int8", **kwargs) -> List[List[Any]]: # Output type might change
        """
        Applies a specified compression technique to the given embeddings.
        Placeholder for actual compression logic.
        """
        if not embeddings:
            return [] # Return type should be List[Tuple[List[int], Dict[str, float]]] for quantization

        logger.info(f"Attempting to apply '{technique}' compression to {len(embeddings)} embeddings.")

        if technique == "scalar_quantization_int8":
            results_with_params = [] # Will store tuples of (quantized_vector_list, params_dict)

            for emb_float_list in embeddings:
                emb_np = np.array(emb_float_list, dtype=np.float32)

                abs_max = np.max(np.abs(emb_np))

                if abs_max < 1e-9: # Treat near-zero vectors as zero to avoid large scales
                    scale = 1.0
                    zero_point = 0
                    quantized_emb_np = np.zeros_like(emb_np, dtype=np.int8)
                else:
                    # Symmetric quantization for int8: scale = abs_max / 127.0
                    scale = abs_max / 127.0
                    zero_point = 0 # For symmetric int8, zero_point is 0

                    # Quantize: (value / scale) -> round -> clip -> cast to int8
                    # Add epsilon to scale denominator to prevent division by zero if abs_max is tiny but not zero.
                    scaled_values = emb_np / (scale + 1e-12) # Add epsilon for stability
                    rounded_values = np.round(scaled_values)
                    # Clip to ensure values are within the int8 range before casting
                    clipped_values = np.clip(rounded_values, -127, 127) # Standard range for signed int8 for this scheme
                    quantized_emb_np = clipped_values.astype(np.int8)

                quantization_params = {"scale": float(scale), "zero_point": int(zero_point)}
                results_with_params.append((quantized_emb_np.tolist(), quantization_params))

            logger.info(f"Applied scalar_quantization_int8. Returning list of (quantized_vector, params) tuples.")
            return results_with_params # Return list of tuples

        elif technique == "pca":
            target_dim = kwargs.get("target_dim")
            # PCA returns float embeddings, so the List[List[Any]] signature for _apply_compression is okay,
            # but the overall plan mentioned returning params for scalar quantization.
            # For PCA, the "params" would be the PCA model itself, which is more complex to return here.
            # This function is primarily designed for techniques that modify the vector and have simple per-vector params.
            if not target_dim:
                logger.warning("PCA compression requested but 'target_dim' not provided. Returning original embeddings.")
                return embeddings
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=target_dim)
                embeddings_np = np.array(embeddings)
                compressed_embeddings_np = pca.fit_transform(embeddings_np)
                logger.info(f"Applied PCA compression to target_dim={target_dim}. Original_dim={embeddings_np.shape[1]}.")
                # TODO: The PCA model (pca.components_, pca.mean_) would need to be saved/managed.
                return compressed_embeddings_np.tolist()
            except ImportError:
                logger.error("scikit-learn not installed. Cannot perform PCA compression.")
                return embeddings # Return original
            except Exception as e_pca:
                logger.error(f"Error during PCA compression: {e_pca}")
                return embeddings


        else:
            logger.warning(f"Unknown or not fully implemented compression technique: {technique}. Returning original embeddings.")
            return embeddings # Return original embeddings if technique is unknown/unsupported

    def get_fine_tuned_embedding_provider(self, model_path_or_id: str, original_model_meta: Optional[EmbeddingModelMeta] = None) -> EmbeddingProvider:
        """
        Loads a fine-tuned embedding model, typically a local SentenceTransformer model.

        Args:
            model_path_or_id: Path to the local fine-tuned model directory or a registered ID
                              that resolves to such a path.
            original_model_meta: Optional metadata of the base model that was fine-tuned.
                                 Used for fallbacks or if some provider_kwargs are needed.
        Returns:
            An initialized EmbeddingProvider for the fine-tuned model.
        """
        # Construct a unique key for caching this provider instance
        # Sanitize model_path_or_id for use in key, e.g., replace slashes
        sanitized_model_id = model_path_or_id.replace('/', '_').replace('\\', '_')
        provider_key = f"local_hf_finetuned_{sanitized_model_id}"

        if provider_key not in self.providers:
            logger.info(f"Initializing fine-tuned embedding provider from path/ID: {model_path_or_id}")

            # Default provider_kwargs from original model if available, or empty dict
            provider_kwargs = original_model_meta.provider_kwargs.copy() if original_model_meta and original_model_meta.provider_kwargs else {}

            # Assuming the fine-tuned model is a local SentenceTransformer model,
            # we use the "local_hf" provider type.
            # The 'model_path' argument for LocalHuggingFaceEmbeddings is crucial.
            try:
                self.providers[provider_key] = create_embedding_provider(
                    provider="local_hf",
                    model_path=model_path_or_id, # Pass the path directly
                    **provider_kwargs # Pass other relevant args like batch_size, device
                )
                logger.info(f"Successfully initialized fine-tuned provider '{provider_key}' for model at '{model_path_or_id}'.")
            except Exception as e:
                logger.error(f"Failed to create provider for fine-tuned model at '{model_path_or_id}': {e}")
                # Optionally, could fall back to the original base model provider if original_model_meta is provided
                if original_model_meta:
                    logger.warning(f"Falling back to original model provider for {original_model_meta.provider_name}/{original_model_meta.model_name}")
                    return self.get_embedding_provider({
                        "provider_name": original_model_meta.provider_name,
                        "model_name": original_model_meta.model_name
                    })
                raise # Re-raise if no fallback

        return self.providers[provider_key]

# Example Usage:
# default_model_configs = [
#     EmbeddingModelMeta(provider_name="ollama", model_name="snowflake-arctic-embed", dim=1536, languages=["en"], modality="text"),
#     EmbeddingModelMeta(provider_name="local_hf", model_name="sentence-transformers/all-MiniLM-L6-v2", dim=384, languages=["en"], modality="text"),
#     # Example CLIP model configuration
#     EmbeddingModelMeta(provider_name="clip", model_name="clip-ViT-B-32", dim=512, languages=["en"], modality="multimodal", provider_kwargs={}),
#     EmbeddingModelMeta(provider_name="clip", model_name="clip-ViT-L-14", dim=768, languages=["en"], modality="multimodal", provider_kwargs={}),
#     # Example for a text-only use of a CLIP model if provider supports it or if it's a text tower
#     EmbeddingModelMeta(provider_name="clip", model_name="clip-ViT-B-32-text", dim=512, languages=["en"], modality="text", provider_kwargs={}),
#     # Example for an image-only use
#     EmbeddingModelMeta(provider_name="clip", model_name="clip-ViT-B-32-image", dim=512, languages=["en"], modality="image", provider_kwargs={}),
# ]
# strategy_manager = EmbeddingStrategyManager(model_configs=default_model_configs)
# text_provider = strategy_manager.get_embedding_provider({"language": "en", "modality": "text"})
# image_provider = strategy_manager.get_multimodal_embedding_provider({"modality": "image"}) # or "multimodal"

    @staticmethod
    def dequantize_vector(quantized_vector: List[int], params: Dict[str, float]) -> List[float]:
        """
        Dequantizes an int8 vector back to its approximate float representation
        using the provided scale and zero_point.
        """
        if not isinstance(quantized_vector, list) or not all(isinstance(x, int) for x in quantized_vector):
            raise TypeError("quantized_vector must be a list of integers.")
        if not isinstance(params, dict) or "scale" not in params or "zero_point" not in params:
            raise ValueError("params must be a dict with 'scale' and 'zero_point' keys.")

        scale = params["scale"]
        zero_point = params["zero_point"] # Typically 0 for our symmetric int8 quantization

        if scale == 0: # Should ideally not happen if abs_max was > 0 during quantization
            logger.warning("Dequantization scale is zero, returning zero vector.")
            return [0.0] * len(quantized_vector)

        # Dequantize: (quantized_value - zero_point) * scale
        # For symmetric int8, zero_point is 0, so it's just quantized_value * scale
        dequantized_np = (np.array(quantized_vector, dtype=np.float32) - zero_point) * scale
        return dequantized_np.tolist()
```
