import pytest
from unittest.mock import patch, MagicMock
import asyncio

# Ensure Scripts directory is in path for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Scripts')))

from embedding_strategy import EmbeddingStrategyManager, EmbeddingModelMeta
from embeddings import EmbeddingProvider, OllamaEmbeddings, OpenAIEmbeddings, LocalHuggingFaceEmbeddings # Using existing providers for testing

# --- Fixtures ---

@pytest.fixture
def sample_model_configs():
    return [
        EmbeddingModelMeta(provider_name="ollama", model_name="model1_ollama", dim=768, languages=["en"], domain="general", modality="text"),
        EmbeddingModelMeta(provider_name="openai", model_name="model2_openai", dim=1536, languages=["en", "es"], domain="finance", modality="text", provider_kwargs={"api_key": "fake_openai_key"}),
        EmbeddingModelMeta(provider_name="local_hf", model_name="model3_local", dim=384, languages=["en"], domain="legal", modality="text", provider_kwargs={"device":"cpu"}),
        EmbeddingModelMeta(provider_name="cohere", model_name="model4_cohere_multilingual", dim=1024, languages=["multi"], modality="text", provider_kwargs={"api_key": "fake_cohere_key"}),
    ]

@pytest.fixture
def strategy_manager(sample_model_configs):
    return EmbeddingStrategyManager(model_configs=sample_model_configs)

# --- Tests for EmbeddingStrategyManager ---

def test_esm_initialization(strategy_manager, sample_model_configs):
    assert strategy_manager is not None
    assert len(strategy_manager.model_configs) == len(sample_model_configs)

def test_esm_initialization_no_configs():
    with pytest.raises(ValueError, match="At least one embedding model configuration must be provided."):
        EmbeddingStrategyManager(model_configs=[])

def test_esm_initialization_invalid_config():
    invalid_configs = [EmbeddingModelMeta(provider_name="ollama", model_name="bad", dim=0)] # Missing dim or other critical fields
    with pytest.raises(ValueError, match="Invalid model config, missing essential fields"):
         EmbeddingStrategyManager(model_configs=invalid_configs)


@patch('Scripts.embeddings.create_embedding_provider') # Patching at the source of create_embedding_provider
def test_get_embedding_provider_direct_specification(mock_create_provider, strategy_manager: EmbeddingStrategyManager):
    mock_ollama_instance = MagicMock(spec=OllamaEmbeddings)
    mock_ollama_instance.get_embedding_dim.return_value = 768 # Ensure it matches config
    mock_create_provider.return_value = mock_ollama_instance

    provider = strategy_manager.get_embedding_provider({
        "provider_name": "ollama",
        "model_name": "model1_ollama"
    })

    assert provider == mock_ollama_instance
    mock_create_provider.assert_called_once_with(
        provider="ollama",
        model="model1_ollama"
        # provider_kwargs from EmbeddingModelMeta should be passed if create_embedding_provider uses them
    )
    # Check if provider is cached
    provider_key = strategy_manager._get_provider_key(strategy_manager.model_configs[0])
    assert provider_key in strategy_manager.providers
    assert strategy_manager.providers[provider_key] == mock_ollama_instance

@patch('Scripts.embeddings.create_embedding_provider')
def test_get_embedding_provider_adaptive_selection_simple(mock_create_provider, strategy_manager: EmbeddingStrategyManager):
    mock_openai_instance = MagicMock(spec=OpenAIEmbeddings)
    mock_openai_instance.get_embedding_dim.return_value = 1536
    mock_create_provider.return_value = mock_openai_instance

    # Should pick model2_openai for finance
    provider = strategy_manager.get_embedding_provider({"language": "en", "domain": "finance", "modality": "text"})
    assert provider == mock_openai_instance
    # The actual call to create_embedding_provider will have model="model2_openai"
    # and provider="openai" due to selected_config
    args, kwargs = mock_create_provider.call_args
    assert kwargs['provider'] == "openai"
    assert kwargs['model'] == "model2_openai"
    assert kwargs['api_key'] == "fake_openai_key" # From provider_kwargs in EmbeddingModelMeta

@patch('Scripts.embeddings.create_embedding_provider')
def test_get_embedding_provider_fallback(mock_create_provider, strategy_manager: EmbeddingStrategyManager):
    # Mock the first config's provider
    mock_first_provider_instance = MagicMock(spec=OllamaEmbeddings)
    mock_first_provider_instance.get_embedding_dim.return_value = 768
    mock_create_provider.return_value = mock_first_provider_instance

    # Request a non-existent domain, should fall back to the first compatible (ollama model1)
    provider = strategy_manager.get_embedding_provider({"language": "en", "domain": "non_existent_domain", "modality": "text"})
    assert provider == mock_first_provider_instance
    args, kwargs = mock_create_provider.call_args
    assert kwargs['provider'] == "ollama"
    assert kwargs['model'] == "model1_ollama"


def test_get_embedding_provider_not_found(strategy_manager: EmbeddingStrategyManager):
    with pytest.raises(ValueError, match="No matching model configuration found"):
        strategy_manager.get_embedding_provider({"provider_name": "non_existent_provider", "model_name": "non_existent_model"})

@patch('Scripts.embeddings.create_embedding_provider')
def test_get_fine_tuned_embedding_provider(mock_create_provider, strategy_manager: EmbeddingStrategyManager):
    mock_local_hf_instance = MagicMock(spec=LocalHuggingFaceEmbeddings)
    mock_local_hf_instance.get_embedding_dim.return_value = 384 # Example
    mock_create_provider.return_value = mock_local_hf_instance

    model_path = "path/to/my_finetuned_model"

    # Get the original meta for "local_hf" to pass as original_model_meta (optional for this test)
    original_meta = next((mc for mc in strategy_manager.model_configs if mc.provider_name == "local_hf"), None)

    provider = strategy_manager.get_fine_tuned_embedding_provider(model_path, original_model_meta=original_meta)

    assert provider == mock_local_hf_instance
    mock_create_provider.assert_called_once_with(
        provider="local_hf",
        model_path=model_path,
        device="cpu" # From the sample_model_configs for model3_local
    )
    provider_key = f"local_hf_finetuned_{model_path.replace('/', '_')}"
    assert provider_key in strategy_manager.providers

@patch('Scripts.embeddings.create_embedding_provider')
def test_get_fine_tuned_embedding_provider_fallback(mock_create_provider, strategy_manager: EmbeddingStrategyManager):
    # Simulate create_embedding_provider failing for the fine-tuned path
    # and then successfully creating the original model
    original_model_config = strategy_manager.model_configs[0] # ollama/model1_ollama

    # First call (for fine-tuned) will raise an error
    # Second call (for fallback) will return a mock
    mock_original_provider_instance = MagicMock(spec=OllamaEmbeddings)
    mock_original_provider_instance.get_embedding_dim.return_value = original_model_config.dim

    mock_create_provider.side_effect = [
        ValueError("Failed to load fine-tuned model"), # First call fails
        mock_original_provider_instance # Second call (fallback) succeeds
    ]

    provider = strategy_manager.get_fine_tuned_embedding_provider(
        model_path_or_id="bad/path/finetuned",
        original_model_meta=original_model_config
    )

    assert provider == mock_original_provider_instance
    assert mock_create_provider.call_count == 2
    # First call (failed)
    assert mock_create_provider.call_args_list[0][1]['provider'] == "local_hf"
    assert mock_create_provider.call_args_list[0][1]['model_path'] == "bad/path/finetuned"
    # Second call (fallback)
    assert mock_create_provider.call_args_list[1][1]['provider'] == original_model_config.provider_name
    assert mock_create_provider.call_args_list[1][1]['model'] == original_model_config.model_name


@pytest.mark.asyncio
async def test_apply_compression_placeholder(strategy_manager: EmbeddingStrategyManager):
    sample_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    # Test scalar quantization (placeholder behavior)
    compressed_sq = await strategy_manager._apply_compression(sample_embeddings, technique="scalar_quantization_int8")
    assert len(compressed_sq) == len(sample_embeddings)
    assert isinstance(compressed_sq[0][0], int) # Should be int8, but placeholder might just cast

    # Test PCA (placeholder behavior)
    compressed_pca = await strategy_manager._apply_compression(sample_embeddings, technique="pca")
    assert compressed_pca == sample_embeddings # Placeholder returns original

    # Test unknown technique
    compressed_unknown = await strategy_manager._apply_compression(sample_embeddings, technique="unknown_tech")
    assert compressed_unknown == sample_embeddings # Placeholder returns original

@patch('Scripts.embeddings.create_embedding_provider')
def test_get_multimodal_embedding_provider(mock_create_provider):
    clip_meta = EmbeddingModelMeta(provider_name="clip", model_name="clip-ViT-B-32", dim=512, languages=["en"], modality="multimodal")
    text_meta = EmbeddingModelMeta(provider_name="local_hf", model_name="st_model", dim=384, languages=["en"], modality="text")

    manager = EmbeddingStrategyManager(model_configs=[clip_meta, text_meta])

    mock_clip_instance = MagicMock(spec=EmbeddingProvider) # Use EmbeddingProvider from Scripts.embeddings
    mock_clip_instance.get_embedding_dim.return_value = 512

    # Configure create_embedding_provider to return the mock_clip_instance when "clip" is requested
    def create_provider_side_effect(provider, model, **kwargs):
        if provider == "clip" and model == "clip-ViT-B-32":
            return mock_clip_instance
        raise ValueError(f"Unexpected provider/model for mock: {provider}/{model}")
    mock_create_provider.side_effect = create_provider_side_effect

    # Test selecting by "multimodal"
    provider_multi = manager.get_multimodal_embedding_provider({"modality": "multimodal"})
    assert provider_multi == mock_clip_instance
    mock_create_provider.assert_called_with(provider="clip", model="clip-ViT-B-32", modality="multimodal", provider_kwargs=None)

    # Test selecting by "image" (should also pick up "multimodal" if it's the best/only fit)
    provider_image = manager.get_multimodal_embedding_provider({"modality": "image"})
    assert provider_image == mock_clip_instance # Expecting it to find the same 'multimodal' one

    # Test when no suitable multimodal/image model is configured
    manager_only_text = EmbeddingStrategyManager(model_configs=[text_meta])
    provider_none = manager_only_text.get_multimodal_embedding_provider({"modality": "multimodal"})
    assert provider_none is None

# TODO: Test get_compressed_embeddings (the public method) once _apply_compression is more than a placeholder
#       and interacts with an actual embedding generation step.

```
