import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import numpy as np # Import numpy for array comparisons

# Ensure Scripts directory is in path for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Scripts')))

from embedding_strategy import EmbeddingStrategyManager, EmbeddingModelMeta
from embeddings import EmbeddingProvider, OllamaEmbeddings, OpenAIEmbeddings, LocalHuggingFaceEmbeddings

# --- Fixtures ---

@pytest.fixture
def sample_model_configs():
    return [
        EmbeddingModelMeta(provider_name="ollama", model_name="model1_ollama", dim=768, languages=["en"], domain="general", modality="text"),
        EmbeddingModelMeta(provider_name="openai", model_name="model2_openai", dim=1536, languages=["en", "es"], domain="finance", modality="text", provider_kwargs={"api_key": "fake_openai_key"}),
        EmbeddingModelMeta(provider_name="local_hf", model_name="model3_local", dim=384, languages=["en"], domain="legal", modality="text", provider_kwargs={"device":"cpu"}),
        EmbeddingModelMeta(provider_name="cohere", model_name="model4_cohere_multilingual", dim=1024, languages=["multi"], modality="text", provider_kwargs={"api_key": "fake_cohere_key"}),
        EmbeddingModelMeta(provider_name="clip", model_name="clip-ViT-B-32", dim=512, languages=["en"], modality="multimodal", provider_kwargs={}), # Added for multimodal test
    ]

@pytest.fixture
def strategy_manager(sample_model_configs):
    # Patch create_embedding_provider globally for this manager instance for most tests
    # to avoid actual provider instantiations unless specifically testing that.
    with patch('Scripts.embedding_strategy.create_embedding_provider') as mock_create_provider_global:
        # Generic mock provider that can be returned
        mock_provider = AsyncMock(spec=EmbeddingProvider)
        mock_provider.get_embedding_dim.return_value = 384 # A default, can be overridden per test
        mock_create_provider_global.return_value = mock_provider

        manager = EmbeddingStrategyManager(model_configs=sample_model_configs)
        manager.mock_create_provider_global = mock_create_provider_global # Attach for assertions if needed
        yield manager


# --- Tests for EmbeddingStrategyManager ---

def test_esm_initialization(strategy_manager, sample_model_configs):
    assert strategy_manager is not None
    assert len(strategy_manager.model_configs) == len(sample_model_configs)

def test_esm_initialization_no_configs():
    with pytest.raises(ValueError, match="At least one embedding model configuration must be provided."):
        EmbeddingStrategyManager(model_configs=[])

def test_esm_initialization_invalid_config():
    invalid_configs = [EmbeddingModelMeta(provider_name="ollama", model_name="bad", dim=0)]
    with pytest.raises(ValueError, match="Invalid model config, missing essential fields"):
         EmbeddingStrategyManager(model_configs=invalid_configs)


@patch('Scripts.embedding_strategy.create_embedding_provider')
def test_get_embedding_provider_direct_specification(mock_create_provider_local, sample_model_configs):
    # Use sample_model_configs directly to avoid fixture interaction issues with global patch
    manager = EmbeddingStrategyManager(model_configs=sample_model_configs)

    mock_ollama_instance = AsyncMock(spec=OllamaEmbeddings)
    mock_ollama_instance.get_embedding_dim.return_value = 768
    mock_create_provider_local.return_value = mock_ollama_instance

    provider = manager.get_embedding_provider({
        "provider_name": "ollama",
        "model_name": "model1_ollama"
    })

    assert provider == mock_ollama_instance
    mock_create_provider_local.assert_called_once_with(
        provider="ollama",
        model="model1_ollama"
    )
    provider_key = manager._get_provider_key(sample_model_configs[0])
    assert provider_key in manager.providers
    assert manager.providers[provider_key] == mock_ollama_instance

@patch('Scripts.embedding_strategy.create_embedding_provider')
def test_get_embedding_provider_adaptive_selection_simple(mock_create_provider_local, sample_model_configs):
    manager = EmbeddingStrategyManager(model_configs=sample_model_configs)
    mock_openai_instance = AsyncMock(spec=OpenAIEmbeddings)
    mock_openai_instance.get_embedding_dim.return_value = 1536
    mock_create_provider_local.return_value = mock_openai_instance

    provider = manager.get_embedding_provider({"language": "en", "domain": "finance", "modality": "text"})
    assert provider == mock_openai_instance
    args_list = mock_create_provider_local.call_args_list
    called_with_correct_args = False
    for call in args_list:
        _, kwargs = call
        if kwargs.get('provider') == "openai" and kwargs.get('model') == "model2_openai":
            assert kwargs.get('api_key') == "fake_openai_key"
            called_with_correct_args = True
            break
    assert called_with_correct_args

@patch('Scripts.embedding_strategy.create_embedding_provider')
def test_get_embedding_provider_fallback(mock_create_provider_local, sample_model_configs):
    manager = EmbeddingStrategyManager(model_configs=sample_model_configs)
    mock_first_provider_instance = AsyncMock(spec=OllamaEmbeddings)
    mock_first_provider_instance.get_embedding_dim.return_value = 768
    mock_create_provider_local.return_value = mock_first_provider_instance

    provider = manager.get_embedding_provider({"language": "en", "domain": "non_existent_domain", "modality": "text"})
    assert provider == mock_first_provider_instance
    args, kwargs = mock_create_provider_local.call_args
    assert kwargs['provider'] == "ollama"
    assert kwargs['model'] == "model1_ollama"

def test_get_embedding_provider_not_found(strategy_manager: EmbeddingStrategyManager):
    with pytest.raises(ValueError, match="No matching model configuration found"):
        strategy_manager.get_embedding_provider({"provider_name": "non_existent_provider", "model_name": "non_existent_model"})

@patch('Scripts.embedding_strategy.create_embedding_provider')
def test_get_fine_tuned_embedding_provider(mock_create_provider_local, sample_model_configs):
    manager = EmbeddingStrategyManager(model_configs=sample_model_configs)
    mock_local_hf_instance = AsyncMock(spec=LocalHuggingFaceEmbeddings)
    mock_local_hf_instance.get_embedding_dim.return_value = 384
    mock_create_provider_local.return_value = mock_local_hf_instance

    model_path = "path/to/my_finetuned_model"
    original_meta = next((mc for mc in manager.model_configs if mc.provider_name == "local_hf"), None)

    provider = manager.get_fine_tuned_embedding_provider(model_path, original_model_meta=original_meta)

    assert provider == mock_local_hf_instance
    mock_create_provider_local.assert_called_once_with(
        provider="local_hf",
        model_path=model_path,
        device="cpu"
    )
    provider_key = f"local_hf_finetuned_{model_path.replace('/', '_')}"
    assert provider_key in manager.providers

@patch('Scripts.embedding_strategy.create_embedding_provider')
def test_get_fine_tuned_embedding_provider_fallback(mock_create_provider_local, sample_model_configs):
    manager = EmbeddingStrategyManager(model_configs=sample_model_configs)
    original_model_config = sample_model_configs[0]

    mock_original_provider_instance = AsyncMock(spec=OllamaEmbeddings)
    mock_original_provider_instance.get_embedding_dim.return_value = original_model_config.dim

    mock_create_provider_local.side_effect = [
        ValueError("Failed to load fine-tuned model"),
        mock_original_provider_instance
    ]

    provider = manager.get_fine_tuned_embedding_provider(
        model_path_or_id="bad/path/finetuned",
        original_model_meta=original_model_config
    )

    assert provider == mock_original_provider_instance
    assert mock_create_provider_local.call_count == 2

@patch('Scripts.embedding_strategy.create_embedding_provider')
def test_get_multimodal_embedding_provider(mock_create_provider_local, sample_model_configs):
    # Use the sample_model_configs which now includes a CLIP model
    manager = EmbeddingStrategyManager(model_configs=sample_model_configs)

    mock_clip_instance = AsyncMock(spec=EmbeddingProvider)
    mock_clip_instance.get_embedding_dim.return_value = 512

    def create_provider_side_effect(provider, model, **kwargs):
        if provider == "clip" and model == "clip-ViT-B-32":
            return mock_clip_instance
        # Fallback for other calls if any, though this test is specific
        generic_mock = AsyncMock(spec=EmbeddingProvider)
        generic_mock.get_embedding_dim.return_value = kwargs.get('dim', 300)
        return generic_mock
    mock_create_provider_local.side_effect = create_provider_side_effect

    provider_multi = manager.get_multimodal_embedding_provider({"modality": "multimodal"})
    assert provider_multi == mock_clip_instance
    # Check that create_embedding_provider was called with the correct arguments for the CLIP model
    # The call would be from within get_embedding_provider, which get_multimodal_embedding_provider calls
    # We need to inspect the call that led to mock_clip_instance
    found_call = False
    for call_args_tuple in mock_create_provider_local.call_args_list:
        _, kwargs_call = call_args_tuple
        if kwargs_call.get('provider') == "clip" and kwargs_call.get('model') == "clip-ViT-B-32":
            found_call = True
            break
    assert found_call

    manager_only_text_configs = [m for m in sample_model_configs if m.modality == "text"]
    manager_only_text = EmbeddingStrategyManager(model_configs=manager_only_text_configs)
    provider_none = manager_only_text.get_multimodal_embedding_provider({"modality": "multimodal"})
    assert provider_none is None

@pytest.mark.asyncio
async def test_apply_compression_scalar_quantization_detailed(strategy_manager: EmbeddingStrategyManager):
    sample_embeddings_float = [
        [0.127, 0.508, -0.254, 1.27],
        [0.0, 0.0, 0.0, 0.0],
        [-2.54, 1.27, -0.508, 0.254]
    ]
    results_with_params = await strategy_manager._apply_compression(sample_embeddings_float, technique="scalar_quantization_int8")

    assert len(results_with_params) == len(sample_embeddings_float)

    quant_vec1, params1 = results_with_params[0]
    assert isinstance(quant_vec1[0], int)
    assert params1["scale"] == pytest.approx(1.27 / 127.0)
    assert params1["zero_point"] == 0
    expected_q1 = np.round(np.array(sample_embeddings_float[0]) / (1.27 / 127.0)).astype(np.int8).tolist()
    assert quant_vec1 == expected_q1

    quant_vec2, params2 = results_with_params[1]
    assert all(v == 0 for v in quant_vec2)
    assert params2["scale"] == 1.0
    assert params2["zero_point"] == 0

    quant_vec3, params3 = results_with_params[2]
    assert params3["scale"] == pytest.approx(2.54 / 127.0)
    expected_q3 = np.round(np.array(sample_embeddings_float[2]) / (2.54 / 127.0)).astype(np.int8).tolist()
    assert quant_vec3 == expected_q3

    for res_tuple in results_with_params:
        quant_vec = res_tuple[0]
        for val in quant_vec:
            assert -127 <= val <= 127, f"Quantized value {val} out of range for int8 symmetric."

@pytest.mark.asyncio
@patch('sklearn.decomposition.PCA')
async def test_apply_compression_pca(mock_pca_class, strategy_manager: EmbeddingStrategyManager):
    sample_embeddings_float = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    target_dimension = 2

    mock_pca_instance = MagicMock()
    mock_pca_instance.fit_transform.return_value = np.array([[0.15, 0.25], [0.55, 1.05]]) # Mocked PCA output
    mock_pca_class.return_value = mock_pca_instance

    compressed = await strategy_manager._apply_compression(sample_embeddings_float, technique="pca", target_dim=target_dimension)

    mock_pca_class.assert_called_once_with(n_components=target_dimension)
    # fit_transform is called with np.array(sample_embeddings_float)
    # Check if the argument was a numpy array
    call_arg = mock_pca_instance.fit_transform.call_args[0][0]
    assert isinstance(call_arg, np.ndarray)
    assert call_arg.shape == (2,4)

    assert len(compressed) == len(sample_embeddings_float)
    assert len(compressed[0]) == target_dimension
    assert compressed == [[0.15, 0.25], [0.55, 1.05]] # From mock return

@pytest.mark.asyncio
async def test_apply_compression_unknown_technique(strategy_manager: EmbeddingStrategyManager):
    sample_embeddings_float = [[0.1, 0.2], [0.3, 0.4]]
    compressed = await strategy_manager._apply_compression(sample_embeddings_float, technique="non_existent_tech")
    assert compressed == sample_embeddings_float

@pytest.mark.asyncio
async def test_get_compressed_embeddings(strategy_manager: EmbeddingStrategyManager):
    texts = ["hello", "world"]
    base_params = {"provider_name": "ollama", "model_name": "model1_ollama"} # From sample_model_configs

    mock_float_embeddings = [[0.1]*768, [0.2]*768]

    # Mock the base provider's generate_embeddings
    mock_base_provider = AsyncMock(spec=EmbeddingProvider)
    mock_base_provider.generate_embeddings.return_value = mock_float_embeddings

    # Ensure get_embedding_provider returns this mock when called
    strategy_manager.get_embedding_provider = MagicMock(return_value=mock_base_provider)

    # Mock _apply_compression
    mock_compressed_data = [([10, 20], {"scale": 0.01, "zero_point": 0}), ([30, 40], {"scale": 0.02, "zero_point": 0})]
    strategy_manager._apply_compression = AsyncMock(return_value=mock_compressed_data)

    result = await strategy_manager.get_compressed_embeddings(texts, base_provider_strategy_params=base_params, compression_technique="scalar_quantization_int8")

    strategy_manager.get_embedding_provider.assert_called_once_with(base_params)
    mock_base_provider.generate_embeddings.assert_called_once_with(texts)
    strategy_manager._apply_compression.assert_called_once_with(mock_float_embeddings, technique="scalar_quantization_int8")
    assert result == mock_compressed_data

def test_dequantize_vector(strategy_manager: EmbeddingStrategyManager):
    quant_vec = [10, 50, -30, 127]
    params = {"scale": 0.01, "zero_point": 0}
    expected_float_vec = [0.1, 0.5, -0.3, 1.27]

    dequantized = EmbeddingStrategyManager.dequantize_vector(quant_vec, params)

    assert len(dequantized) == len(expected_float_vec)
    for dv, ev in zip(dequantized, expected_float_vec):
        assert dv == pytest.approx(ev, abs=1e-5)

    # Test with non-zero zero_point (though current symmetric quantization uses 0)
    # Example: if quantized_value = (original_value / scale) + zero_point
    # Then original_value = (quantized_value - zero_point) * scale
    quant_vec_asym = [138, 255, 0] # Example if zero_point was 128, scale 0.01 -> original 0.1, 1.27, -1.28
    params_asym = {"scale": 0.01, "zero_point": 128}
    expected_float_asym = [0.1, 1.27, -1.28]
    dequantized_asym = EmbeddingStrategyManager.dequantize_vector(quant_vec_asym, params_asym)
    for dv, ev in zip(dequantized_asym, expected_float_asym):
        assert dv == pytest.approx(ev, abs=1e-5)

    with pytest.raises(TypeError):
        EmbeddingStrategyManager.dequantize_vector("not a list", params)
    with pytest.raises(ValueError):
        EmbeddingStrategyManager.dequantize_vector([10], {"scale_only": 0.1})

```
