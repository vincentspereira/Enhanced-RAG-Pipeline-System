import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import numpy as np
from PIL import Image # Import PIL.Image for CLIP provider tests

# Ensure Scripts directory is in path for imports if running tests from root
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Scripts')))

from embeddings import (
    OllamaEmbeddings,
    OpenAIEmbeddings,
    CohereEmbeddings,
    VoyageAIEmbeddings,
    JinaAIEmbeddings,
    LocalHuggingFaceEmbeddings,
    CLIPEmbeddingProvider, # Import new provider
    create_embedding_provider,
    EmbeddingProvider
)

# --- Fixtures ---

@pytest.fixture
def ollama_provider():
    return OllamaEmbeddings(model="test_ollama_model")

@pytest.fixture
def openai_provider():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        return OpenAIEmbeddings(model="test_openai_model")

@pytest.fixture
def cohere_provider():
    with patch.dict(os.environ, {"COHERE_API_KEY": "test_key"}):
        return CohereEmbeddings(model="test_cohere_model")

@pytest.fixture
def voyage_provider():
    with patch.dict(os.environ, {"VOYAGE_API_KEY": "test_key"}):
        return VoyageAIEmbeddings(model="test_voyage_model")

@pytest.fixture
def jina_provider():
    return JinaAIEmbeddings(model="test_jina_model")

@pytest.fixture
@patch('sentence_transformers.SentenceTransformer')
def local_hf_provider(mock_sentence_transformer):
    mock_model_instance = MagicMock()
    mock_model_instance.get_sentence_embedding_dimension.return_value = 384
    mock_sentence_transformer.return_value = mock_model_instance
    provider = LocalHuggingFaceEmbeddings(model_path="mock/path/to/model")
    provider.model = mock_model_instance # Ensure it uses the mock
    return provider

@pytest.fixture
@patch('sentence_transformers.SentenceTransformer')
def clip_provider(mock_sentence_transformer):
    mock_clip_model_instance = MagicMock()
    mock_clip_model_instance.get_sentence_embedding_dimension.return_value = 512 # Example CLIP dim
    mock_sentence_transformer.return_value = mock_clip_model_instance
    # Provider initialization will call SentenceTransformer(model_name, device=device)
    # So, the mock_sentence_transformer will be used.
    provider = CLIPEmbeddingProvider(model_name="mock_clip_model")
    provider.model = mock_clip_model_instance # Explicitly set mock model instance
    return provider


# --- Basic Instantiation and Dimension Tests ---

def test_ollama_instantiation(ollama_provider):
    assert ollama_provider is not None
    assert ollama_provider.model == "test_ollama_model"
    assert ollama_provider.get_embedding_dim() == 1536

@pytest.mark.asyncio
async def test_openai_instantiation_and_dim(openai_provider):
    assert openai_provider is not None
    assert openai_provider.model == "test_openai_model"
    openai_provider._dim_cache = 1536 # Mock expected dim
    assert openai_provider.get_embedding_dim() == 1536

def test_cohere_instantiation_and_dim(cohere_provider):
    assert cohere_provider is not None
    assert cohere_provider.model == "test_cohere_model"
    cohere_provider._dim_cache = 1024
    assert cohere_provider.get_embedding_dim() == 1024

def test_voyage_instantiation_and_dim(voyage_provider):
    assert voyage_provider is not None
    assert voyage_provider.model == "test_voyage_model"
    voyage_provider._dim_cache = 1024
    assert voyage_provider.get_embedding_dim() == 1024

def test_jina_instantiation_and_dim(jina_provider):
    assert jina_provider is not None
    assert jina_provider.model == "test_jina_model"
    jina_provider._dim_cache = 768
    assert jina_provider.get_embedding_dim() == 768

def test_local_hf_instantiation_and_dim(local_hf_provider):
    assert local_hf_provider is not None
    assert local_hf_provider.model_path == "mock/path/to/model"
    assert local_hf_provider.get_embedding_dim() == 384

def test_clip_provider_instantiation_and_dim(clip_provider):
    assert clip_provider is not None
    assert clip_provider.model_name == "mock_clip_model"
    assert clip_provider.get_embedding_dim() == 512

# --- Embedding Generation Tests (with mocking) ---

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post')
async def test_ollama_generate_embeddings(mock_post, ollama_provider):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embedding": [0.1] * 1536}
    mock_post.return_value = mock_response
    texts = ["hello world", "another test"]
    embeddings = await ollama_provider.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536

@pytest.mark.asyncio
async def test_openai_generate_embeddings(openai_provider):
    mock_embedding_data = [MagicMock(embedding=[0.1] * 1536), MagicMock(embedding=[0.2] * 1536)]
    mock_openai_response = MagicMock(data=mock_embedding_data)
    openai_provider.client = MagicMock()
    openai_provider.client.embeddings.create = AsyncMock(return_value=mock_openai_response)
    texts = ["hello from openai", "openai test"]
    embeddings = await openai_provider.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post')
async def test_cohere_generate_embeddings(mock_post, cohere_provider):
    cohere_provider._dim_cache = 1024
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1] * 1024, [0.2] * 1024]}
    mock_post.return_value = mock_response
    texts = ["hello from cohere", "cohere test"]
    embeddings = await cohere_provider.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post')
async def test_voyage_generate_embeddings(mock_post, voyage_provider):
    voyage_provider._dim_cache = 1024
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"embedding": [0.1] * 1024}, {"embedding": [0.2] * 1024}]}
    mock_post.return_value = mock_response
    texts = ["hello from voyage", "voyage test"]
    embeddings = await voyage_provider.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post')
async def test_jina_generate_embeddings(mock_post, jina_provider):
    jina_provider._dim_cache = 768
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"embedding": [0.1] * 768}, {"embedding": [0.2] * 768}]}
    mock_post.return_value = mock_response
    texts = ["hello from jina", "jina test"]
    embeddings = await jina_provider.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768

@pytest.mark.asyncio
async def test_local_hf_generate_embeddings(local_hf_provider):
    mock_model_instance = local_hf_provider.model
    mock_model_instance.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
    texts = ["hello from local hf", "local hf test"]
    embeddings = await local_hf_provider.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    mock_model_instance.encode.assert_called_once_with(texts, batch_size=local_hf_provider.batch_size, show_progress_bar=False)

@pytest.mark.asyncio
async def test_clip_provider_generate_text_embeddings(clip_provider):
    mock_model_instance = clip_provider.model
    expected_text_embedding = np.array([[0.15] * 512])
    mock_model_instance.encode.return_value = expected_text_embedding
    text_input = "a photo of a cat"
    embeddings = await clip_provider.generate_embeddings(text_input)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 512
    assert np.array_equal(embeddings[0], expected_text_embedding[0])
    mock_model_instance.encode.assert_called_once_with([text_input], batch_size=clip_provider.batch_size, show_progress_bar=False)

@pytest.mark.asyncio
async def test_clip_provider_generate_image_embeddings(clip_provider):
    mock_model_instance = clip_provider.model
    expected_image_embedding = np.array([[0.25] * 512])
    mock_model_instance.encode.return_value = expected_image_embedding
    dummy_image = Image.new('RGB', (60, 30), color = 'red')
    embeddings = await clip_provider.generate_embeddings(dummy_image)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 512
    assert np.array_equal(embeddings[0], expected_image_embedding[0])
    called_arg = mock_model_instance.encode.call_args[0][0]
    assert isinstance(called_arg[0], Image.Image)


# --- Test create_embedding_provider ---
def test_create_ollama_provider():
    provider = create_embedding_provider(provider="ollama", model="test_m", enable_cache=False)
    assert isinstance(provider, OllamaEmbeddings)

def test_create_openai_provider():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        provider = create_embedding_provider(provider="openai", model="test_m", enable_cache=False)
        assert isinstance(provider, OpenAIEmbeddings)

def test_create_cohere_provider():
    with patch.dict(os.environ, {"COHERE_API_KEY": "test_key"}):
        provider = create_embedding_provider(provider="cohere", model="test_m", enable_cache=False)
        assert isinstance(provider, CohereEmbeddings)

def test_create_voyage_provider():
    with patch.dict(os.environ, {"VOYAGE_API_KEY": "test_key"}):
        provider = create_embedding_provider(provider="voyage", model="test_m", enable_cache=False)
        assert isinstance(provider, VoyageAIEmbeddings)

def test_create_jina_provider():
    provider = create_embedding_provider(provider="jina", model="test_m", enable_cache=False)
    assert isinstance(provider, JinaAIEmbeddings)

@patch('sentence_transformers.SentenceTransformer')
def test_create_local_hf_provider(mock_st):
    mock_model_instance = MagicMock()
    mock_model_instance.get_sentence_embedding_dimension.return_value = 128
    mock_st.return_value = mock_model_instance
    provider = create_embedding_provider(provider="local_hf", model_path="mock/path", enable_cache=False)
    assert isinstance(provider, LocalHuggingFaceEmbeddings)
    provider2 = create_embedding_provider(provider="local_hf", model="another/mock/path", enable_cache=False)
    assert isinstance(provider2, LocalHuggingFaceEmbeddings)
    assert provider2.model_path == "another/mock/path"

@patch('sentence_transformers.SentenceTransformer')
def test_create_clip_provider(mock_st_general):
    mock_clip_model_instance = MagicMock()
    mock_clip_model_instance.get_sentence_embedding_dimension.return_value = 512
    def side_effect_st_init(model_name, device):
        if "clip" in model_name.lower():
            return mock_clip_model_instance
        generic_mock_st = MagicMock()
        generic_mock_st.get_sentence_embedding_dimension.return_value = 100
        return generic_mock_st
    mock_st_general.side_effect = side_effect_st_init
    provider = create_embedding_provider(provider="clip", model="clip-ViT-B-32", enable_cache=False)
    assert isinstance(provider, CLIPEmbeddingProvider)
    assert provider.get_embedding_dim() == 512

def test_create_unknown_provider():
    with pytest.raises(ValueError):
        create_embedding_provider(provider="unknown_provider_type", enable_cache=False)

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post')
async def test_cohere_generate_embeddings_api_error(mock_post, cohere_provider):
    cohere_provider._dim_cache = 1024
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response
    mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_response))
    texts = ["test text"]
    embeddings = await cohere_provider.generate_embeddings(texts)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1024
    assert all(v == 0.0 for v in embeddings[0])
    mock_post.assert_called_once()
```
