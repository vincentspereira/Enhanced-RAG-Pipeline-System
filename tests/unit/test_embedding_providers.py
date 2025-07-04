import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import numpy as np

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
    create_embedding_provider,
    EmbeddingProvider
)

# --- Fixtures ---

@pytest.fixture
def ollama_provider():
    return OllamaEmbeddings(model="test_ollama_model")

@pytest.fixture
def openai_provider():
    # Mock os.getenv for API keys if needed, or assume they are set in test env
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
    # Jina might not strictly require API key for all models
    return JinaAIEmbeddings(model="test_jina_model")

@pytest.fixture
@patch('sentence_transformers.SentenceTransformer')
def local_hf_provider(mock_sentence_transformer):
    mock_model_instance = MagicMock()
    mock_model_instance.get_sentence_embedding_dimension.return_value = 384 # Example dim
    mock_sentence_transformer.return_value = mock_model_instance
    return LocalHuggingFaceEmbeddings(model_path="mock/path/to/model")

# --- Basic Instantiation and Dimension Tests ---

def test_ollama_instantiation(ollama_provider):
    assert ollama_provider is not None
    assert ollama_provider.model == "test_ollama_model"
    # Dimension for OllamaEmbeddings is hardcoded for snowflake-arctic-embed2,
    # but for a generic test_ollama_model, it might be different.
    # Let's assume it's known or can be mocked if Ollama had a config endpoint.
    # For now, the actual get_embedding_dim is tested by its fixed return.
    assert ollama_provider.get_embedding_dim() == 1536 # Default from class

@pytest.mark.asyncio
async def test_openai_instantiation_and_dim(openai_provider):
    assert openai_provider is not None
    assert openai_provider.model == "test_openai_model"
    # Mock the async _get_embedding_dim call
    openai_provider._get_embedding_dim = AsyncMock(return_value=1536)
    assert await openai_provider._get_embedding_dim() == 1536 # Test the async version
    # To test the sync get_embedding_dim, we'd need to ensure the async part is handled
    # For unit test, directly mocking _dim_cache or the result of the async call is easier
    openai_provider._dim_cache = 1536
    assert openai_provider.get_embedding_dim() == 1536


def test_cohere_instantiation_and_dim(cohere_provider):
    assert cohere_provider is not None
    assert cohere_provider.model == "test_cohere_model"
    cohere_provider._dim_cache = 1024 # Mock expected dim
    assert cohere_provider.get_embedding_dim() == 1024

def test_voyage_instantiation_and_dim(voyage_provider):
    assert voyage_provider is not None
    assert voyage_provider.model == "test_voyage_model"
    voyage_provider._dim_cache = 1024 # Mock expected dim
    assert voyage_provider.get_embedding_dim() == 1024

def test_jina_instantiation_and_dim(jina_provider):
    assert jina_provider is not None
    assert jina_provider.model == "test_jina_model"
    jina_provider._dim_cache = 768 # Mock expected dim
    assert jina_provider.get_embedding_dim() == 768

def test_local_hf_instantiation_and_dim(local_hf_provider):
    assert local_hf_provider is not None
    assert local_hf_provider.model_path == "mock/path/to/model"
    assert local_hf_provider.get_embedding_dim() == 384

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
    mock_post.assert_called() # Check that post was called

@pytest.mark.asyncio
async def test_openai_generate_embeddings(openai_provider):
    # Mock the OpenAI client's embedding creation
    mock_embedding_data = [MagicMock(embedding=[0.1] * 1536), MagicMock(embedding=[0.2] * 1536)]
    mock_openai_response = MagicMock(data=mock_embedding_data)

    # Patch the client instance within the provider
    openai_provider.client = MagicMock()
    openai_provider.client.embeddings.create = AsyncMock(return_value=mock_openai_response)

    texts = ["hello from openai", "openai test"]
    embeddings = await openai_provider.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536
    openai_provider.client.embeddings.create.assert_called_once()

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post')
async def test_cohere_generate_embeddings(mock_post, cohere_provider):
    cohere_provider._dim_cache = 1024 # Ensure dim is set for potential error fallback
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1] * 1024, [0.2] * 1024]}
    mock_post.return_value = mock_response

    texts = ["hello from cohere", "cohere test"]
    embeddings = await cohere_provider.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024
    mock_post.assert_called_once()


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
    mock_post.assert_called_once()

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
    mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_local_hf_generate_embeddings(local_hf_provider):
    # The SentenceTransformer model is already mocked in the fixture
    mock_model_instance = local_hf_provider.model
    mock_model_instance.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])

    texts = ["hello from local hf", "local hf test"]
    embeddings = await local_hf_provider.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    mock_model_instance.encode.assert_called_once_with(texts, batch_size=local_hf_provider.batch_size, show_progress_bar=False)


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
    # Test the model_path alias
    provider2 = create_embedding_provider(provider="local_hf", model="another/mock/path", enable_cache=False)
    assert isinstance(provider2, LocalHuggingFaceEmbeddings)
    assert provider2.model_path == "another/mock/path"


def test_create_unknown_provider():
    with pytest.raises(ValueError):
        create_embedding_provider(provider="unknown_provider_type", enable_cache=False)

# TODO: Add tests for CacheEmbeddings if its logic becomes more complex
# For now, create_embedding_provider wraps it, and its basic pass-through is implicitly tested.

# TODO: Add tests for error handling in providers (e.g., API errors, model load failures)
# The current tests for generation mock successful responses.
# Need to add tests where mock_post/etc. raise exceptions or return error codes.

# Example of testing Cohere API error
@pytest.mark.asyncio
@patch('httpx.AsyncClient.post')
async def test_cohere_generate_embeddings_api_error(mock_post, cohere_provider):
    cohere_provider._dim_cache = 1024 # Ensure dim for fallback
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    # Configure the mock to raise HTTPStatusError
    mock_post.return_value = mock_response
    mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_response))


    texts = ["test text"]
    embeddings = await cohere_provider.generate_embeddings(texts)

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1024 # Should return zeros of correct dimension
    assert all(v == 0.0 for v in embeddings[0])
    mock_post.assert_called_once()
```
