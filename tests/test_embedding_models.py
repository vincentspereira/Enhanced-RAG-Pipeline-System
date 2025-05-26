import pytest
import numpy as np
from unittest.mock import Mock, patch
from Scripts.models.embeddings import (
    EmbeddingModel,
    HuggingFaceEmbedding,
    EmbeddingModelRegistry
)

@pytest.fixture
def mock_model():
    model = Mock()
    model.last_hidden_state = Mock()
    model.last_hidden_state.mean.return_value = torch.randn(1, 768)
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.randint(0, 1000, (1, 10)),
        "attention_mask": torch.ones(1, 10)
    }
    return tokenizer

class TestEmbeddingModel:
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError"""
        class ConcreteModel(EmbeddingModel):
            pass
            
        with pytest.raises(TypeError):
            ConcreteModel()

class TestHuggingFaceEmbedding:
    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_initialization(self, mock_tokenizer_init, mock_model_init):
        """Test model initialization"""
        model = HuggingFaceEmbedding("test-model")
        
        mock_model_init.assert_called_once_with("test-model")
        mock_tokenizer_init.assert_called_once_with("test-model")
    
    def test_encode_single_text(self, mock_model, mock_tokenizer):
        """Test encoding single text input"""
        with patch("transformers.AutoModel.from_pretrained") as mock_model_init:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_init:
                mock_model_init.return_value = mock_model
                mock_tokenizer_init.return_value = mock_tokenizer
                
                model = HuggingFaceEmbedding("test-model")
                result = model.encode("test text")
                
                assert isinstance(result, np.ndarray)
                assert result.shape[1] == 768
    
    def test_encode_batch_text(self, mock_model, mock_tokenizer):
        """Test encoding batch of texts"""
        with patch("transformers.AutoModel.from_pretrained") as mock_model_init:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_init:
                mock_model_init.return_value = mock_model
                mock_tokenizer_init.return_value = mock_tokenizer
                
                model = HuggingFaceEmbedding("test-model")
                texts = ["text1", "text2", "text3"]
                result = model.encode(texts)
                
                assert isinstance(result, np.ndarray)
                assert result.shape[0] == len(texts)
                assert result.shape[1] == 768

class TestEmbeddingModelRegistry:
    def test_register_and_get_model(self):
        """Test registering and retrieving models"""
        registry = EmbeddingModelRegistry()
        model = Mock(spec=EmbeddingModel)
        
        registry.register_model("test-model", model)
        retrieved_model = registry.get_model("test-model")
        
        assert retrieved_model == model
    
    def test_get_nonexistent_model(self):
        """Test getting non-existent model raises error"""
        registry = EmbeddingModelRegistry()
        
        with pytest.raises(KeyError):
            registry.get_model("nonexistent-model")
    
    def test_list_models(self):
        """Test listing registered models"""
        registry = EmbeddingModelRegistry()
        model1 = Mock(spec=EmbeddingModel)
        model2 = Mock(spec=EmbeddingModel)
        
        registry.register_model("model1", model1)
        registry.register_model("model2", model2)
        
        models = registry.list_models()
        assert set(models) == {"model1", "model2"}
