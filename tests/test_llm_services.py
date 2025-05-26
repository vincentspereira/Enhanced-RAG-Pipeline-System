"""
Tests for LLM service implementations.
"""
import pytest
from unittest.mock import patch, MagicMock, ANY
import torch
import os

from Scripts.llm.service import LLMService, HuggingFaceLLM, LLMRegistry
from Scripts.llm.openai_llm import OpenAILLM, AnthropicLLM
from Scripts.llm import init_llm_services

class TestHuggingFaceLLM:
    @pytest.fixture
    def mock_model(self):
        with patch('Scripts.llm.service.AutoModelForCausalLM') as mock_model:
            with patch('Scripts.llm.service.AutoTokenizer') as mock_tokenizer:
                with patch('Scripts.llm.service.torch') as mock_torch:
                    # Setup mocks
                    mock_model_instance = MagicMock()
                    mock_tokenizer_instance = MagicMock()
                    
                    # Configure generate method
                    mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3]])
                    
                    # Configure tokenizer method
                    mock_tokenizer_instance.return_tensors = "pt"
                    mock_tokenizer_instance.decode.return_value = "Generated text"
                    
                    # Setup returns
                    mock_model.from_pretrained.return_value = mock_model_instance
                    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
                    mock_torch.cuda.is_available.return_value = False
                    
                    yield mock_model, mock_tokenizer, mock_model_instance, mock_tokenizer_instance
    
    def test_initialization(self, mock_model):
        """Test HuggingFace LLM initialization"""
        mock_model_class, mock_tokenizer_class, _, _ = mock_model
        
        # Initialize LLM
        llm = HuggingFaceLLM("test-model")
        
        # Verify
        mock_model_class.from_pretrained.assert_called_once_with("test-model")
        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
    
    def test_generate(self, mock_model):
        """Test generating text with HuggingFace LLM"""
        _, _, mock_model_instance, mock_tokenizer_instance = mock_model
        
        # Initialize LLM
        llm = HuggingFaceLLM("test-model")
        
        # Generate text
        result = llm.generate("Test prompt")
        
        # Verify
        mock_model_instance.generate.assert_called_once()
        mock_tokenizer_instance.decode.assert_called_once()
        assert result == "Generated text"
    
    def test_batch_generate(self, mock_model):
        """Test batch generating text with HuggingFace LLM"""
        _, _, mock_model_instance, mock_tokenizer_instance = mock_model
        
        # Initialize LLM
        llm = HuggingFaceLLM("test-model")
        
        # Generate text
        results = llm.batch_generate(["Prompt 1", "Prompt 2"])
        
        # Verify
        assert mock_model_instance.generate.call_count == 2
        assert mock_tokenizer_instance.decode.call_count == 2
        assert len(results) == 2
        assert results == ["Generated text", "Generated text"]

class TestOpenAILLM:
    @pytest.fixture
    def mock_openai(self):
        with patch('Scripts.llm.openai_llm.openai') as mock_openai:
            # Setup mock response
            mock_choice = MagicMock()
            mock_choice.message.content = "OpenAI generated text"
            
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            
            mock_openai.ChatCompletion.create.return_value = mock_response
            
            # Mock env var
            with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"}):
                yield mock_openai
    
    def test_initialization(self, mock_openai):
        """Test OpenAI LLM initialization"""
        # Initialize with env var
        llm = OpenAILLM()
        
        # Verify api key set
        assert mock_openai.api_key == "fake-api-key"
        
        # Initialize with explicit api key
        llm = OpenAILLM(api_key="explicit-api-key")
        
        # Verify api key set
        assert mock_openai.api_key == "explicit-api-key"
    
    def test_generate(self, mock_openai):
        """Test generating text with OpenAI LLM"""
        # Initialize LLM
        llm = OpenAILLM()
        
        # Generate text
        result = llm.generate("Test prompt")
        
        # Verify
        mock_openai.ChatCompletion.create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test prompt"}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        assert result == "OpenAI generated text"
    
    def test_batch_generate(self, mock_openai):
        """Test batch generating text with OpenAI LLM"""
        # Initialize LLM
        llm = OpenAILLM()
        
        # Generate text
        results = llm.batch_generate(["Prompt 1", "Prompt 2"])
        
        # Verify
        assert mock_openai.ChatCompletion.create.call_count == 2
        assert len(results) == 2
        assert results == ["OpenAI generated text", "OpenAI generated text"]

class TestAnthropicLLM:
    @pytest.fixture
    def mock_anthropic(self):
        with patch('Scripts.llm.openai_llm.anthropic') as mock_module:
            # Setup mock client
            mock_client = MagicMock()
            mock_module.Anthropic.return_value = mock_client
            
            # Setup mock response
            mock_content = MagicMock()
            mock_content.text = "Anthropic generated text"
            
            mock_response = MagicMock()
            mock_response.content = [mock_content]
            
            mock_client.messages.create.return_value = mock_response
            
            # Mock env var
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-anthropic-key"}):
                yield mock_module, mock_client
    
    def test_initialization(self, mock_anthropic):
        """Test Anthropic LLM initialization"""
        mock_module, _ = mock_anthropic
        
        # Initialize with env var
        llm = AnthropicLLM()
        
        # Verify client created
        mock_module.Anthropic.assert_called_once_with(api_key="fake-anthropic-key")
        
        # Reset mock
        mock_module.reset_mock()
        
        # Initialize with explicit api key
        llm = AnthropicLLM(api_key="explicit-api-key")
        
        # Verify client created with explicit key
        mock_module.Anthropic.assert_called_once_with(api_key="explicit-api-key")
    
    def test_generate(self, mock_anthropic):
        """Test generating text with Anthropic LLM"""
        _, mock_client = mock_anthropic
        
        # Initialize LLM
        llm = AnthropicLLM()
        
        # Generate text
        result = llm.generate("Test prompt")
        
        # Verify
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.7,
            messages=[
                {"role": "user", "content": "Test prompt"}
            ]
        )
        assert result == "Anthropic generated text"
    
    def test_batch_generate(self, mock_anthropic):
        """Test batch generating text with Anthropic LLM"""
        _, mock_client = mock_anthropic
        
        # Initialize LLM
        llm = AnthropicLLM()
        
        # Generate text
        results = llm.batch_generate(["Prompt 1", "Prompt 2"])
        
        # Verify
        assert mock_client.messages.create.call_count == 2
        assert len(results) == 2
        assert results == ["Anthropic generated text", "Anthropic generated text"]

class TestLLMRegistry:
    def test_register_and_get_model(self):
        """Test registering and retrieving LLM models"""
        registry = LLMRegistry()
        llm = MagicMock(spec=LLMService)
        
        registry.register_model("test-model", llm)
        retrieved_llm = registry.get_model("test-model")
        
        assert retrieved_llm == llm
    
    def test_list_models(self):
        """Test listing registered models"""
        registry = LLMRegistry()
        llm1 = MagicMock(spec=LLMService)
        llm2 = MagicMock(spec=LLMService)
        
        registry.register_model("model1", llm1)
        registry.register_model("model2", llm2)
        
        models = registry.list_models()
        assert set(models) == {"model1", "model2"}

class TestLLMInitialization:
    @patch('Scripts.llm.register_huggingface_llm')
    @patch('Scripts.llm.register_openai_llm')
    def test_init_llm_services(self, mock_register_openai, mock_register_hf):
        """Test initializing LLM services from config"""
        # Test config
        config = {
            "llm": {
                "default_provider": "huggingface",
                "providers": {
                    "huggingface": {
                        "models": ["model1", "model2"]
                    },
                    "openai": {
                        "api_key": "test-key",
                        "models": ["gpt-4o"]
                    }
                }
            }
        }
        
        # Initialize
        registry = init_llm_services(config)
        
        # Verify
        assert mock_register_hf.call_count == 2
        mock_register_openai.assert_called_once_with("gpt-4o", "test-key")

if __name__ == '__main__':
    pytest.main()
