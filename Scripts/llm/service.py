from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Assuming ModelConfig is accessible for type hinting, or use a more generic dict/config object
# from ..config.manager import ModelConfig # This creates a circular dependency if service.py is imported by config.manager indirectly.
# For now, let's assume config is passed as a dict or a compatible object.
from .openai_llm import OpenAILLM # Assuming this was the intended OpenAI client
from .ollama_llm import OllamaLLM # Import the new OllamaLLM

class LLMService(ABC):
    """Abstract base class for LLM services."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt."""
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        pass

class HuggingFaceLLM(LLMService):
    """Implementation for HuggingFace models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None, cpu_thread_count: Optional[int] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cpu" and cpu_thread_count is not None and cpu_thread_count > 0:
            torch.set_num_threads(cpu_thread_count)
            logger.info(f"HuggingFaceLLM: Using device: {self.device} with {torch.get_num_threads()} threads for PyTorch.")
        else:
            logger.info(f"HuggingFaceLLM: Using device: {self.device}. Default PyTorch threads for CPU, or GPU active.")

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval() # Set to evaluation mode

    def generate(self, prompt: str, max_length: int = 100, # Default max_length from class
                 temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def batch_generate(self, prompts: List[str], max_length: int = 100,
                      temperature: float = 0.7) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, max_length, temperature))
        return results

class LLMRegistry:
    """Registry for managing different LLM implementations."""
    
    def __init__(self):
        self._models: Dict[str, LLMService] = {}
        
    def register_model(self, name: str, model: LLMService):
        """Register a new LLM model."""
        self._models[name] = model
        
    def get_model(self, name: str) -> LLMService:
        """Get a registered model by name."""
        if name not in self._models:
            raise KeyError(f"Model {name} not found in registry")
        return self._models[name]
        
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

# Create global registry instance
llm_registry = LLMRegistry()

# Factory function to create LLM service based on configuration
def create_llm_service_from_config(config: Any) -> LLMService: # Use Any for config type to avoid circular import for now
    """
    Creates an LLM service instance based on the provided configuration.

    Args:
        config: A configuration object compatible with ModelConfig structure,
                containing llm_service name and specific settings.
                Expected attributes:
                - llm_service (str): 'openai', 'huggingface', 'ollama', etc.
                - llm_model (str): Name of the model for HuggingFace.
                - openai_api_key (str, optional): For OpenAI.
                - openai_model_name (str, optional): For OpenAI.
                - ollama_api_url (str, optional): For Ollama.
                - ollama_completion_model (str, optional): For Ollama.
                - ollama_request_timeout (int, optional): For Ollama.
                (and other specific configs for each service)

    Returns:
        LLMService: An instance of the configured LLM service.

    Raises:
        ValueError: If the configured llm_service is unknown.
    """
    service_name = getattr(config, 'llm_service', 'huggingface').lower()

    if service_name == "huggingface":
        hf_model_name = getattr(config, 'llm_model', "gpt2")
        device = getattr(config, 'device', None) # Get device from ModelConfig
        cpu_threads = getattr(config, 'cpu_thread_count', None) # Get cpu_thread_count
        return HuggingFaceLLM(
            model_name=hf_model_name,
            device=device,
            cpu_thread_count=cpu_threads
        )
    elif service_name == "openai":
        # OpenAI client doesn't use local device or cpu_thread_count config directly
        api_key = getattr(config, 'openai_api_key', os.getenv("OPENAI_API_KEY"))
        model_name = getattr(config, 'openai_model_name', "gpt-3.5-turbo")
        # Add other OpenAI specific params from config if OpenAILLM supports them
        # e.g., temperature, max_tokens from llm_config or ModelConfig itself
        return OpenAILLM(api_key=api_key, model_name=model_name)
    elif service_name == "ollama":
        api_url = getattr(config, 'ollama_api_url', "http://localhost:11434")
        completion_model = getattr(config, 'ollama_completion_model', "llama2")
        # embedding_model is not directly used by LLMService for text generation
        # embedding_model = getattr(config, 'ollama_embedding_model', None)
        timeout = getattr(config, 'ollama_request_timeout', 120)
        return OllamaLLM(
            host=api_url,
            completion_model_name=completion_model,
            # embedding_model_name=embedding_model, # OllamaLLM handles this internally for its own embedding method
            request_timeout=float(timeout)
        )
    # Add other services here as elif blocks
    # elif service_name == "anthropic":
    #     return AnthropicLLM(...)
    else:
        raise ValueError(f"Unknown LLM service configured: {service_name}")
