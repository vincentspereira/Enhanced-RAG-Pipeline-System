from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
    
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def generate(self, prompt: str, max_length: int = 100, 
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
