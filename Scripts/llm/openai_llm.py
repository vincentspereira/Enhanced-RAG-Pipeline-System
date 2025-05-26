"""
OpenAI integration for the LLM service.
"""
from typing import List, Dict, Any, Optional
import os
import openai

from Scripts.llm.service import LLMService

class OpenAILLM(LLMService):
    """OpenAI GPT integration."""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        """Initialize OpenAI LLM.
        
        Args:
            model_name: Name of the OpenAI model (e.g., gpt-4o, gpt-3.5-turbo)
            api_key: OpenAI API key (uses environment variable if not provided)
        """
        # Set API key from param or environment
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        self.model_name = model_name
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt."""
        # Get parameters
        max_tokens = kwargs.get("max_tokens", 1000)
        temperature = kwargs.get("temperature", 0.7)
        
        # Create completion
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract text
        return response.choices[0].message.content
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results
        
class AnthropicLLM(LLMService):
    """Anthropic Claude integration."""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: Optional[str] = None):
        """Initialize Anthropic Claude LLM.
        
        Args:
            model_name: Name of the Anthropic model
            api_key: Anthropic API key (uses environment variable if not provided)
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Install it with 'pip install anthropic'")
            
        # Set API key from param or environment
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment variables")
        
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt."""
        # Get parameters
        max_tokens = kwargs.get("max_tokens", 1000)
        temperature = kwargs.get("temperature", 0.7)
        
        # Create completion
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract text
        return response.content[0].text
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results
