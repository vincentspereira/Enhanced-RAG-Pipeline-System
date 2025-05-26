"""
Module for initializing and registering various LLM services.
"""
from typing import Dict, Any, Optional

from Scripts.llm.service import llm_registry, HuggingFaceLLM
from Scripts.llm.openai_llm import OpenAILLM, AnthropicLLM

def register_huggingface_llm(model_name: str):
    """Register a HuggingFace LLM."""
    llm = HuggingFaceLLM(model_name=model_name)
    llm_registry.register_model(f"huggingface/{model_name}", llm)
    return llm

def register_openai_llm(model_name: str = "gpt-4o", 
                      api_key: Optional[str] = None):
    """Register an OpenAI LLM."""
    llm = OpenAILLM(model_name=model_name, api_key=api_key)
    llm_registry.register_model(f"openai/{model_name}", llm)
    return llm

def register_anthropic_llm(model_name: str = "claude-3-opus-20240229", 
                         api_key: Optional[str] = None):
    """Register an Anthropic Claude LLM."""
    llm = AnthropicLLM(model_name=model_name, api_key=api_key)
    llm_registry.register_model(f"anthropic/{model_name}", llm)
    return llm

def init_llm_services(config: Dict[str, Any]):
    """Initialize all LLM services from configuration."""
    # Get LLM configuration
    llm_config = config.get("llm", {})
    default_provider = llm_config.get("default_provider", "huggingface")
    
    # Initialize providers
    providers = llm_config.get("providers", {})
    
    # Initialize HuggingFace models
    if "huggingface" in providers:
        hf_config = providers["huggingface"]
        for model_name in hf_config.get("models", ["google/flan-t5-base"]):
            register_huggingface_llm(model_name)
    
    # Initialize OpenAI models
    if "openai" in providers:
        openai_config = providers["openai"]
        api_key = openai_config.get("api_key")
        for model_name in openai_config.get("models", ["gpt-4o"]):
            register_openai_llm(model_name, api_key)
    
    # Initialize Anthropic models
    if "anthropic" in providers:
        anthropic_config = providers["anthropic"]
        api_key = anthropic_config.get("api_key")
        for model_name in anthropic_config.get("models", ["claude-3-opus-20240229"]):
            register_anthropic_llm(model_name, api_key)
    
    return llm_registry
