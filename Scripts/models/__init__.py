"""
Initialization module for model components.
"""
from typing import Dict, Any, List, Optional, Union
import os

from Scripts.models.embeddings import get_embedding_model
from Scripts.models.registry import model_registry, VersionedModel
from Scripts.models.benchmarking import ModelBenchmark, ModelVersionTracker

# Global benchmark and tracker instances
benchmark = ModelBenchmark()
version_tracker = ModelVersionTracker()

def init_model_system(config: Dict[str, Any]):
    """Initialize the model system from configuration.
    
    Args:
        config: Configuration dictionary
    """
    # Register embedding model factory
    model_registry.register_factory('embedding', get_embedding_model)
    
    # Initialize default embedding model if configured
    if 'model' in config and 'embedding_model' in config['model']:
        embedding_model_name = config['model']['embedding_model']
        # Load default embedding model
        try:
            _ = model_registry.get_model('embedding', embedding_model_name)
        except Exception as e:
            print(f"Warning: Could not initialize default embedding model: {e}")

def get_registered_model(model_type: str, model_name: str, **kwargs) -> VersionedModel:
    """Get a registered model with version tracking.
    
    Args:
        model_type: Type of model (embedding, llm, etc.)
        model_name: Name of the model
        **kwargs: Additional arguments for model creation
        
    Returns:
        VersionedModel instance
    """
    return model_registry.get_model(model_type, model_name, **kwargs)

def benchmark_model(model_type: str, model_name: str, **kwargs) -> Dict[str, Any]:
    """Run benchmark for a model.
    
    Args:
        model_type: Type of model (embedding, llm, vector_store)
        model_name: Name of the model
        **kwargs: Additional arguments for benchmarking
        
    Returns:
        Benchmark results
    """
    if model_type == 'embedding':
        # Get sample texts
        sample_texts = kwargs.get('texts', ['Sample text for benchmarking'] * 100)
        batch_sizes = kwargs.get('batch_sizes', None)
        runs = kwargs.get('runs', 3)
        
        return benchmark.benchmark_embedding_model(
            model_name=model_name,
            texts=sample_texts,
            batch_sizes=batch_sizes,
            runs_per_batch=runs
        )
    elif model_type == 'llm':
        # Get sample prompts
        sample_prompts = kwargs.get('prompts', ['Generate a response to this prompt.'] * 5)
        max_tokens = kwargs.get('max_tokens', 100)
        temperature = kwargs.get('temperature', 0.7)
        
        return benchmark.benchmark_llm(
            model_name=model_name,
            prompts=sample_prompts,
            max_tokens=max_tokens,
            temperature=temperature
        )
    elif model_type == 'vector_store':
        # Get vector store params
        num_vectors = kwargs.get('num_vectors', 10000)
        vector_dim = kwargs.get('vector_dim', 768)
        batch_sizes = kwargs.get('batch_sizes', None)
        search_queries = kwargs.get('search_queries', 100)
        
        return benchmark.benchmark_vector_store(
            store_name=model_name,
            num_vectors=num_vectors,
            vector_dim=vector_dim,
            batch_sizes=batch_sizes,
            search_queries=search_queries
        )
    else:
        raise ValueError(f"Unsupported model type for benchmarking: {model_type}")

# Export key components
__all__ = [
    'model_registry',
    'get_registered_model',
    'benchmark_model',
    'benchmark',
    'version_tracker',
    'VersionedModel'
]
