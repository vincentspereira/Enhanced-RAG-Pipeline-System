from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
import yaml

@dataclass
class ModelConfig:
    embedding_model: str = "Snowflake-Labs/arctic-embed2"
    llm_model: str = "meta-llama/Llama-2-7b-hf"
    device: str = "cuda"  # or "cpu"
    batch_size: int = 32
    max_length: int = 512

@dataclass
class VectorStoreConfig:
    store_type: str = "qdrant"
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "documents"
    vector_size: int = 768

@dataclass
class ProcessingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    batch_size: int = 32
    max_workers: int = 4

@dataclass
class SystemConfig:
    model: ModelConfig
    vector_store: VectorStoreConfig
    processing: ProcessingConfig
    cache_dir: str
    enable_active_learning: bool = True
    enable_categorization: bool = True
    enable_workflows: bool = True

class ConfigManager:
    """Configuration manager for the RAG pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from file or use defaults."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = {}
        
        return SystemConfig(
            model=ModelConfig(**config_dict.get('model', {})),
            vector_store=VectorStoreConfig(**config_dict.get('vector_store', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {})),
            cache_dir=config_dict.get('cache_dir', 'cache'),
            enable_active_learning=config_dict.get('enable_active_learning', True),
            enable_categorization=config_dict.get('enable_categorization', True),
            enable_workflows=config_dict.get('enable_workflows', True)
        )
    
    def save_config(self):
        """Save current configuration to file."""
        config_dict = {
            'model': self.config.model.__dict__,
            'vector_store': self.config.vector_store.__dict__,
            'processing': self.config.processing.__dict__,
            'cache_dir': self.config.cache_dir,
            'enable_active_learning': self.config.enable_active_learning,
            'enable_categorization': self.config.enable_categorization,
            'enable_workflows': self.config.enable_workflows
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()
