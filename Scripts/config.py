from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

@dataclass
class SystemConfig:
    # Model Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_model_config: Dict[str, Any] = None
    vector_store: str = "qdrant"
    vector_store_config: Dict[str, Any] = None
    llm_service: str = "openai"
    llm_config: Dict[str, Any] = None

    # Feature Flags
    enable_active_learning: bool = True
    enable_knowledge_graph: bool = True
    enable_custom_embeddings: bool = False
    enable_auto_categorization: bool = True

    # Paths
    workflow_config_path: Path = Path("workflows")
    model_cache_path: Path = Path("model_cache")
    knowledge_graph_path: Path = Path("knowledge_graph")

    # API Configuration
    api_host: str = "localhost"
    api_port: int = 8000
    api_workers: int = 4
    enable_ssl: bool = False

    # Performance Settings
    batch_size: int = 32
    max_concurrent_requests: int = 10
    cache_ttl: int = 3600
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """Create SystemConfig from dictionary."""
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in SystemConfig.__dataclass_fields__
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert SystemConfig to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
