from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    embedding_model: str = "Snowflake-Labs/arctic-embed2"
    embedding_model_config: Dict[str, Any] = field(default_factory=dict)
    llm_service: str = "openai" # Default from Scripts/config.py
    llm_model: str = "meta-llama/Llama-2-7b-hf" # Original default
    llm_config: Dict[str, Any] = field(default_factory=dict)
    device: str = "cuda"  # or "cpu"
    batch_size: int = 32 # Original default, also in Scripts/config.py
    max_length: int = 512
    # Ollama specific configurations
    ollama_api_url: Optional[str] = "http://localhost:11434" # Base URL for Ollama API
    ollama_completion_model: Optional[str] = "llama2" # Default completion model
    ollama_embedding_model: Optional[str] = None # Specific model for embeddings via Ollama, if different
    ollama_request_timeout: int = 120 # Timeout in seconds for Ollama requests
    cpu_thread_count: Optional[int] = None # Number of threads for PyTorch on CPU, None for default

@dataclass
class VectorStoreConfig:
    store_type: str = "qdrant"
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "documents"
    vector_size: int = 768
    config: Dict[str, Any] = field(default_factory=dict) # Renamed from vector_store_config for clarity

@dataclass
class ProcessingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    batch_size: int = 32 # This seems specific to document processing batching
    max_workers: int = 4

@dataclass
class APIConfig:
    host: str = "localhost"
    port: int = 8000
    workers: int = 4
    enable_ssl: bool = False
    max_concurrent_requests: int = 10

@dataclass
class PathsConfig:
    workflow_config_path: Path = Path("workflows")
    model_cache_path: Path = Path("model_cache") # from Scripts/config.py, distinct from cache_dir
    knowledge_graph_path: Path = Path("knowledge_graph")
    cache_dir: str = "cache" # Original from manager.py for general disk caching (e.g. for QueryCache disk type)

@dataclass
class QueryCacheSettings: # Replaces old CacheSettingsConfig, maps to query_cache.CacheConfig
    cache_type: str = "memory"  # "memory", "redis", or "disk"
    redis_url: Optional[str] = "redis://localhost:6379/0" # Default Redis URL if redis type is used
    # disk_cache_dir will use paths.cache_dir by default if disk type is used
    default_ttl: int = 3600  # 1 hour
    # max_memory_size, compression, etc. can be added if needed from query_cache.CacheConfig
    # For now, keeping it to essential configurable parts. QueryCache will use its own defaults for others.

@dataclass
class FeatureFlagsConfig:
    enable_active_learning: bool = True
    enable_elasticsearch_fallback: bool = True
    enable_categorization: bool = True
    enable_workflows: bool = True
    enable_knowledge_graph: bool = True # from Scripts/config.py
    enable_custom_embeddings: bool = False # from Scripts/config.py

@dataclass
class NotificationConfig:
    notifier_type: str = "logging"  # e.g., "logging", "email", "slack"
    log_level: str = "INFO" # For LoggingNotifier
    # Add other notifier-specific configs here, e.g.:
    # email_host: Optional[str] = None
    # email_port: Optional[int] = None
    # slack_webhook_url: Optional[str] = None

@dataclass
class AuditLoggerConfig:
    log_dir: str = "logs/audit" # Default log directory for audit file logs
    use_elasticsearch: bool = False
    elasticsearch_url: Optional[str] = "http://localhost:9200" # Default ES URL for audit logs
    elasticsearch_index_prefix: str = "rag_audit"

@dataclass
class RabbitMQConfig:
    host: str = "localhost"
    port: int = 5672
    username: Optional[str] = "raguser" # Default from docker-compose, should be from secret in prod
    password: Optional[str] = "ragpassword" # Default from docker-compose, should be from secret in prod
    virtual_host: str = "/"
    default_document_queue: str = "document_processing_queue"
    # Add other params like prefetch_count, connection_attempts, retry_delay if needed

@dataclass
class SystemConfig:
    model: ModelConfig
    vector_store: VectorStoreConfig
    processing: ProcessingConfig
    api: APIConfig
    paths: PathsConfig
    query_cache: QueryCacheSettings
    feature_flags: FeatureFlagsConfig
    elasticsearch: 'ElasticsearchConfig'
    notification: NotificationConfig
    audit: AuditLoggerConfig
    rabbitmq: RabbitMQConfig # Added RabbitMQConfig


@dataclass
class ElasticsearchConfig:
    hosts: List[str] = field(default_factory=lambda: ["http://localhost:9200"])
    index_name: str = "rag_elasticsearch_index"
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    cloud_id: Optional[str] = None
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    enable_hybrid_search_in_es: bool = False # Default to false, RAGPipeline will combine
    # Add other fields from ElasticsearchFallback.ElasticsearchConfig as needed e.g.
    # index_settings: Optional[Dict[str, Any]] = None
    # index_mappings: Optional[Dict[str, Any]] = None
    batch_size: int = 1000
    timeout: int = 30
    max_retries: int = 3
    retry_interval: int = 1
    enable_fallback_for_es_itself: bool = True # Renamed from enable_fallback to avoid confusion
    fallback_threshold_es: float = 0.8 # Renamed for clarity


class ConfigManager:
    """Configuration manager for the RAG pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from file or use defaults."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {} # Ensure config_dict is a dict
        else:
            config_dict = {}
        
        return SystemConfig(
            model=ModelConfig(**config_dict.get('model', {})),
            vector_store=VectorStoreConfig(**config_dict.get('vector_store', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {})),
            api=APIConfig(**config_dict.get('api', {})),
            paths=PathsConfig(**config_dict.get('paths', {})),
            query_cache=QueryCacheSettings(**config_dict.get('query_cache', {})), # Updated
            feature_flags=FeatureFlagsConfig(**config_dict.get('feature_flags', {})),
            elasticsearch=ElasticsearchConfig(**config_dict.get('elasticsearch', {})),
            notification=NotificationConfig(**config_dict.get('notification', {})),
            audit=AuditLoggerConfig(**config_dict.get('audit', {})),
            rabbitmq=RabbitMQConfig(**config_dict.get('rabbitmq', {})) # Added rabbitmq
        )
    
    def save_config(self):
        """Save current configuration to file."""
        # Helper to convert Path objects to strings for YAML serialization
        def path_to_str_representer(dumper, data):
            return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))
        yaml.add_representer(Path, path_to_str_representer)

        config_dict = {
            'model': self.config.model.__dict__,
            'vector_store': self.config.vector_store.__dict__,
            'processing': self.config.processing.__dict__,
            'api': self.config.api.__dict__,
            'paths': {k: str(v) if isinstance(v, Path) else v for k, v in self.config.paths.__dict__.items()},
            'query_cache': self.config.query_cache.__dict__, # Updated
            'feature_flags': self.config.feature_flags.__dict__,
            'elasticsearch': self.config.elasticsearch.__dict__,
            'notification': self.config.notification.__dict__,
            'audit': self.config.audit.__dict__,
            'rabbitmq': self.config.rabbitmq.__dict__ # Added rabbitmq
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        # This needs to be more sophisticated to handle nested dataclasses
        # For now, assuming top-level key updates or direct attribute setting if needed
        for key, value in updates.items():
            if hasattr(self.config, key):
                if isinstance(getattr(self.config, key), (ModelConfig, VectorStoreConfig, ProcessingConfig, APIConfig, PathsConfig, CacheSettingsConfig, FeatureFlagsConfig)) and isinstance(value, dict):
                    # Update nested dataclass
                    nested_config = getattr(self.config, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(nested_config, sub_key):
                            setattr(nested_config, sub_key, sub_value)
                else:
                    setattr(self.config, key, value)
        self.save_config()
