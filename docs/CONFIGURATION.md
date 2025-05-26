# Configuration Guide

This guide details the configuration options and best practices for the Enhanced RAG Pipeline system.

## Configuration Files

### 1. Main Configuration (config.yaml)

```yaml
model:
  embedding_model: "Snowflake-Labs/arctic-embed2"
  llm_model: "meta-llama/Llama-2-7b-hf"
  device: "cuda"  # or "cpu"
  batch_size: 32
  max_length: 512

vector_store:
  store_type: "qdrant"
  host: "localhost"
  port: 6333
  collection_name: "documents"
  vector_size: 768

processing:
  chunk_size: 512
  chunk_overlap: 50
  batch_size: 32
  max_workers: 4

cache_dir: "cache"
enable_active_learning: true
enable_categorization: true
enable_workflows: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/rag_pipeline.log"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 60
  cors_origins: ["*"]
```

### 2. Environment Variables (.env)

```env
# API Keys
OPENAI_API_KEY=your_api_key
HUGGINGFACE_API_KEY=your_api_key

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9090
```

## Configuration Categories

### 1. Model Configuration

#### Embedding Models
```yaml
model:
  embedding_model: "Snowflake-Labs/arctic-embed2"  # HuggingFace model path
  device: "cuda"  # Use "cpu" for non-GPU systems
  batch_size: 32  # Adjust based on available memory
```

#### Language Models
```yaml
model:
  llm_model: "meta-llama/Llama-2-7b-hf"
  max_length: 512
  temperature: 0.7
```

### 2. Vector Store Configuration

#### Qdrant Settings
```yaml
vector_store:
  store_type: "qdrant"
  host: "localhost"
  port: 6333
  collection_name: "documents"
  vector_size: 768  # Must match embedding dimension
```

### 3. Processing Configuration

#### Chunking Settings
```yaml
processing:
  chunk_size: 512        # Characters per chunk
  chunk_overlap: 50      # Overlap between chunks
  batch_size: 32        # Processing batch size
  max_workers: 4        # Parallel processing workers
```

### 4. Feature Flags

```yaml
# Enable/disable features
enable_active_learning: true
enable_categorization: true
enable_workflows: true
enable_cache: true
```

## Environment-Specific Configurations

### 1. Development

```yaml
environment: "development"
debug: true
logging:
  level: "DEBUG"
cache:
  enable: false
```

### 2. Staging

```yaml
environment: "staging"
debug: false
logging:
  level: "INFO"
monitoring:
  enable: true
```

### 3. Production

```yaml
environment: "production"
debug: false
logging:
  level: "WARNING"
security:
  ssl_enabled: true
  jwt_auth: true
```

## Monitoring Configuration

### 1. Prometheus Settings

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
    scrape_interval: 15s
```

### 2. Logging Configuration

```yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/rag_pipeline.log"
  max_size: 100MB
  backup_count: 10
```

## Security Configuration

### 1. SSL/TLS Settings

```yaml
security:
  ssl:
    enabled: true
    cert_path: "/etc/ssl/certs/rag-pipeline.crt"
    key_path: "/etc/ssl/private/rag-pipeline.key"
```

### 2. Authentication Settings

```yaml
security:
  authentication:
    enabled: true
    jwt_secret_key: "${JWT_SECRET_KEY}"
    token_expiry: 3600
```

## Best Practices

1. **Environment Variables**
   - Store sensitive information in environment variables
   - Use .env file for development
   - Use secrets management in production

2. **Resource Management**
   - Adjust batch sizes based on available memory
   - Configure worker count based on CPU cores
   - Monitor GPU memory usage

3. **Security**
   - Enable SSL in production
   - Implement authentication
   - Restrict CORS origins
   - Use secure passwords and API keys

4. **Monitoring**
   - Enable metrics collection
   - Configure appropriate log levels
   - Set up log rotation

5. **Performance**
   - Optimize chunk sizes for your use case
   - Configure caching appropriately
   - Tune batch processing parameters
