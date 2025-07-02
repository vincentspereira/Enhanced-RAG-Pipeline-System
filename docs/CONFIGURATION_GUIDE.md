# Configuration Guide

This document outlines the configuration strategy for the Algorithmic Trading System services. The primary method of configuration is through environment variables, which is ideal for containerized deployments, especially on Kubernetes.

## Guiding Principles

1.  **Environment Variables First**: All service configurations that vary by environment (dev, staging, prod, local-k8s) or deployment instance MUST be configurable via environment variables.
2.  **Clear Naming Convention**: Environment variables should be prefixed with the system/service area they relate to, e.g., `POSTGRES_`, `MONGO_`, `QDRANT_`, `API_GATEWAY_`, `RAG_SERVICE_`.
3.  **Defaults for Local Development**: Services can have sensible defaults in their code for local, non-containerized development, but these MUST be overridable by environment variables.
4.  **Kubernetes ConfigMaps/Secrets**: In Kubernetes deployments, environment variables will be populated from ConfigMaps (for non-sensitive data) and Secrets (for sensitive data like passwords, API keys).
5.  **`config.yaml` for Base/Local Non-Docker**: The existing `config.yaml` can serve as a base template or for local development when not using Docker/Kubernetes. However, environment variables will always take precedence. Values from `config.yaml` should be loaded by applications, and then immediately checked against/overridden by any corresponding environment variables.

## Common Environment Variables

This section will list common environment variables. As services are developed or refactored, relevant variables will be added here.

### Database Configurations

*   **PostgreSQL**:
    *   `POSTGRES_HOST`: Hostname of the PostgreSQL server.
    *   `POSTGRES_PORT`: Port of the PostgreSQL server (default: `5432`).
    *   `POSTGRES_USER`: Username for PostgreSQL connection.
    *   `POSTGRES_PASSWORD`: Password for PostgreSQL connection (use Secrets for production).
    *   `POSTGRES_DB`: Database name to connect to.
    *   **Note on pgVector**: If using the pgVector extension with PostgreSQL for vector similarity search, ensure the `vector` extension is enabled in your database (`CREATE EXTENSION IF NOT EXISTS vector;`). The `PostgresConnector` can be used to store and query vector data. Refer to `Scripts/storage/postgres_connector.py` for example query patterns.
*   **MongoDB**:
    *   `MONGO_HOST`: Hostname of the MongoDB server or replica set.
    *   `MONGO_PORT`: Port of the MongoDB server (default: `27017`).
    *   `MONGO_USER`: Username for MongoDB connection (if authentication is enabled).
    *   `MONGO_PASSWORD`: Password for MongoDB connection (use Secrets for production).
    *   `MONGO_DB`: Database name to use.
    *   `MONGO_CONNECTION_STRING`: Alternative to host/port/user/pass, provides the full MongoDB connection URI. If set, it typically overrides individual parameters.
*   **MySQL**:
    *   `MYSQL_HOST`: Hostname of the MySQL server.
    *   `MYSQL_PORT`: Port of the MySQL server (default: `3306`).
    *   `MYSQL_USER`: Username for MySQL connection.
    *   `MYSQL_PASSWORD`: Password for MySQL connection (use Secrets for production).
    *   `MYSQL_DB`: Database name to connect to.
    *   `MYSQL_CHARSET`: Character set for the MySQL connection (default: `utf8mb4`).
    *   `MYSQL_MIN_CONN`: Minimum connections for the pool (default: `1`).
    *   `MYSQL_MAX_CONN`: Maximum connections for the pool (default: `5`).
*   **SQLite**:
    *   `SQLITE_DB_PATH`: Filesystem path to the SQLite database file (e.g., `data/application.db`). Default is `data/sqlite.db`.
*   **Qdrant**:
    *   `QDRANT_URL`: URL of the Qdrant instance (e.g., `http://qdrant-service:6333`).
    *   `QDRANT_HOST`: Hostname of the Qdrant server (alternative to `QDRANT_URL` if only host is needed).
    *   `QDRANT_PORT`: Port of the Qdrant server (alternative to `QDRANT_URL`, default: `6333` for HTTP, `6334` for gRPC).
    *   `QDRANT_API_KEY`: API key for Qdrant Cloud or secured instances.
    *   `QDRANT_GRPC_PORT`: gRPC port for Qdrant (if using gRPC client).
    *   `QDRANT_COLLECTION_NAME`: Default collection name for Qdrant operations.
*   **ChromaDB**:
    *   `CHROMA_HOST`: Hostname for ChromaDB HTTP client (e.g., `localhost`). If not set and `CHROMA_PATH` is not set, an in-memory client is used.
    *   `CHROMA_PORT`: Port for ChromaDB HTTP client (e.g., `8000`).
    *   `CHROMA_PATH`: Filesystem path for ChromaDB persistent client (e.g., `data/chroma_db`). If set and host/port are not, a persistent client is used.
    *   `CHROMA_COLLECTION_NAME`: Default collection name for ChromaDB operations (default: `default_collection`).
    *   `CHROMA_EMBEDDING_FUNCTION`: Name of the SentenceTransformer model Chroma should use (e.g., `all-MiniLM-L6-v2`, or `default` for Chroma's default).

### Service-Specific Configurations

*   **API Gateway**:
    *   `API_GATEWAY_PORT`: Port the API Gateway listens on (default: `8000`).
    *   `RAG_QUERY_SERVICE_URL`: Upstream URL for the RAG Query Service (e.g., `http://rag-query-service.k8s-namespace.svc.cluster.local:8001`).
    *   `DOC_PROCESSING_SERVICE_URL`: Upstream URL for the Document Processing Service.
*   **RAG Query Service**:
    *   `RAG_SERVICE_PORT`: Port the RAG Query Service listens on (default: `8001`).
    *   `RAG_SERVICE_DEFAULT_MODEL`: Default embedding model to use.
*   **Document Processing Service**:
    *   `DOC_PROCESSING_SERVICE_PORT`: Port the Document Processing Service listens on (default: `8002`).

## Loading Configuration in Python Applications

Services should implement a utility function to load configuration, prioritizing environment variables.

**Example (Conceptual)**:

```python
# In a config_loader.py utility
import os
import yaml

DEFAULT_CONFIG_PATH = "config.yaml" # Path relative to where the script is run or an absolute path

# Helper function to navigate nested dictionaries
def _get_nested_value(config_dict, keys):
    for key in keys:
        if isinstance(config_dict, dict) and key in config_dict:
            config_dict = config_dict[key]
        else:
            return None
    return config_dict

def get_config_value(env_var_name, default=None, yaml_path=None, config_file_path=DEFAULT_CONFIG_PATH):
    """
    Retrieves a configuration value with the following priority:
    1. Environment Variable (env_var_name)
    2. Value from YAML file (config_file_path) using yaml_path (e.g., "vector_store.qdrant.host")
    3. Provided default value

    Args:
        env_var_name (str): The name of the environment variable (e.g., "QDRANT_HOST").
        default: The default value to return if not found elsewhere.
        yaml_path (str, optional): The dot-separated path to the value in the YAML file
                                   (e.g., "vector_store.qdrant.host"). If None, YAML lookup is skipped for this key.
        config_file_path (str): Path to the YAML configuration file.
    Returns:
        The configuration value.
    """
    # 1. Check environment variable first
    env_value = os.getenv(env_var_name)
    if env_value is not None:
        return env_value

    # 2. Check config.yaml (or specified file) if yaml_path is provided
    if yaml_path:
        try:
            with open(config_file_path, 'r') as f:
                config_from_file = yaml.safe_load(f)

            if config_from_file:
                nested_keys = yaml_path.split('.')
                yaml_value = _get_nested_value(config_from_file, nested_keys)
                if yaml_value is not None:
                    return yaml_value
        except FileNotFoundError:
            # File not found is okay if environment variables or defaults are expected to be used.
            # Consider logging a warning if the file is mandatory but missing.
            pass
        except yaml.YAMLError:
            # Log an error if the YAML is malformed.
            # For now, pass, assuming env vars or defaults might still work.
            pass

    # 3. Return provided default
    return default

# Example usage in a service:
# from path.to.config_loader import get_config_value # Adjust import path as needed
#
# class Settings:
#     POSTGRES_HOST = get_config_value("POSTGRES_HOST", yaml_path="database.postgres.host", default="localhost")
#     POSTGRES_PORT = int(get_config_value("POSTGRES_PORT", yaml_path="database.postgres.port", default=5432))
#     # For a key that might only exist as an env var (e.g. an API key)
#     OPENAI_API_KEY = get_config_value("OPENAI_API_KEY", yaml_path="llm.providers.openai.api_key")
#     # ... other settings
#
# settings = Settings()
```

**Note on `yaml_path`**: The `yaml_path` in `get_config_value` should correspond to the actual structure in your `config.yaml`. For example, if `config.yaml` has:
```yaml
vector_store:
  qdrant:
    host: "localhost"
```
Then to get the Qdrant host, you'd use `get_config_value("QDRANT_HOST", yaml_path="vector_store.qdrant.host", default="localhost")`.

This utility would need to be refined and placed in a shared location (e.g., `Scripts/utils/config_loader.py`) if multiple services use it. For now, the principle is what's important. Each service will ensure it loads its required parameters using this hierarchy.

## Updating `config.yaml`

The root `config.yaml` should be reviewed. If it contains sensitive information, it should be removed and users instructed to set them via environment variables or a local, gitignored `.env` file (if using `python-dotenv` for local non-Docker development).

*TODO: Review `config.yaml` and add/remove/update entries to align with this guide. For now, this document establishes the strategy.*
---

*This guide will be updated as more services are added or existing ones are refactored for standardized configuration.*
