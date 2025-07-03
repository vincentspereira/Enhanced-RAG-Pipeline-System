# Configuring Services and Managing Secrets

This document provides guidance on configuring various services and components within the RAG system, with a special focus on managing sensitive information like API keys, passwords, and connection strings.

## General Principles for Configuration

*   **Environment Variables:** Most service configurations can be controlled via environment variables. This is the preferred method for Kubernetes deployments, where ConfigMaps and Secrets can populate environment variables in containers.
*   **`config.yaml`:** A central `config.yaml` file provides default configurations and can also be a source for settings. Environment variables typically override values from `config.yaml`. See `config.yaml` for detailed structure.
*   **Kubernetes Secrets:** Sensitive information (API keys, passwords, tokens) **MUST** be managed using Kubernetes Secrets. These secrets can then be mounted as environment variables or files into your application pods.
*   **Helm Chart (`values.yaml`):** When deploying via the Helm chart (`charts/rag-system/`), many configurations are exposed in `values.yaml`. For secrets, `values.yaml` often provides a way to reference an *existing* Kubernetes Secret (e.g., `existingSecretName` fields) or, for development only, to create a secret from direct values (e.g., `createSecret: true` and `directValues` fields – **NOT recommended for production**).

## Service-Specific Configurations

Below are key configurations required for various services. Refer to the respective Helm chart `values.yaml` and the main `config.yaml` for more comprehensive options.

### 1. Core Application (`config.yaml` based)

*   **Embedding & LLM Models:**
    *   `MODEL_EMBEDDING_MODEL`, `MODEL_LLM_MODEL`, `MODEL_DEVICE`, `MODEL_BATCH_SIZE`
    *   See `config.yaml` section: `model`
*   **LLM Provider API Keys (OpenAI, Anthropic):**
    *   `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
    *   Set these as environment variables. In Kubernetes, store them in a Secret and mount them.
    *   See `config.yaml` section: `llm.providers`
*   **Logging:**
    *   `LOGGING_LEVEL`, `LOGGING_FORMAT`, `LOGGING_FILE`
    *   See `config.yaml` section: `logging`

### 2. Vector Store Providers

The active vector store is chosen via `VECTOR_STORE_PROVIDER` (env var) or `vector_store.provider` in `config.yaml`.

*   **Qdrant (`provider: "qdrant"`)**
    *   `QDRANT_HOST`, `QDRANT_PORT`
    *   `QDRANT_COLLECTION_NAME`
    *   `QDRANT_API_KEY` (optional, if Qdrant is secured) - **Store in K8s Secret.**
    *   See `config.yaml` section: `vector_store.qdrant`
    *   Helm: Configure via `qdrant.*` subchart values or direct connection details in service configs if using external Qdrant.

*   **Weaviate (`provider: "weaviate"`)**
    *   `WEAVIATE_URL` (e.g., "http://localhost:8080")
    *   `WEAVIATE_API_KEY` (optional, if Weaviate is secured) - **Store in K8s Secret.**
    *   `WEAVIATE_CLASS_NAME` (for the main collection/class)
    *   See `config.yaml` section: `vector_store.weaviate`

*   **Milvus (`provider: "milvus"`)**
    *   `MILVUS_HOST`, `MILVUS_PORT`
    *   `MILVUS_COLLECTION_NAME`
    *   `MILVUS_USER`, `MILVUS_PASSWORD` (optional, if Milvus auth is enabled) - **Store password in K8s Secret.**
    *   See `config.yaml` section: `vector_store.milvus`

*   **Chroma (`provider: "chroma"`)**
    *   For HTTP client: `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_USE_SSL`, `CHROMA_HEADERS` (e.g., for auth tokens - **Store sensitive headers/tokens in K8s Secret configuration**).
    *   For Persistent client: `CHROMA_PATH` (path to database directory).
    *   If none of the above, an in-memory EphemeralClient is used.
    *   `CHROMA_COLLECTION_NAME`
    *   See `config.yaml` section: `vector_store.chroma`

*   **FAISS (`provider: "faiss"`)**
    *   `FAISS_INDEX_PATH` (path to save/load the FAISS index file).
    *   `FAISS_METADATA_PATH` (path to save/load associated metadata).
    *   These paths should point to persistent storage if data needs to survive pod restarts.
    *   See `config.yaml` section: `vector_store.faiss`

### 3. Notification Service (`Scripts/services/notification_service.py`)

*   Requires SMTP server details to send emails. These **MUST** be configured securely.
*   **Environment Variables (expected by the service):**
    *   `SMTP_SERVER`
    *   `SMTP_PORT` (e.g., "587" or "465")
    *   `SMTP_USERNAME`
    *   `SMTP_PASSWORD` - **Store in K8s Secret.**
    *   `SENDER_EMAIL`
*   **Helm Configuration:**
    *   The `charts/rag-system/values.yaml` under `notificationService.secrets` allows specifying an `existingSecretName`. This is the **recommended production approach**.
    *   Create a Kubernetes secret like:
        ```yaml
        apiVersion: v1
        kind: Secret
        metadata:
          name: my-smtp-credentials
          namespace: your-namespace
        type: Opaque
        stringData: # or 'data:' for base64 encoded values
          SMTP_SERVER: "smtp.example.com"
          SMTP_PORT: "587"
          SMTP_USERNAME: "user@example.com"
          SMTP_PASSWORD: "yourSmtpPassword"
          SENDER_EMAIL: "noreply@example.com"
        ```
    *   Then set `notificationService.secrets.existingSecretName: my-smtp-credentials` in your Helm values.
    *   The `notificationService.secrets.create: true` option in `values.yaml` along with direct SMTP values is for development/testing only.

### 4. RabbitMQ (Message Queue)

Used by `DocProcessingService` (producer) and `DocumentEventConsumer` (worker).

*   **Environment Variables (for producer/consumer clients):**
    *   `RABBITMQ_HOST`
    *   `RABBITMQ_PORT` (default: 5672)
    *   `RABBITMQ_USER` (optional, e.g., "guest")
    *   `RABBITMQ_PASSWORD` (optional, e.g., "guest") - **Store in K8s Secret if not default/guest.**
    *   `RABBITMQ_VHOST` (default: "/")
*   **Helm Configuration:**
    *   If using the RabbitMQ subchart (via `rabbitmq.enabled: true` in `charts/rag-system/values.yaml`), credentials can be configured there (e.g., `rabbitmq.auth.username`, `rabbitmq.auth.password`). The Bitnami RabbitMQ chart often creates its own secret. Application services would then need to be configured to use these credentials.
    *   If connecting to an external RabbitMQ, provide connection details via environment variables (populated from K8s Secrets for passwords).
*   **Advanced Features (DLQ, Priority Queues):** These are primarily configured on the RabbitMQ server itself. Refer to `documentation/MESSAGE_QUEUEING.md`.

### 5. n8n Workflow Engine

*   **Docker Compose:** `docker-compose.yml` includes an n8n service definition.
    *   `GENERIC_TIMEZONE`: Set your local timezone.
    *   `WEBHOOK_URL`: (Optional) If n8n needs a publicly accessible base URL for its webhooks.
*   **Interaction:** Application services trigger n8n workflows via HTTP POST requests to n8n webhook URLs.
    *   These URLs are obtained from your n8n workflow editor.
    *   `Scripts/utils/n8n_client.py` provides a helper function `trigger_n8n_workflow`.
*   **Callbacks from n8n to Application:**
    *   If n8n workflows need to call back into your application's APIs (e.g., the `internal-api-gateway`), they will need an API key for your application.
    *   This API key should be stored securely in n8n's credential manager and used in n8n's "HTTP Request" node (e.g., via Header Auth, `X-API-Key`).
*   Refer to `documentation/workflow_n8n_integration.md` for more details.

### 6. Database Connectors (SQL, NoSQL - other than Vector Stores)

*   **PostgreSQL (`PostgresConnector`):**
    *   `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
    *   **Store `POSTGRES_PASSWORD` in K8s Secret.**
*   **MySQL (`MySQLConnector`):**
    *   `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`
    *   **Store `MYSQL_PASSWORD` in K8s Secret.**
*   **MongoDB (`MongoDBConnector`):**
    *   `MONGO_CONNECTION_STRING` (preferred, includes auth if needed) OR
    *   `MONGO_HOST`, `MONGO_PORT`, `MONGO_DB`, `MONGO_USER`, `MONGO_PASSWORD`
    *   **Store connection string or password in K8s Secret.**
*   **SQLite (`SQLiteConnector`):**
    *   `SQLITE_DB_PATH` (path to database file, e.g., `data/sqlite.db`). Ensure this path is on persistent storage if data needs to survive pod restarts.
*   **InfluxDB (`InfluxDBConnector`):**
    *   `INFLUXDB_URL`, `INFLUXDB_TOKEN`, `INFLUXDB_ORG`, `INFLUXDB_BUCKET`
    *   **Store `INFLUXDB_TOKEN` in K8s Secret.**
*   **Neo4j (`Neo4jConnector`):**
    *   `NEO4J_URI` (e.g., "bolt://localhost:7687" or "neo4j://localhost:7687")
    *   `NEO4J_USER`, `NEO4J_PASSWORD`
    *   **Store `NEO4J_PASSWORD` in K8s Secret.**

### 7. Cloud Storage Providers (`Scripts/storage/cloud_storage.py`)

Used by `BackupManager` or potentially other services.

*   **AWS S3 (`S3StorageProvider`):**
    *   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` (optional)
    *   `S3_BUCKET_NAME` (specific to usage, e.g., for backups)
    *   **Store AWS credentials in K8s Secrets.** For EKS, consider using IAM Roles for Service Accounts (IRSA).
*   **Azure Blob Storage (`AzureBlobStorageProvider`):**
    *   `AZURE_STORAGE_CONNECTION_STRING`
    *   `AZURE_CONTAINER_NAME`
    *   **Store connection string in K8s Secret.**
*   **Google Cloud Storage (`GCSStorageProvider`):**
    *   `GOOGLE_APPLICATION_CREDENTIALS` (path to service account JSON key file). Mount this file from a K8s Secret.
    *   `GCS_BUCKET_NAME`
    *   For GKE, Workload Identity is the recommended way to grant pods access to GCP services.

### 8. Data Warehouses

*   **Snowflake (`SnowflakeConnector`):**
    *   `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_ROLE` (optional)
    *   **Store `SNOWFLAKE_PASSWORD` in K8s Secret.**
*   **Google BigQuery (`BigQueryConnector`):**
    *   `BIGQUERY_PROJECT_ID`
    *   `GOOGLE_APPLICATION_CREDENTIALS` (path to service account JSON key file) - Mount from K8s Secret.
    *   `BIGQUERY_LOCATION` (optional, e.g., "US")
    *   For GKE, Workload Identity is recommended.

## General Security Note on Secrets in Helm

When using the Helm chart, avoid hardcoding secrets directly in `values.yaml` for production. Instead:
1.  Create Kubernetes Secrets manually or using a secrets management tool (like Vault, Sealed Secrets).
2.  Reference these existing secrets in your `values.yaml` file (e.g., using `existingSecretName` fields if the chart supports it, or by configuring services to read environment variables that are populated from these secrets).

Example for mounting K8s secret env vars in a Helm deployment template:
```yaml
# In your service's deployment template (e.g., my-service-deployment.yaml)
# ...
env:
  - name: MY_API_KEY
    valueFrom:
      secretKeyRef:
        name: my-app-secrets # Name of your K8s Secret
        key: MY_API_KEY_IN_SECRET # Key within the Secret data
  - name: SMTP_PASSWORD
    valueFrom:
      secretKeyRef:
        name: smtp-credentials
        key: smtp-password
# ...
```

This document should serve as a reference. Always consult the specific component's code, its section in `config.yaml`, and the Helm chart's `values.yaml` for the most up-to-date and detailed configuration options.
