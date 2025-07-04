# Data Architecture and Flow

This document outlines the data architecture and flow for the RAG (Retrieval Augmented Generation) system, from document ingestion to query processing and feedback loops.

## 1. Overview

The system is designed to ingest various document types, process them into a queryable format, and use them to augment Large Language Model (LLM) responses to user queries. The architecture emphasizes modularity, enabling different components like data ingestion, embedding generation, search, and agent collaboration to be updated or scaled independently. Phase 2 enhances these capabilities significantly, particularly in document processing intelligence, search methodologies, and agent collaboration.

```mermaid
graph TD
    A[Data Sources] --> B(Document Ingestion & Processing);
    B --> C{Content Storage (Raw/Normalized)};
    B --> D(Metadata Storage - PostgreSQL/MongoDB);
    B --> E(Intelligent Chunking);
    E --> F(Embedding Generation w/ Strategy);
    F --> G{Vector Database (Embeddings + Chunk IDs)};
    D --> H(Keyword Index - BM25/Elasticsearch);
    B --> I(Knowledge Graph Construction);
    I --> J{Graph Database};

    K[User Query] --> L(Query Processing & Enhancement);
    L --> M{Advanced Hybrid Search};
    M -- Semantic --> G;
    M -- Keyword --> H;
    M -- Graph --> J;
    M -- Federated --> Z[External Data Sources];

    M --> N(Ranking & Re-ranking w/ Neural Models);
    N --> N_PERS(Personalized Results);
    N_PERS --> O(Context Assembly);
    O --> P(LLM Prompt Augmentation);
    P --> Q[LLM (RAFT System)];
    Q --> R(Generated Response);
    R --> S[User];

    S --> T(User Feedback);
    T --> U_PREFS{User Preferences (PostgreSQL)};
    T --> U_BEHAVIOR{User Behavior Logs (MongoDB/InfluxDB)};
    U_PREFS & U_BEHAVIOR --> AS[Analytics Service];
    AS --> N_PERS;
    AS --> V;

    V(Active Learning & RAFT Fine-tuning);
    V -- RL Enhanced --> F;
    V -- RL Enhanced --> Q;
    V --> I;

    X[Enhanced Agent Network] <--> L;
    X <--> P;
    X <--> V;
    X -.-> Z;

    Y[Monitoring & Analytics] --> B;
    Y --> M;
    Y --> Q;
    Y --> X;

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style S fill:#f9f,stroke:#333,stroke-width:2px
    style Q fill:#ccf,stroke:#333,stroke-width:2px
    style K fill:#lightgrey,stroke:#333,stroke-width:2px
    style G fill:#9f9,stroke:#333,stroke-width:2px
    style H fill:#9f9,stroke:#333,stroke-width:2px
    style J fill:#9f9,stroke:#333,stroke-width:2px
    style C fill:#9f9,stroke:#333,stroke-width:2px
    style D fill:#9f9,stroke:#333,stroke-width:2px
    style U_PREFS fill:#9f9,stroke:#333,stroke-width:2px
    style U_BEHAVIOR fill:#9f9,stroke:#333,stroke-width:2px
    style AS fill:#orange,stroke:#333,stroke-width:2px
    style Z fill:#lightblue,stroke:#333,stroke-width:2px
```

## 2. Data Ingestion and Processing (`Scripts/document_processor.py`)

### 2.1. Data Sources
The system supports ingestion from:
- Local file systems.
- Cloud storage (AWS S3, Azure Blob, GCP Storage - future, plan for connectors).
- Direct uploads (via API).
- Web URLs (future).
- Database dumps (e.g., SQL).
- API connectors for tools like Google Drive (Slides, Sheets), Confluence.
- Email files (EML, MSG).
- CAD files (DXF metadata initially).

### 2.2. Supported Formats
A comprehensive list including:
- **Text**: PDF (with advanced OCR, table/chart extraction), DOCX, RTF, TXT, MD, HTML, HTM, XML, JSON, YAML, YML, TOML, INI, LOG.
- **Presentations**: PPTX, PPT, ODP, KEY, Google Slides (via API).
- **Spreadsheets**: XLSX, XLS, CSV, TSV, ODS, NUMBERS, Google Sheets (via API).
- **Specialized**: EPUB, MOBI, LaTeX, ReStructuredText, AsciiDoc, Jupyter Notebooks, Confluence exports.
- **Archives**: ZIP, RAR, 7Z, TAR (with recursive extraction and placeholder for malware scanning).
- **Images (OCR)**: PNG, JPEG, JPG, TIFF, BMP, WEBP, SVG, with advanced layout detection capabilities (via enhanced `ImageProcessor`).
- **Audio/Video**: MP4, AVI, MOV, MP3, WAV, FLAC (transcription via Whisper, with speaker identification).
- **Multi-Modal**: PDFs with integrated OCR for text/images, table extraction, and placeholder for chart recognition.
- **New Formats**: CAD files (metadata, e.g., DXF), Email files (EML, MSG), Database exports (SQL dumps).

### 2.3. Processing Pipeline
The `DocumentProcessor` (enhanced by `ChunkOptimizer`, `MetadataExtractor`, and `ImageProcessor`) handles:
1.  **Format Detection & Parsing**: Identifies file type and uses appropriate handlers (from `format_handlers.py`) to extract raw text/content. For archives, it performs recursive extraction.
2.  **Content Validation**: Includes checks for empty/excessively large files, MIME type validation against extension. Placeholder for malware scanning of archives.
3.  **Noise Removal & Cleaning**: Normalizes Unicode, removes excessive whitespace, URLs, emails. Future: boilerplate removal, PII redaction.
4.  **Format Normalization**: Converts extracted content into a consistent plain text representation. For PDF, includes extracted table data. For audio/video, includes speaker-annotated transcripts.
5.  **Metadata Extraction**: Uses `MetadataExtractor` for rich metadata: file system stats, content-based info (language, readability, named entities via spaCy, hash), and format-specific details (e.g., PDF author, Word paragraph count).
6.  **Content Categorization (Future)**: Assigns categories to documents using `DocumentCategorizer` post-processing.
7.  **Intelligent Chunking**: Uses `ChunkOptimizer` to divide processed text into semantically coherent chunks, respecting sentence/paragraph boundaries and optimizing size.
    - Output: List of text chunks with associated metadata (source document ID, chunk ID, hash).

### 2.4. Storage for Processed Data
-   **Raw/Normalized Content Storage**: Original files and/or their normalized text versions may be stored (e.g., in MongoDB or S3) for archival, re-processing, or providing original context.
-   **Metadata Storage**: PostgreSQL is preferred for structured document/chunk metadata. MongoDB can be used for more flexible or voluminous metadata. This includes source information, extracted metadata from `MetadataExtractor`, processing status, chunk IDs, etc.
-   **Chunk Storage**: While chunks are embedded, the text of the chunks themselves is typically stored alongside their embeddings in the Vector DB or referenced via ID from the Metadata Store.

## 3. Embedding Generation (`Scripts/embeddings.py`, `Scripts/embedding_strategy.py`)

1.  **Input**: Text chunks from the processing pipeline.
2.  **Embedding Strategy Manager**: Adaptively selects an embedding model based on document type, language, domain, modality (text, image, audio - future for unified space), or specific fine-tuned model requirements. Supports:
    - Local: `snowflake-arctic-embed2` (via Ollama or `LocalHuggingFaceEmbeddings`).
    - Open-Source: `SentenceTransformers`, `E5`, `BGE`, `all-MiniLM-L6-v2`, `multilingual-e5-large` (via `LocalHuggingFaceEmbeddings`).
    - Proprietary: `OpenAI Ada-002`, `Cohere`, `Voyage AI`, `Jina AI`.
3.  **Embedding Providers**: Concrete implementations for each supported model/service.
4.  **Domain-Specific Fine-tuning**: Supports loading fine-tuned SentenceTransformer models (via `get_fine_tuned_embedding_provider`).
5.  **Embedding Compression (Future)**: Placeholder for techniques like scalar quantization or PCA to reduce storage/latency, managed by `EmbeddingStrategyManager`.
6.  **Output**: Dense vector embeddings for each chunk.
7.  **Storage**:
    -   **Vector Database**: Embeddings are stored in a specialized vector database (e.g., Qdrant, FAISS, Weaviate, Pinecone) indexed by chunk ID. The vector DB also stores associated metadata or references.

## 4. Indexing for Search (`Scripts/search_system.py`)

### 4.1. Vector Index
-   The Vector Database is the primary index for semantic search.

### 4.2. Keyword Index
-   `AdvancedSearchSystem` uses BM25 (via `ExistingHybridSearch`) for keyword indexing over text chunks.
-   The index stores document/chunk IDs.

### 4.3. Knowledge Graph (`Scripts/knowledge_graph/graph.py`)
-   Entities and relations are extracted.
-   Stored in a graph database (e.g., Neo4j) or `networkx`.
-   Nodes link back to source document/chunk IDs.

## 5. Query Processing and Retrieval (`Scripts/search_system.py`)

1.  **User Query**: Input from the user.
2.  **Query Enhancement**: Spell checking, query expansion (synonyms, related terms) via `ExistingHybridSearch` component. Future: advanced query understanding.
3.  **Hybrid Search Execution**: `AdvancedSearchSystem` orchestrates:
    -   **Semantic Search**: Query embedded adaptively (via `EmbeddingStrategyManager`), searched against Vector DB.
    -   **Keyword Search**: Processed query terms searched against BM25 Keyword Index.
    -   **Graph Search (Future)**: Query/entities searched against Graph Database.
    -   **Federated Search (New)**: Placeholder to query external data sources (databases, cloud storage, web APIs) via a plugin architecture.
4.  **Result Fusion (RRF)**: Results from semantic, keyword, graph, and federated searches are combined using Reciprocal Rank Fusion.
5.  **Contextual Re-ranking**: Top N fused results are re-ranked using a deep learning cross-encoder model (e.g., `ms-marco-MiniLM-L-6-v2`).
6.  **Personalized Search (New)**: Placeholder to adjust re-ranked scores based on user profile/behavior queried from an **Analytics Service**.
7.  **Context Assembly**: Relevant chunks/documents selected to form context for the LLM.

## 6. LLM Augmentation and Response Generation (`Scripts/llm/service.py`, `Scripts/raft_system.py`)

1.  **Prompt Engineering**: Assembled context and query formatted into a prompt.
2.  **LLM Interaction (RAFT System)**:
    -   The `RAFTSystem`'s intelligent routing selects the best LLM (pre-tuned or custom fine-tuned).
    -   The prompt is sent to the selected LLM.
3.  **Response Generation**: LLM generates a response.
4.  **Output**: Final response presented to the user.

## 7. Feedback Loop and Continuous Learning (`Scripts/active_learning/learner.py`, `Scripts/raft_system.py`)

1.  **User Feedback**: Explicit and implicit feedback collected.
2.  **Feedback Storage**:
    - User preferences explicitly stated in conversations: Stored in **PostgreSQL** (via Analytics Service).
    - Detailed behavioral data (queries, clicks, interaction patterns): Logged by **Audit Service**, streamed to **MongoDB** or **InfluxDB** (via Analytics Service).
3.  **Active Learning**: `ActiveLearner` identifies data for fine-tuning.
4.  **Model Fine-tuning (RAFT System)**:
    -   `RAFTSystem` uses feedback/new data for Retrieval-Augmented Fine-Tuning of embedding models and LLMs.
    -   Includes custom dataset uploads, automated data quality assessment.
    -   Enhanced with **Reinforcement Learning** integration for model optimization.
    -   Fine-tuned models are versioned and registered in `ModelRegistry`, with experiments tracked (e.g., via W&B).
5.  **Knowledge Graph Updates**: Feedback can inform KG updates.

## 8. Agent Network Collaboration (`Scripts/agents/`)

-   The enhanced Agent Network uses an A2A protocol for complex workflows.
-   **New Agent Types**: `ResearchAgent`, `CreativeAgent`, `AnalysisAgent`, `TranslationAgent`, `ComplianceAgent` join existing ones (`Orchestrator`, `Specialist`, `Coordinator`, `QA`, `Learning`).
-   **Enhanced Protocol Features**:
    -   **Agent Performance Monitoring**: `AgentNetwork` provides detailed stats on agent status, mailbox size, and performance metrics.
    -   **Dynamic Scaling**: `AgentNetwork` includes (simulated) capability to scale agents based on workload.
    -   **Inter-Agent Learning**: Agents share knowledge via `KNOWLEDGE_SHARE` messages. `LearningAgent` analyzes interactions and broadcasts `SHARED_INSIGHT` messages to agents like Orchestrators.
-   **Data Flow**:
    -   `OrchestratorAgent` manages workflows, delegating to appropriate new and existing specialist agents.
    -   Specialists perform tasks, interacting with search systems, LLMs, or external tools (via federated search connectors).
    -   `LearningAgent` facilitates collective learning.

## 9. Monitoring and Analytics Service

-   Comprehensive monitoring of document processing, embedding pipelines, search performance (semantic, keyword, federated), LLM responses (RAFT), and agent network activity.
-   The **Analytics Service** provides deep insights into user behavior patterns by querying PostgreSQL (preferences) and MongoDB/InfluxDB (behavior logs), feeding into personalized search.
-   Data quality metrics and model drift detection are critical components.

This architecture provides a comprehensive framework for building an intelligent RAG system with continuous improvement capabilities.
```
