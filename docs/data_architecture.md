# Data Architecture and Flow

This document outlines the data architecture and flow for the RAG (Retrieval Augmented Generation) system, from document ingestion to query processing and feedback loops.

## 1. Overview

The system is designed to ingest various document types, process them into a queryable format, and use them to augment Large Language Model (LLM) responses to user queries. The architecture emphasizes modularity, enabling different components like data ingestion, embedding generation, search, and agent collaboration to be updated or scaled independently.

```mermaid
graph TD
    A[Data Sources] --> B(Document Ingestion & Processing);
    B --> C{Content Storage (Raw/Normalized)};
    B --> D(Metadata Storage);
    B --> E(Chunking);
    E --> F(Embedding Generation);
    F --> G{Vector Database (Embeddings + Chunk IDs)};
    D --> H(Keyword Index);
    B --> I(Knowledge Graph Construction);
    I --> J{Graph Database};

    K[User Query] --> L(Query Processing);
    L --> M{Hybrid Search};
    M --> G;
    M --> H;
    M --> J;

    M --> N(Ranking & Re-ranking);
    N --> O(Context Assembly);
    O --> P(LLM Prompt Augmentation);
    P --> Q[LLM];
    Q --> R(Generated Response);
    R --> S[User];

    S --> T(User Feedback);
    T --> U{Feedback Storage};
    U --> V(Active Learning & Model Fine-tuning);
    V --> F;
    V --> Q;
    V --> I;

    X[Agent Network] <--> L;
    X <--> P;
    X <--> V;

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
    style U fill:#9f9,stroke:#333,stroke-width:2px
```

## 2. Data Ingestion and Processing (`Scripts/document_processor.py`)

### 2.1. Data Sources
The system supports ingestion from:
- Local file systems.
- Cloud storage (future).
- Direct uploads (via API).
- Web URLs (future).
- Database dumps.
- API connectors (e.g., Google Drive, Confluence - future).

### 2.2. Supported Formats
A comprehensive list of formats including text, presentations, spreadsheets, specialized documents, archives, images (OCR), audio/video (transcription), and multi-modal PDFs. (Refer to `DocumentProcessor` for the full list).

### 2.3. Processing Pipeline
The `DocumentProcessor` handles the following stages:
1.  **Format Detection & Parsing**: Identifies file type and uses appropriate handlers to extract raw text/content. For archives, it performs recursive extraction.
2.  **Content Validation**: Basic checks for empty files, excessively large files. Placeholder for malware scanning and advanced corruption checks.
3.  **Noise Removal & Cleaning**: Normalizes Unicode, removes excessive whitespace, boilerplate (future), and optionally PII (future).
4.  **Format Normalization**: Converts extracted content into a consistent plain text representation (though structure preservation is a goal for some formats like Markdown/HTML).
5.  **Metadata Extraction**: Extracts file system metadata (name, size, dates), content-based metadata (language, hash), and potentially source-specific metadata.
6.  **Content Categorization (Future)**: Assigns categories to documents using `DocumentCategorizer`.
7.  **Chunking**: Divides the processed text into smaller, manageable chunks suitable for embedding. Aims to respect semantic boundaries (e.g., sentences, paragraphs).
    - Output: List of text chunks with associated metadata (including source document ID, chunk ID).

### 2.4. Storage for Processed Data
-   **Raw/Normalized Content Storage**: Original files and/or their normalized text versions might be stored (e.g., in a document store like MongoDB or a blob storage like S3) for archival, re-processing, or providing original context.
-   **Metadata Storage**: A relational database (e.g., PostgreSQL, MySQL) or a NoSQL document store (e.g., MongoDB) is used to store metadata for each document and chunk. This includes source information, extracted metadata, processing status, chunk IDs, etc.
-   **Chunk Storage**: While chunks are embedded, the text of the chunks themselves might be stored alongside their embeddings in the Vector DB or referenced via ID from the Metadata Store.

## 3. Embedding Generation (`Scripts/embeddings.py`, `Scripts/embedding_strategy.py`)

1.  **Input**: Text chunks from the processing pipeline.
2.  **Embedding Strategy Manager**: Selects an appropriate embedding model based on language, domain, modality (future), or other criteria (e.g., fine-tuned models).
3.  **Embedding Providers**: Supports various local, open-source, and proprietary embedding models (Ollama, OpenAI, Cohere, Voyage, Jina, SentenceTransformers).
4.  **Output**: Dense vector embeddings for each chunk.
5.  **Storage**:
    -   **Vector Database**: Embeddings are stored in a specialized vector database (e.g., Qdrant, FAISS, Weaviate, Pinecone) indexed by chunk ID. The vector DB also stores associated metadata or references (like document ID, chunk sequence) for efficient retrieval.

## 4. Indexing for Search

### 4.1. Vector Index
-   The Vector Database (as mentioned above) is the primary index for semantic search.

### 4.2. Keyword Index (`Scripts/search_system.py` using `Scripts/enhancers/hybrid_search.py`)
-   A traditional keyword index (e.g., BM25, TF-IDF) is built over the text chunks.
-   This can be implemented using libraries like `rank_bm25` for in-memory indexing or integrated with a search engine like Elasticsearch/OpenSearch.
-   The index stores document/chunk IDs to allow retrieval of full content.

### 4.3. Knowledge Graph (`Scripts/knowledge_graph/graph.py`)
-   Entities and relations are extracted from documents (or other sources).
-   This graph data is stored in a graph database (e.g., Neo4j) or an in-memory graph structure (`networkx` for smaller scale).
-   Nodes in the graph may link back to source document/chunk IDs.

## 5. Query Processing and Retrieval (`Scripts/search_system.py`)

1.  **User Query**: Input from the user.
2.  **Query Enhancement**:
    -   Spell checking.
    -   Query expansion (synonyms, related terms).
3.  **Hybrid Search Execution**:
    -   The processed query is sent to multiple search components:
        -   **Semantic Search**: Query is embedded (using an adaptive strategy via `EmbeddingStrategyManager`), and the vector is used to search the Vector Database.
        -   **Keyword Search**: Processed query terms are used to search the Keyword Index.
        -   **Graph Search (Future)**: Query (or extracted entities) is used to search the Graph Database.
4.  **Result Fusion (RRF)**: Results from different search components are combined using Reciprocal Rank Fusion (RRF) to produce an initial unified ranked list.
5.  **Contextual Re-ranking**: The top N results from fusion are re-ranked using a more powerful cross-encoder model for improved relevance.
6.  **Context Assembly**: Relevant chunks/documents are selected based on the final ranked list to form the context for the LLM.

## 6. LLM Augmentation and Response Generation (`Scripts/llm/service.py`, `Scripts/raft_system.py`)

1.  **Prompt Engineering**: The assembled context and the original user query are formatted into a prompt for the LLM.
2.  **LLM Interaction**: The prompt is sent to a selected LLM (potentially chosen by the RAFT system's intelligent routing).
3.  **Response Generation**: The LLM generates a response based on the query and the provided context.
4.  **Output**: The final response is presented to the user.

## 7. Feedback Loop and Continuous Learning (`Scripts/active_learning/learner.py`, `Scripts/raft_system.py`)

1.  **User Feedback**: Explicit (ratings, corrections) and implicit (click-throughs, engagement) feedback is collected.
2.  **Feedback Storage**: Feedback is stored, linked to queries, contexts, and responses.
3.  **Active Learning**: The `ActiveLearner` component identifies valuable data points for re-labeling or model fine-tuning.
4.  **Model Fine-tuning (RAFT)**:
    -   Collected feedback and new data are used to fine-tune embedding models and/or LLMs via the RAFT system.
    -   This includes custom dataset uploads and automated data quality assessment.
    -   Fine-tuned models are versioned and registered in the `ModelRegistry`.
5.  **Knowledge Graph Updates**: Feedback can also inform updates or corrections to the Knowledge Graph.

## 8. Agent Network Collaboration (`Scripts/agents/`)

-   The Agent Network facilitates complex workflows, task delegation, and specialized processing.
-   Agents interact using a structured A2A protocol.
-   **Data Flow**:
    -   `OrchestratorAgent` receives tasks (e.g., complex queries, content generation workflows).
    -   It breaks down tasks and delegates to `SpecialistAgent`s (e.g., `ResearchAgent` for search, `AnalysisAgent` for data processing, `CreativeAgent` for text generation).
    -   Specialist agents might query the search system (vector DB, keyword index, graph DB) or interact with LLMs.
    -   Results are returned to the Orchestrator, potentially validated by a `QualityAssuranceAgent`.
    -   `LearningAgent` collects interaction logs and outcomes for network-wide learning.
    -   `CoordinatorAgent` monitors network health and performance.

## 9. Monitoring and Analytics

-   Performance of document processing, embedding, search, LLMs, and agents is monitored.
-   Data quality metrics are tracked.
-   Drift detection mechanisms monitor for changes in data and model behavior.
-   Analytics inform system improvements, scaling decisions, and potential issues.

This architecture provides a comprehensive framework for building an intelligent RAG system with continuous improvement capabilities.
```
