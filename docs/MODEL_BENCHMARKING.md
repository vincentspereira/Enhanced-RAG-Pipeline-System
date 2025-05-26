# Model Performance Benchmarking

The RAG Pipeline includes a comprehensive model performance benchmarking system that allows you to:

- Benchmark embedding models, LLMs, and vector stores
- Track performance metrics over time
- Compare different models and configurations
- Manage model versions and track improvements

## Running Benchmarks

You can benchmark individual components or the entire system:

```bash
# Benchmark everything
python Scripts/benchmark_system.py

# Benchmark a specific embedding model
python Scripts/models/benchmark_cli.py embedding "Snowflake-Labs/arctic-embed2"

# Benchmark a specific LLM
python Scripts/models/benchmark_cli.py llm "meta-llama/Llama-2-7b-hf"

# Benchmark a specific vector store
python Scripts/models/benchmark_cli.py vector_store "qdrant"

# Compare multiple models
python Scripts/models/benchmark_cli.py compare embedding --names "Snowflake-Labs/arctic-embed2,BAAI/bge-small-en-v1.5"
```

## Model Versioning

The system includes a model versioning system to track different versions of models:

```bash
# Register a new model version
python Scripts/models/benchmark_cli.py register embedding "custom-model" --description "Fine-tuned embedding model" --set-active

# List model versions
python Scripts/models/benchmark_cli.py versions embedding "custom-model"
```

## Using in Code

You can integrate benchmarking and version tracking into your code:

```python
from Scripts.models import benchmark_model, get_registered_model

# Get a version-tracked model
model = get_registered_model('embedding', 'Snowflake-Labs/arctic-embed2')

# Benchmark a model
results = benchmark_model('embedding', 'Snowflake-Labs/arctic-embed2')
```

## Visualization

Benchmark results include visualizations to help compare model performance:

- Bar charts comparing throughput
- Line charts showing performance over time
- Detailed reports with metrics for different batch sizes and configurations

# Knowledge Graph Visualization

The RAG Pipeline includes powerful knowledge graph visualization capabilities:

## Visualization Options

- **Static Visualization**: Generate static network visualizations using matplotlib
- **Interactive HTML**: Create interactive network graphs with Plotly
- **Network Explorer**: Build explorable network visualizations with PyVis

## Usage

```python
from Scripts.knowledge_graph import init_knowledge_graph
from Scripts.knowledge_graph.visualization import KnowledgeGraphVisualizer

# Initialize knowledge graph
builder, visualizer = init_knowledge_graph(config)

# Build the knowledge graph
graph = builder.build_from_documents(documents)

# Generate visualizations
static_path = visualizer.visualize_matplotlib(graph.graph, "knowledge_graph_static")
interactive_path = visualizer.visualize_plotly(graph.graph, "knowledge_graph_interactive")
network_path = visualizer.visualize_pyvis(graph.graph, "knowledge_graph_network")

# Generate graph statistics
stats = visualizer.generate_graph_statistics(graph.graph)
```

## Features

- **Entity Highlighting**: Entities are color-coded by type
- **Relationship Visualization**: Different relationship types use distinct styles
- **Interactive Exploration**: Zoom, pan, and explore complex knowledge graphs
- **Rich Tooltips**: Hover for detailed information about entities and relationships
- **Graph Statistics**: Calculate centrality measures, density, and other metrics

The knowledge graph visualization tools make it easy to understand the complex relationships extracted from your documents and improve your RAG system's ability to utilize this structured knowledge.
