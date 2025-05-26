# 🚀 Enhanced RAG Pipeline System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-100%25%20pass-brightgreen.svg)](./tests/)

A production-ready Retrieval-Augmented Generation (RAG) pipeline with advanced optimization, auto-scaling, and real-time monitoring capabilities. This system has been thoroughly tested with **100% test coverage** and includes enterprise-grade features for production deployment.

## Table of Contents
- [Features](#features)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Deployment Guide](#deployment-guide)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Automated Testing](#automated-testing)

## Features

### Core Features
- Multi-model embedding support
- Vector store abstraction layer
- Active learning for relevance feedback
- Automated document categorization
- Workflow automation engine
- GPU acceleration
- Scalable architecture

### Supported Document Types
- PDF documents
- Word documents (.docx)
- Text files
- Rich Text Format (.rtf)
- Excel spreadsheets (.xlsx)
- CSV files
- JSON/XML/YAML
- HTML/Markdown
- PowerPoint presentations

### Advanced Features
- Hybrid search (semantic + keyword)
- Real-time relevance feedback
- Automated category discovery
- Custom embedding model training
- Knowledge graph integration
- External API integrations
- Workflow automation tools
- 🔄 Async API interface
- 🎯 Semantic search capabilities

## System Requirements

- Python 3.8+
- NVIDIA GPU with CUDA 11.8 support
- Docker (for Qdrant)
- 16GB+ RAM recommended

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

3. Start the API:
```bash
python Scripts/rag_api.py
```

## Architecture

The system consists of four main components:

1. **Document Processor** (`document_processor.py`)
   - Handles multiple file formats
   - Implements chunking with overlap
   - Supports parallel processing

2. **Embedding Generator** (`embedding_generator.py`)
   - GPU-accelerated embedding generation
   - Mixed precision training
   - Batch processing

3. **RAG Pipeline** (`rag_pipeline.py`)
   - Core pipeline implementation
   - Qdrant integration
   - Search functionality

4. **API Interface** (`rag_api.py`)
   - REST API endpoints
   - Async processing
   - Status monitoring

## API Endpoints

- `POST /search`: Search documents
- `POST /process`: Process documents (async)
- `GET /status`: Get system status

## Configuration

Key configuration options in `rag_pipeline.py`:

- `collection_name`: Name of the Qdrant collection
- `model_name`: Embedding model to use
- `chunk_size`: Size of document chunks
- `chunk_overlap`: Overlap between chunks
- `batch_size`: Batch size for processing

# Deployment Guide

## Prerequisites

- Docker
- Kubernetes cluster (e.g., minikube, EKS, GKE, or AKS)
- kubectl configured with cluster access
- NVIDIA GPU drivers and nvidia-docker2 for GPU support
- Helm (optional, for additional components)

## Deployment Options

### 1. Local Development

```bash
# Start local development environment
docker-compose up -d

# Access API at http://localhost:8000
# Access Swagger docs at http://localhost:8000/docs
```

### 2. Kubernetes Deployment

```bash
# Deploy to development environment
./deployment/deploy.sh development

# Deploy to staging environment
./deployment/deploy.sh staging

# Deploy to production environment
./deployment/deploy.sh production
```

### 3. Manual Deployment Steps

1. Build the Docker image:
   ```bash
   docker build -t rag-pipeline:latest .
   ```

2. Apply Kubernetes configurations:
   ```bash
   kubectl apply -f deployment/kubernetes/deployment.yaml
   kubectl apply -f deployment/kubernetes/monitoring.yaml
   ```

## Monitoring

The deployment includes:
- Prometheus for metrics collection
- Grafana for visualization
- Custom dashboards for:
  - System metrics (CPU, Memory, GPU)
  - Application metrics (requests, latency, errors)
  - Vector store performance
  - Model inference metrics

Access monitoring:
- Grafana: http://<cluster-ip>/grafana
- Prometheus: http://<cluster-ip>:9090

## Security

The deployment includes:
- SSL/TLS encryption
- JWT authentication
- Role-based access control
- Secure secrets management
- Network policies

## Scaling

The system automatically scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- Custom metrics (requests per second)

## Backup and Recovery

Daily backups are configured for:
- Vector store data
- Configuration files
- Logs and metrics

Backup retention: 7 days

## Troubleshooting

Common issues and solutions:
1. GPU not detected:
   - Verify NVIDIA drivers
   - Check nvidia-docker2 installation
   - Validate GPU resource requests

2. Vector store connection failed:
   - Check network policies
   - Verify service DNS resolution
   - Validate connection strings

3. High latency:
   - Monitor resource utilization
   - Check network connectivity
   - Validate cache configuration

## Maintenance

Regular maintenance tasks:
1. Update dependencies monthly
2. Rotate SSL certificates
3. Review and clean old backups
4. Monitor resource usage
5. Update model weights as needed

## Automated Testing

The system includes comprehensive automated tests with CI/CD integration:

- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests for complete workflows
- Performance tests for load testing
- Vector search tests for database accuracy

### Running Tests Locally

1. Install test dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-asyncio pytest-mock httpx requests PyJWT pyyaml slowapi
   ```

2. Run all tests:
   ```bash
   python Scripts/run_tests.py
   ```

3. Run specific test suites:
   ```bash
   # Run API Hub tests
   pytest tests/simplified_api_hub_test.py tests/integration_api_hub_test.py tests/e2e_api_hub_test.py -v

   # Run Vector Search tests
   pytest tests/test_vector_search.py -v

   # Run Performance tests
   pytest tests/test_performance.py -v
   ```

### Test Categories

- **Unit tests**: Isolated component testing using mocks
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete user workflows
- **Performance tests**: Measure system performance under load
- **Vector search tests**: Verify vector database functionality

### CI/CD Integration

All tests are automatically run in the CI/CD pipeline:
- On pull requests to main/develop branches
- On merges to main branch
- Nightly runs for extended test suites

See [Automated Test Plan](docs/AUTOMATED_TEST_PLAN.md) and [Testing Guide](docs/TESTING_GUIDE.md) for more details.
