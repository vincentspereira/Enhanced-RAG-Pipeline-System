# Installation Guide

This guide provides detailed instructions for installing and setting up the Enhanced RAG Pipeline system.

## Prerequisites

### Hardware Requirements
- CPU: 4+ cores recommended
- RAM: 16GB minimum, 32GB recommended
- Storage: 100GB+ SSD recommended
- GPU: NVIDIA GPU with 8GB+ VRAM (for GPU acceleration)

### Software Requirements
- Python 3.9+
- Docker 20.10+
- Kubernetes 1.24+ (for production deployment)
- NVIDIA drivers & CUDA 11.8+ (for GPU support)

## Installation Steps

### 1. Base System Setup

#### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3.11 \
    python3-pip \
    python3-venv \
    git \
    docker.io \
    nvidia-driver-535

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

#### Windows
1. Install Python 3.11 from [python.org](https://www.python.org/downloads/)
2. Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
3. Install NVIDIA drivers from [nvidia.com](https://www.nvidia.com/download/index.aspx)
4. Install CUDA Toolkit from [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads)

### 2. Project Setup

```bash
# Clone repository
git clone https://github.com/vincentspereira/rag-pipeline.git
cd rag-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file:
```env
# API Keys
OPENAI_API_KEY=your_api_key_here
HUGGINGFACE_API_KEY=your_api_key_here

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Service Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=false

# Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 4. Database Setup

```bash
# Start Qdrant
docker-compose up -d qdrant

# Initialize database
python Scripts/init_db.py
```

### 5. Model Setup

```bash
# Download default models
python Scripts/download_models.py

# Verify GPU support
python Scripts/verify_gpu.py
```

### 6. Testing Installation

```bash
# Run tests
pytest tests/

# Test GPU support
python Scripts/test_gpu.py

# Test database connection
python Scripts/test_db.py
```

## Troubleshooting

### Common Issues

#### 1. CUDA not found
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. Database Connection Issues
```bash
# Check Qdrant status
curl http://localhost:6333/health

# Check logs
docker logs qdrant
```

#### 3. Python Package Issues
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Clean install
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

## Next Steps

After successful installation:
1. Review the [Configuration Guide](CONFIGURATION.md)
2. Follow the [Usage Guide](USAGE.md)
3. Set up [Monitoring](MONITORING.md)

## Support

For additional help:
- Create an issue on GitHub
- Check the [FAQ](FAQ.md)
- Review [Troubleshooting Guide](TROUBLESHOOTING.md)
