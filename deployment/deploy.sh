#!/bin/bash

# Deployment script for RAG Pipeline

# Configuration
ENVIRONMENT=${1:-development}
NAMESPACE="rag-pipeline"
DEPLOYMENT_CONFIG="deployment/config.yaml"
K8S_DIR="deployment/kubernetes"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl first."
        exit 1
    }
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "docker not found. Please install Docker first."
        exit 1
    }
}

build_image() {
    log_info "Building Docker image..."
    docker build -t rag-pipeline:latest .
    
    if [ $? -ne 0 ]; then
        log_error "Failed to build Docker image"
        exit 1
    fi
}

create_namespace() {
    log_info "Creating namespace if it doesn't exist..."
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
}

apply_configs() {
    log_info "Applying Kubernetes configurations..."
    
    # Apply base configurations
    kubectl apply -f $K8S_DIR/deployment.yaml
    
    # Apply monitoring configurations
    kubectl apply -f $K8S_DIR/monitoring.yaml
    
    if [ $? -ne 0 ]; then
        log_error "Failed to apply Kubernetes configurations"
        exit 1
    }
}

wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/rag-pipeline-api -n $NAMESPACE
    
    if [ $? -ne 0 ]; then
        log_error "Deployment failed"
        exit 1
    }
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Wait for Prometheus
    kubectl rollout status deployment/prometheus -n $NAMESPACE
    
    # Wait for Grafana
    kubectl rollout status deployment/grafana -n $NAMESPACE
    
    # Get Grafana service URL
    GRAFANA_URL=$(kubectl get service grafana-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    log_info "Grafana dashboard available at: http://$GRAFANA_URL"
}

main() {
    log_info "Starting deployment for environment: $ENVIRONMENT"
    
    # Check prerequisites
    check_prerequisites
    
    # Build Docker image
    build_image
    
    # Create namespace
    create_namespace
    
    # Apply Kubernetes configurations
    apply_configs
    
    # Wait for deployment
    wait_for_deployment
    
    # Setup monitoring
    setup_monitoring
    
    log_info "Deployment completed successfully!"
    
    # Print access information
    API_URL=$(kubectl get service rag-pipeline-api -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    log_info "API available at: http://$API_URL"
    log_info "Swagger documentation at: http://$API_URL/docs"
}

# Run main function
main
