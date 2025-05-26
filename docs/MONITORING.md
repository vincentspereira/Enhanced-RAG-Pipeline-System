# Monitoring Guide

This guide explains how to monitor and maintain the Enhanced RAG Pipeline system in production.

## Monitoring Components

### 1. Prometheus Metrics

Default metrics endpoint: `http://<host>:9090/metrics`

#### Key Metrics

1. **System Metrics**
```
# Document Processing
rag_documents_processed_total
rag_document_processing_duration_seconds
rag_document_processing_errors_total

# Search Performance
rag_search_requests_total
rag_search_duration_seconds
rag_search_errors_total

# Vector Store
rag_vector_count_total
rag_vector_store_operations_total
rag_vector_store_errors_total

# Resource Usage
rag_gpu_memory_usage_bytes
rag_cpu_usage_percent
rag_memory_usage_bytes
```

2. **Custom Metrics**
```
# Active Learning
rag_feedback_count_total
rag_relevance_score_changes

# Model Performance
rag_embedding_generation_duration_seconds
rag_inference_requests_total
rag_model_cache_hits_total
```

### 2. Grafana Dashboards

#### Main Dashboard Panels

1. **System Overview**
   - Document Processing Rate
   - Search Request Rate
   - Error Rate
   - System Resource Usage

2. **Performance Metrics**
   - Response Time Distribution
   - Query Latency
   - Processing Queue Length
   - Cache Hit Rate

3. **Resource Utilization**
   - GPU Usage
   - Memory Usage
   - CPU Usage
   - Network I/O

### 3. Logging

#### Log Levels
```
ERROR   - System errors requiring immediate attention
WARNING - Potential issues or degraded performance
INFO    - Normal operation events
DEBUG   - Detailed debugging information
```

#### Log Format
```
timestamp [level] component: message {
    "request_id": "uuid",
    "duration_ms": 123,
    "details": {}
}
```

## Health Checks

### 1. System Health Endpoints

```http
GET /health
Response:
{
    "status": "healthy",
    "components": {
        "api": "healthy",
        "vector_store": "healthy",
        "model_service": "healthy"
    },
    "metrics": {
        "uptime": "10d 4h 30m",
        "document_count": 10000,
        "last_processed": "2025-05-23T10:30:00Z"
    }
}
```

### 2. Component Health Checks

#### Vector Store
```bash
curl http://localhost:6333/health
```

#### Model Service
```bash
curl http://localhost:8000/models/health
```

## Performance Monitoring

### 1. Key Performance Indicators (KPIs)

- Document Processing Rate
- Search Response Time
- Relevance Score
- Error Rate
- Resource Utilization

### 2. Performance Thresholds

```yaml
thresholds:
  response_time:
    warning: 500ms
    critical: 1000ms
  error_rate:
    warning: 1%
    critical: 5%
  gpu_utilization:
    warning: 80%
    critical: 95%
  memory_usage:
    warning: 80%
    critical: 90%
```

## Alerts

### 1. Alert Rules

```yaml
alerts:
  high_error_rate:
    condition: error_rate > 5%
    duration: 5m
    severity: critical

  high_latency:
    condition: p95_response_time > 1s
    duration: 10m
    severity: warning

  resource_exhaustion:
    condition: memory_usage > 90% || gpu_usage > 95%
    duration: 5m
    severity: critical
```

### 2. Alert Channels

- Email Notifications
- Slack Integration
- PagerDuty
- SMS Alerts

## Maintenance Procedures

### 1. Backup Procedures

```bash
# Backup vector store
./scripts/backup.sh --type vector-store

# Backup configurations
./scripts/backup.sh --type config

# Backup models
./scripts/backup.sh --type models
```

### 2. Update Procedures

```bash
# Update models
./scripts/update_models.sh

# Update system components
./scripts/update_system.sh

# Apply security patches
./scripts/apply_patches.sh
```

### 3. Cleanup Procedures

```bash
# Clean old logs
./scripts/cleanup.sh --type logs --older-than 30d

# Clean cache
./scripts/cleanup.sh --type cache

# Remove old backups
./scripts/cleanup.sh --type backups --older-than 90d
```

## Troubleshooting

### 1. Common Issues

#### High Memory Usage
```bash
# Check memory usage
./scripts/monitor.py --check memory

# Clear cache if needed
./scripts/clear_cache.sh
```

#### Slow Response Times
```bash
# Check system load
./scripts/monitor.py --check load

# Analyze slow queries
./scripts/analyze_queries.sh
```

#### GPU Issues
```bash
# Check GPU status
nvidia-smi

# Reset GPU if needed
./scripts/reset_gpu.sh
```

### 2. Debug Tools

```bash
# Enable debug logging
./scripts/set_log_level.sh DEBUG

# Profile system performance
./scripts/profile.py

# Generate system report
./scripts/generate_report.sh
```

## Best Practices

1. **Regular Monitoring**
   - Check dashboards daily
   - Review alerts promptly
   - Monitor resource usage trends

2. **Maintenance Schedule**
   - Weekly system checks
   - Monthly performance reviews
   - Quarterly security audits

3. **Documentation**
   - Keep runbooks updated
   - Document incidents
   - Track system changes

4. **Capacity Planning**
   - Monitor growth trends
   - Plan resource upgrades
   - Test scaling limits
