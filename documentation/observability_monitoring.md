# Observability & Monitoring

This document details the observability and monitoring setup and strategy for the system. Effective observability is crucial for maintaining system health, performance, and reliability by collecting, visualizing, and alerting on key metrics, logs, and traces.

## 1. Introduction

The goal of our observability strategy is to provide deep insights into the system's behavior, enabling rapid troubleshooting, performance optimization, and informed capacity planning. We aim to cover three main pillars of observability:
*   **Metrics**: Quantitative measurements of system health and performance over time.
*   **Logs**: Timestamped records of events occurring within the system.
*   **Traces**: Records of the path of a request as it flows through various services (to be implemented/detailed further in later phases).

## 2. Current Logging Strategy

*   **Standard Output**: All services (Internal API Gateway, RAG Query Service, Document Processing Service) are configured to log to standard output (`stdout`) and standard error (`stderr`). This is a best practice for containerized applications.
*   **Log Format**: Logs generally follow a pattern including timestamp, logger name, log level, and message (e.g., `%(asctime)s - %(name)s - %(levelname)s - %(message)s`).
*   **Log Levels**: Services use standard Python logging levels (INFO, WARNING, ERROR, DEBUG). The default level is typically INFO, configurable via environment variables (e.g., `LOGGING_LEVEL`).

### Log Aggregation (Conceptual for Kubernetes)
In a Kubernetes environment, logs written to `stdout`/`stderr` by containers are typically collected by the cluster's logging agent (e.g., Fluentd, Fluent Bit, or a custom agent) and forwarded to a centralized log aggregation backend.
*   **Recommended Tools**: ELK Stack (Elasticsearch, Logstash, Kibana) or Grafana Loki are common choices.
    *   **Elasticsearch**: Powerful for indexing and searching large volumes of logs.
    *   **Logstash/Fluentd/Fluent Bit**: Used for collecting, parsing, and shipping logs.
    *   **Kibana/Grafana**: Used for visualizing and querying logs.
*   **Benefits**: Centralized searching, analysis, and alerting based on log patterns.

## 3. Metrics Collection (Conceptual for Kubernetes using Prometheus)

Prometheus is the planned tool for metrics collection and alerting, especially within a Kubernetes environment.

### 3.1. Sources of Metrics

*   **Kubernetes API Server**: Provides metrics about the cluster state itself (nodes, pods, deployments, etc.).
*   **Kubelet / cAdvisor**: Each Kubelet includes cAdvisor, which exposes metrics about container resource usage (CPU, memory, network, disk I/O). This directly provides insights into Docker container performance for our services. The `deployment/kubernetes/monitoring.yaml` file might contain configurations related to how these base Kubernetes metrics are scraped or exposed.
*   **Service Endpoints (Application Metrics)**:
    *   FastAPI applications (our current services) can expose Prometheus-compatible metrics via an exporter library (e.g., `starlette-exporter` or `prometheus-fastapi-instrumentator`).
    *   These exporters would expose metrics via a `/metrics` endpoint on each service.
*   **Key Metrics for Current Services**:
    *   **Internal API Gateway**:
        *   HTTP request rate, error rate, latency (overall and per upstream service).
        *   Upstream service health/availability.
    *   **RAG Query Service**:
        *   `/query` endpoint: request rate, error rate, P50/P90/P99 latencies.
        *   Qdrant client: query latency, error rates.
        *   Ollama client: call latency, error rates.
        *   Embedding model load time (if applicable at startup).
        *   Resource utilization (CPU/memory, especially if GPU is used later).
    *   **Document Processing Service**:
        *   `/process_document` endpoint: request rate, error rate, processing time per document.
        *   File type processing counts (TXT, PDF, DOCX).
        *   Qdrant client: indexing latency, error rates.
        *   Embedding model load time and embedding generation time.
*   **Ollama Service**: Ollama itself may expose Prometheus metrics. This needs to be investigated for specific metrics.
*   **Qdrant Service**: Qdrant can be configured to expose Prometheus metrics (e.g., query rates, latencies, segment sizes).

### 3.2. Prometheus Setup (Conceptual)

*   **Prometheus Server**: Deployed within the Kubernetes cluster.
*   **Scraping Configuration**: Prometheus would be configured to scrape the `/metrics` endpoints of:
    *   Application services.
    *   cAdvisor (via Kubelet).
    *   Qdrant.
    *   Ollama (if available).
    *   Kubernetes API server.
*   **Service Discovery**: Kubernetes service discovery (e.g., using annotations on Service objects, or `ServiceMonitor` Custom Resources if using the Prometheus Operator) would be used to automatically find and scrape new service instances.

## 4. Visualization & Dashboards (Grafana - Conceptual)

*   **Grafana**: Planned as the primary tool for visualizing metrics collected by Prometheus and potentially logs from Loki/Elasticsearch.
*   **Dashboards**: Custom dashboards would be created to monitor:
    *   Overall system health.
    *   Performance of individual services (API Gateway, RAG Query, Doc Processing).
    *   Resource utilization of Kubernetes pods and nodes.
    *   Qdrant performance.
    *   Ollama performance.
    *   Key business metrics derived from application metrics.

## 5. Alerting (Prometheus Alertmanager - Conceptual)

*   **Alertmanager**: Would be used with Prometheus to define alerting rules based on metrics.
*   **Example Alerts**:
    *   High error rates on service endpoints.
    *   High latency for critical operations (e.g., RAG query response time).
    *   High resource utilization (CPU/memory pressure).
    *   Services being down or unresponsive.
    *   Qdrant or Ollama issues.
*   **Notification Channels**: Alerts would be routed to appropriate channels (e.g., Slack, PagerDuty, email).

## 6. Service Level Agreements (SLAs), Objectives (SLOs) & Error Budgets

These are critical for defining and measuring service reliability.

*   **Service Level Agreement (SLA)**:
    *   **Definition**: A formal commitment made to users/customers about the level of service they can expect (e.g., 99.9% uptime for the API Gateway). SLAs often have business consequences if not met.
    *   **Current Status**: Specific SLAs are not yet defined and would require business input.

*   **Service Level Objective (SLO)**:
    *   **Definition**: A target value or range of values for a specific service level indicator (SLI). SLIs are quantitative measures of service performance. SLOs are internal targets used to meet SLAs.
    *   **Examples for Current Services (Illustrative - to be refined with business requirements)**:
        *   **API Gateway (`/api/...` routes)**:
            *   Availability SLO: 99.9% of requests in a month return a non-5xx status code.
            *   Latency SLO: 99% of requests served in < 500ms.
        *   **RAG Query Service (`/query` endpoint)**:
        *   The service now exposes a `/metrics` endpoint via `prometheus-fastapi-instrumentator`.
        *   **Default Metrics Exposed**: HTTP request count, latency histogram, error counts, requests in progress.
        *   **Custom Metrics Exposed**:
            *   `rag_cache_hits_total`: Counter for cache hits (labels: `cache_type` e.g., "search_results", "llm_answer").
            *   `rag_cache_misses_total`: Counter for cache misses (labels: `cache_type`).
            *   `rag_qdrant_query_latency_seconds`: Histogram for Qdrant query latencies (labels: `query_type` e.g., "semantic", "keyword_filter").
            *   `rag_ollama_llm_latency_seconds`: Histogram for Ollama LLM call latencies (labels: `model_name`).
        *   **Example SLOs**:
            *   Availability SLO: 99.5% of `/query` requests in a month return a non-5xx status code.
            *   Latency SLO (semantic search + LLM answer): 95% of `/query` requests answered in < 5 seconds.
            *   Cache Hit Ratio (Target): > 50% for LLM answers over a day (example SLI to aim for).
        *   **Document Processing Service (`/process_document` endpoint)**:
            *   Availability SLO: 99.5% of requests return a non-5xx status code.
            *   Processing Success Rate SLO: 99% of supported documents successfully indexed.
            *   Processing Time SLO (for a standard document): 95% of documents processed in < 30 seconds.

### 4.1. Grafana Dashboards (Conceptual Examples)

Once Prometheus is collecting metrics, Grafana dashboards would be created to visualize system health and performance. Here are some conceptual examples for key dashboards and panels:

**A. Overall System Health Dashboard:**
*   **Panel: API Gateway Request Rate & Errors**: Time series graph of total requests per second to the gateway, stacked by HTTP status code (2xx, 3xx, 4xx, 5xx). (Metric: `http_requests_total` from gateway, or Prometheus FastAPI Instrumentator default metrics if gateway adopts it).
*   **Panel: Core Service Availability**: Stat panels or gauges showing current availability (based on successful health checks or uptime SLO) for API Gateway, RAG Query Service, Document Processing Service.
*   **Panel: Latency Percentiles (P95/P99) for Key Endpoints**: Line graphs showing P95/P99 latency for API Gateway main routes, RAG Query `/query`, Doc Processing `/process_document`. (Metric: `http_request_duration_seconds_bucket` from FastAPI Instrumentator).
*   **Panel: Resource Utilization Overview**: Gauges or time series for overall CPU/Memory utilization of the Kubernetes nodes or key namespaces. (Metrics from Kubelet/cAdvisor).

**B. RAG Query Service Dashboard:**
*   **Panel: `/query` Endpoint Performance**:
    *   Request Rate (per second/minute).
    *   Error Rate (percentage of 5xx responses).
    *   Latency Histogram & Percentiles (P50, P90, P95, P99). (Metrics: `fastapi_requests_total`, `fastapi_request_duration_seconds`).
*   **Panel: Qdrant Interaction**:
    *   Qdrant Query Latency Histogram (custom metric: `rag_qdrant_query_latency_seconds_bucket`).
    *   Qdrant Query Count/Rate (custom metric derived from observations or a counter).
    *   Error rate for Qdrant operations (if such custom metrics are added).
*   **Panel: Ollama LLM Interaction**:
    *   Ollama Call Latency Histogram (custom metric: `rag_ollama_llm_latency_seconds_bucket`).
    *   Ollama Call Count/Rate.
    *   Error rate for Ollama calls (if custom metrics added).
*   **Panel: Cache Performance**:
    *   Cache Hit/Miss Rates (custom metrics: `rag_cache_hits_total`, `rag_cache_misses_total` - can be shown as a ratio or stacked graph).
*   **Panel: Resource Usage**: CPU, Memory, Network I/O for RAG Query Service pods.
*   **Panel: Active Requests**: Gauge or time series for `rag_inprogress_requests`.

**C. Document Processing Service Dashboard:**
*   **Panel: `/process_document` Endpoint Performance**: Request Rate, Error Rate, Latency Histogram/Percentiles.
*   **Panel: Document Types Processed**: Count of documents processed, potentially broken down by type (TXT, PDF, DOCX) if custom metrics are added for this.
*   **Panel: Qdrant Indexing Performance**:
    *   Indexing Latency (if a custom metric is added for time taken to embed and upsert).
    *   Number of documents/chunks indexed per minute/hour.
*   **Panel: Resource Usage**: CPU, Memory for Document Processing Service pods.

**D. Kubernetes Cluster & Node Dashboard (Often provided by default K8s Prometheus/Grafana setups):**
*   Node CPU, Memory, Disk, Network utilization.
*   Pod counts, restarts, status.

**Example PromQL Queries for Grafana Panels (RAG Query Service):**

*   **`/query` Endpoint Request Rate (per second over 1m, by status code group)**:
    ```promql
    sum(rate(fastapi_requests_total{job="rag-query-service", path="/query"}[1m])) by (status_code_group)
    ```
    *Grafana Panel Type: Time series graph, stacked.*

*   **`/query` Endpoint P99 Latency (over 5m)**:
    ```promql
    histogram_quantile(0.99, sum(rate(fastapi_request_duration_seconds_bucket{job="rag-query-service", path="/query"}[5m])) by (le))
    ```
    *Grafana Panel Type: Time series graph or Stat panel.*

*   **LLM Answer Cache Hit Ratio (over 1h)**:
    ```promql
    (
      sum(rate(rag_cache_hits_total{job="rag-query-service", cache_type="llm_answer"}[1h]))
    /
      (sum(rate(rag_cache_hits_total{job="rag-query-service", cache_type="llm_answer"}[1h])) + sum(rate(rag_cache_misses_total{job="rag-query-service", cache_type="llm_answer"}[1h])))
    ) * 100
    ```
    *Grafana Panel Type: Gauge or Stat panel (percentage).*
    *Note: Add ` > 0` or `or vector(0)` to avoid "NoData" if denominators are zero.*

*   **Average Qdrant Semantic Query Latency (over 5m)**:
    ```promql
    sum(rate(rag_qdrant_query_latency_seconds_sum{job="rag-query-service", query_type="semantic"}[5m]))
    /
    sum(rate(rag_qdrant_query_latency_seconds_count{job="rag-query-service", query_type="semantic"}[5m]))
    ```
    *Grafana Panel Type: Time series graph.*

### 4.2. Alerting Rules (Concrete Examples for Alertmanager)

Alerting rules would be defined in Prometheus configuration files and managed by Alertmanager. Here are a couple of full examples:

**1. RAGQueryServiceHighP99Latency**
```yaml
groups:
- name: rag_query_service_alerts
  rules:
  - alert: RAGQueryServiceHighP99Latency
    expr: histogram_quantile(0.99, sum(rate(fastapi_request_duration_seconds_bucket{job="rag-query-service", path="/query"}[5m])) by (le, job, instance, path)) > 5
    for: 10m # Alert fires if condition is true for 10 minutes
    labels:
      severity: critical
      service: rag-query-service
    annotations:
      summary: "High P99 latency on RAG Query Service /query endpoint (Instance: {{ $labels.instance }})"
      description: "The 99th percentile latency for the /query endpoint on {{ $labels.instance }} has exceeded 5 seconds for the last 10 minutes. Current value: {{ $value | printf \"%.2f\" }}s."
      runbook_url: "https://internal.example.com/runbooks/rag-query-service-latency" # Placeholder
```

**2. RAGQueryServiceDown**
```yaml
groups:
- name: service_availability_alerts
  rules:
  - alert: RAGQueryServiceInstanceDown
    expr: up{job="rag-query-service"} == 0
    for: 3m # Alert fires if instance is down for 3 minutes
    labels:
      severity: critical
      service: rag-query-service
    annotations:
      summary: "RAG Query Service instance down (Instance: {{ $labels.instance }})"
      description: "The RAG Query Service instance {{ $labels.instance }} has been down for 3 minutes."
      runbook_url: "https://internal.example.com/runbooks/service-down" # Placeholder
```

**Other Conceptual Alert Examples (PromQL snippets):**

*   **HighCPUOrMemoryUsage (Pod)**:
    *   CPU: `sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"my-rag-instance-rag-query-service-.*", container!=""}[5m])) by (pod) / sum(kube_pod_container_resource_limits_cpu_cores{namespace="default", pod=~"my-rag-instance-rag-query-service-.*", container!=""}) by (pod) * 100 > 85`
    *   Memory: `sum(container_memory_working_set_bytes{namespace="default", pod=~"my-rag-instance-rag-query-service-.*", container!=""}) by (pod) / sum(kube_pod_container_resource_limits_memory_bytes{namespace="default", pod=~"my-rag-instance-rag-query-service-.*", container!=""}) by (pod) * 100 > 85`
*   **LowLLMAnswerCacheHitRate**:
    *   `(sum(rate(rag_cache_hits_total{job="rag-query-service",cache_type="llm_answer"}[1h])) / (sum(rate(rag_cache_hits_total{job="rag-query-service",cache_type="llm_answer"}[1h])) + sum(rate(rag_cache_misses_total{job="rag-query-service",cache_type="llm_answer"}[1h])))) * 100 < 30` (if hit rate < 30% for an hour)
*   **QdrantUnavailable (from RAG Service Health Check)**:
    *   `rag_service_component_health{job="rag-query-service", component="qdrant_accessible", status="degraded"} == 1` (assuming health check exposes metrics like this)

These examples would need to be refined with actual deployed metric names, labels, and desired thresholds.

**Example Grafana Dashboard Panel JSON (Conceptual for RAG Query Service P99 Latency):**

This is a simplified JSON structure for a single Grafana time series panel. Actual JSON can be much more complex and is usually generated via the Grafana UI.

```json
{
  "title": "RAG Query Service - P99 Latency (/query)",
  "type": "timeseries",
  "datasource": {
    "type": "prometheus",
    "uid": "your_prometheus_datasource_uid" // Replace with your Prometheus datasource UID in Grafana
  },
  "targets": [
    {
      "refId": "A",
      "expr": "histogram_quantile(0.99, sum(rate(fastapi_request_duration_seconds_bucket{job=\"rag-query-service\", path=\"/query\"}[5m])) by (le, job, instance))",
      "legendFormat": "{{instance}} P99 Latency",
      "interval": ""
    }
  ],
  "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
  "fieldConfig": {
    "defaults": {
      "color": { "mode": "palette-classic" },
      "custom": { "axisCenteredZero": false, "axisColorMode": "text", "axisLabel": "Latency (s)", "axisPlacement": "auto", "barAlignment": 0, "drawStyle": "line", "fillOpacity": 10, "gradientMode": "none", "hideFrom": { "legend": false, "tooltip": false, "viz": false }, "lineInterpolation": "linear", "lineWidth": 1, "pointSize": 5, "scaleDistribution": { "type": "linear" }, "showPoints": "auto", "spanNulls": false, "stacking": { "group": "A", "mode": "none" }, "thresholdsStyle": { "mode": "off" } },
      "mappings": [],
      "thresholds": { "mode": "absolute", "steps": [{ "color": "green", "value": null }, { "color": "red", "value": 80 }] },
      "unit": "s" // Seconds
    },
    "overrides": []
  },
  "options": { "legend": { "calcs": [], "displayMode": "list", "placement": "bottom" }, "tooltip": { "mode": "single", "sort": "none" } }
}
```

**Example Alertmanager Configuration Structure (`alertmanager.yml` snippet):**

This shows how alert rules (defined in Prometheus) are routed through Alertmanager to receivers.

```yaml
# alertmanager.yml (partial example)
route:
  group_by: ['alertname', 'service', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 1h
  receiver: 'default-receiver' # Default receiver if no specific route matches

  routes:
  - receiver: 'critical-alerts-pagerduty'
    matchers:
      - severity="critical"
    continue: false # Stop routing if this matches

  - receiver: 'warning-alerts-slack'
    matchers:
      - severity="warning"
    continue: false

receivers:
- name: 'default-receiver'
  # Placeholder, e.g., log to a file or a dead-end for unrouted alerts
  webhook_configs:
  - url: 'http://localhost:9099/dev/null' # Example no-op

- name: 'critical-alerts-pagerduty'
  pagerduty_configs:
  - service_key: "YOUR_PAGERDUTY_INTEGRATION_KEY_HERE"
    # Details like client, client_url, description can be templated from alert labels/annotations

- name: 'warning-alerts-slack'
  slack_configs:
  - api_url: "YOUR_SLACK_WEBHOOK_URL_HERE"
    channel: '#alerts-team-rag'
    send_resolved: true
    # title, text, etc. can be templated from alert labels/annotations
    # title: '[{{ .Status | toUpper }}{{ if eq .Status "firing" }}:{{ .Alerts.Firing | len }}{{ end }}] {{ .GroupLabels.alertname }} - {{ .CommonLabels.service }}'
    # text: '{{ range .Alerts }}*Summary:* {{ .Annotations.summary }}\n*Description:* {{ .Annotations.description }}\n*Runbook:* {{ .Annotations.runbook_url }}\n{{ end }}'

# Note: Prometheus itself needs to be configured with the Alertmanager's address
# and the rule files (containing the 'groups:' with alert definitions like RAGQueryServiceHighP99Latency).
# Example in prometheus.yml:
# rule_files:
#   - "/etc/prometheus/rules/*.rules.yml" # Path to your alert rule files
# alerting:
#   alertmanagers:
#   - static_configs:
#     - targets: ['alertmanager:9093'] # Address of Alertmanager
```
    *   **Measurement**: SLOs would be measured using metrics collected by Prometheus.

*   **Error Budget**:
    *   **Definition**: Derived from an SLO, the error budget is the acceptable level of unreliability. For example, an SLO of 99.9% availability means a 0.1% error budget over the compliance period.
    *   **Usage**: Error budgets provide a data-driven way to balance reliability work with feature development. If the error budget is being consumed too quickly, focus shifts to reliability. If there's ample budget, more risk can be taken with new releases.
    *   **Current Status**: Error budgets will be calculated once specific SLOs are defined and agreed upon.

## 7. Distributed Tracing (Future Phase)

*   **Concept**: Tracing requests as they flow across multiple services to understand dependencies and pinpoint bottlenecks in distributed systems.
*   **Tools**: Jaeger, Zipkin, OpenTelemetry.
*   **Status**: Not yet implemented. To be considered in later phases as the microservice architecture matures.

## 8. ML-based Predictive Monitoring (From Original Requirements - Future Phase)

*   **Concept**: Using machine learning models to analyze monitoring data (metrics, logs) to predict potential issues or detect anomalies that simple threshold-based alerting might miss.
*   **Application**: Could be applied to Docker health, resource usage, query patterns, etc.
*   **Status**: Not yet implemented. This is an advanced topic requiring a mature monitoring data pipeline and ML expertise.

*This document will be updated as the system evolves and more concrete monitoring and observability solutions are implemented.*
