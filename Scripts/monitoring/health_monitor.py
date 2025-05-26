import psutil
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json
from pathlib import Path
import asyncio
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import requests
from sqlalchemy import create_engine, text
import qdrant_client

class SystemMetrics:
    def __init__(self):
        # CPU Metrics
        self.cpu_usage = Gauge('rag_cpu_usage_percent', 'CPU Usage Percentage')
        self.cpu_load = Gauge('rag_cpu_load_avg', 'CPU Load Average', ['interval'])
        
        # Memory Metrics
        self.memory_usage = Gauge('rag_memory_usage_bytes', 'Memory Usage in Bytes')
        self.memory_percent = Gauge('rag_memory_usage_percent', 'Memory Usage Percentage')
        
        # Disk Metrics
        self.disk_usage = Gauge('rag_disk_usage_bytes', 'Disk Usage', ['mount_point'])
        self.disk_io = Gauge('rag_disk_io_bytes', 'Disk IO', ['operation'])
        
        # Application Metrics
        self.request_count = Counter('rag_request_total', 'Total Request Count', ['endpoint'])
        self.request_latency = Histogram('rag_request_latency_seconds', 'Request Latency', ['endpoint'])
        self.error_count = Counter('rag_errors_total', 'Total Error Count', ['type'])
        
        # Document Processing Metrics
        self.docs_processed = Counter('rag_documents_processed_total', 'Total Documents Processed')
        self.processing_time = Histogram('rag_document_processing_seconds', 'Document Processing Time')
        self.embedding_queue_size = Gauge('rag_embedding_queue_size', 'Embedding Queue Size')

class HealthCheck:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = SystemMetrics()
        self._setup_logging()
        
        # Start Prometheus metrics server
        start_http_server(config.get('metrics_port', 9090))
        
    def _setup_logging(self):
        self.logger = logging.getLogger("health_monitor")
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "health.log")
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    async def check_system_health(self) -> Dict[str, Any]:
        """Collect system health metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg()
            
            self.metrics.cpu_usage.set(cpu_percent)
            for i, load in enumerate(load_avg):
                self.metrics.cpu_load.labels(interval=f"{(i+1)*5}min").set(load)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.memory_usage.set(memory.used)
            self.metrics.memory_percent.set(memory.percent)

            # Disk metrics
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.metrics.disk_usage.labels(mount_point=partition.mountpoint).set(usage.used)
                except PermissionError:
                    continue

            disk_io = psutil.disk_io_counters()
            self.metrics.disk_io.labels(operation='read').set(disk_io.read_bytes)
            self.metrics.disk_io.labels(operation='write').set(disk_io.write_bytes)

            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "cpu": {
                        "usage_percent": cpu_percent,
                        "load_average": load_avg
                    },
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "used": memory.used,
                        "percent": memory.percent
                    },
                    "disk": {
                        "io_read": disk_io.read_bytes,
                        "io_write": disk_io.write_bytes
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
            self.metrics.error_count.labels(type='system_metrics').inc()
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and health."""
        try:
            engine = create_engine(self.config['database_url'])
            with engine.connect() as conn:
                result = conn.execute(text('SELECT 1'))
                assert result.scalar() == 1
                
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Database health check failed: {str(e)}")
            self.metrics.error_count.labels(type='database').inc()
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def check_qdrant_health(self) -> Dict[str, Any]:
        """Check Qdrant vector database health."""
        try:
            client = qdrant_client.QdrantClient(
                url=self.config['qdrant_url'],
                api_key=self.config.get('qdrant_api_key')
            )
            collections = client.get_collections()
            
            return {
                "status": "healthy",
                "collections": len(collections),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Qdrant health check failed: {str(e)}")
            self.metrics.error_count.labels(type='qdrant').inc()
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def check_api_health(self) -> Dict[str, Any]:
        """Check API endpoints health."""
        endpoints = self.config.get('api_health_endpoints', [])
        results = {}
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                response = requests.get(endpoint['url'], timeout=5)
                latency = time.time() - start_time
                
                self.metrics.request_latency.labels(endpoint=endpoint['name']).observe(latency)
                
                if response.status_code == 200:
                    self.metrics.request_count.labels(endpoint=endpoint['name']).inc()
                    results[endpoint['name']] = {
                        "status": "healthy",
                        "latency": latency
                    }
                else:
                    self.metrics.error_count.labels(type='api').inc()
                    results[endpoint['name']] = {
                        "status": "unhealthy",
                        "status_code": response.status_code
                    }
            except Exception as e:
                self.logger.error(f"API health check failed for {endpoint['name']}: {str(e)}")
                self.metrics.error_count.labels(type='api').inc()
                results[endpoint['name']] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return results

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return consolidated results."""
        results = await asyncio.gather(
            self.check_system_health(),
            self.check_database_health(),
            self.check_qdrant_health(),
            self.check_api_health()
        )
        
        health_status = {
            "system": results[0],
            "database": results[1],
            "qdrant": results[2],
            "api_endpoints": results[3],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save health status to file
        health_file = Path(self.config.get('health_file', 'health_status.json'))
        with open(health_file, 'w') as f:
            json.dump(health_status, f, indent=2)
            
        return health_status

    async def monitor_continuously(self, interval: int = 60):
        """Run health checks continuously at specified interval."""
        while True:
            try:
                await self.run_health_checks()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring: {str(e)}")
                await asyncio.sleep(interval)

class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("alert_manager")
        
    async def check_thresholds(self, metrics: Dict[str, Any]):
        """Check if any metrics exceed defined thresholds."""
        thresholds = self.config.get('thresholds', {})
        alerts = []
        
        # CPU threshold check
        if metrics['system']['metrics']['cpu']['usage_percent'] > thresholds.get('cpu_percent', 90):
            alerts.append({
                "level": "warning",
                "message": f"CPU usage is high: {metrics['system']['metrics']['cpu']['usage_percent']}%"
            })
            
        # Memory threshold check
        if metrics['system']['metrics']['memory']['percent'] > thresholds.get('memory_percent', 90):
            alerts.append({
                "level": "warning",
                "message": f"Memory usage is high: {metrics['system']['metrics']['memory']['percent']}%"
            })
            
        # Process alerts
        for alert in alerts:
            await self.send_alert(alert)
            
    async def send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels."""
        # Log alert
        self.logger.warning(f"Alert: {alert['message']}")
        
        # Implement alert sending logic (email, Slack, etc.)
        # This is a placeholder for actual implementation
        print(f"ALERT: {alert['level']} - {alert['message']}")
