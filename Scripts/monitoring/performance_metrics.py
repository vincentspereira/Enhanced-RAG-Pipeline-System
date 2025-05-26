from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time
import psutil
import logging
from datetime import datetime, timedelta
import json
from collections import deque
import threading
import asyncio
from contextlib import contextmanager
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    start_http_server
)

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    history_size: int = 1000
    sampling_interval: int = 5  # seconds
    enable_detailed_memory: bool = True
    enable_gpu_metrics: bool = True
    log_to_file: bool = True
    log_file_path: Optional[str] = "metrics.log"
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_percent": 80.0,
        "memory_percent": 80.0,
        "disk_percent": 80.0,
        "latency_ms": 1000.0
    })

class PerformanceMetrics:
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.metrics_history = deque(maxlen=self.config.history_size)
        self._running = False
        self._lock = threading.Lock()
        
        # Initialize Prometheus metrics if enabled
        if self.config.enable_prometheus:
            self._init_prometheus_metrics()
            start_http_server(self.config.prometheus_port)
        
        # Initialize GPU monitoring if enabled and available
        self.gpu_available = False
        if self.config.enable_gpu_metrics:
            try:
                import torch
                self.gpu_available = torch.cuda.is_available()
            except ImportError:
                logger.warning("torch not available for GPU metrics")
        
        # Initialize metric collectors
        self.request_count = Counter(
            'rag_requests_total',
            'Total number of RAG pipeline requests'
        )
        self.request_latency = Histogram(
            'rag_request_latency_seconds',
            'Request latency in seconds',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf'))
        )
        self.document_count = Gauge(
            'rag_documents_total',
            'Total number of documents in the system'
        )
        self.query_performance = Summary(
            'rag_query_performance_seconds',
            'Query performance statistics'
        )

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics collectors"""
        self.cpu_usage = Gauge('system_cpu_usage', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage', 'Memory usage percentage')
        self.disk_usage = Gauge('system_disk_usage', 'Disk usage percentage')
        
        if self.gpu_available:
            self.gpu_usage = Gauge('system_gpu_usage', 'GPU usage percentage')
            self.gpu_memory = Gauge('system_gpu_memory', 'GPU memory usage percentage')

    async def start_monitoring(self):
        """Start the monitoring loop"""
        self._running = True
        while self._running:
            try:
                metrics = self._collect_metrics()
                self._store_metrics(metrics)
                self._update_prometheus_metrics(metrics)
                self._check_thresholds(metrics)
                
                await asyncio.sleep(self.config.sampling_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self._running = False

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network': self._get_network_stats(),
            'process': self._get_process_stats()
        }
        
        if self.config.enable_detailed_memory:
            metrics['memory_detailed'] = self._get_detailed_memory_stats()
        
        if self.gpu_available:
            metrics['gpu'] = self._get_gpu_stats()
        
        return metrics

    def _get_network_stats(self) -> Dict[str, Any]:
        """Collect network statistics"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }

    def _get_process_stats(self) -> Dict[str, Any]:
        """Collect current process statistics"""
        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'threads': process.num_threads(),
            'open_files': len(process.open_files())
        }

    def _get_detailed_memory_stats(self) -> Dict[str, Any]:
        """Collect detailed memory statistics"""
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            'total': vm.total,
            'available': vm.available,
            'used': vm.used,
            'free': vm.free,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_free': swap.free
        }

    def _get_gpu_stats(self) -> Dict[str, Any]:
        """Collect GPU statistics if available"""
        if not self.gpu_available:
            return {}
        
        try:
            import torch
            return {
                'device_count': torch.cuda.device_count(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_reserved': torch.cuda.memory_reserved(),
                'max_memory_allocated': torch.cuda.max_memory_allocated()
            }
        except Exception as e:
            logger.error(f"Error collecting GPU stats: {e}")
            return {}

    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in the history deque"""
        with self._lock:
            self.metrics_history.append(metrics)
        
        if self.config.log_to_file and self.config.log_file_path:
            try:
                with open(self.config.log_file_path, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
            except Exception as e:
                logger.error(f"Error writing to metrics log: {e}")

    def _update_prometheus_metrics(self, metrics: Dict[str, Any]):
        """Update Prometheus metrics if enabled"""
        if not self.config.enable_prometheus:
            return
        
        self.cpu_usage.set(metrics['cpu_percent'])
        self.memory_usage.set(metrics['memory_percent'])
        self.disk_usage.set(metrics['disk_percent'])
        
        if self.gpu_available and 'gpu' in metrics:
            gpu_stats = metrics['gpu']
            if gpu_stats:
                self.gpu_memory.set(
                    gpu_stats['memory_allocated'] / gpu_stats['memory_reserved']
                    if gpu_stats['memory_reserved'] else 0
                )

    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check if any metrics exceed configured thresholds"""
        for metric, threshold in self.config.alert_thresholds.items():
            current_value = self._get_metric_value(metrics, metric)
            if current_value > threshold:
                self._handle_threshold_exceeded(metric, current_value, threshold)

    def _get_metric_value(self, metrics: Dict[str, Any], metric_path: str) -> float:
        """Get metric value from nested dictionary using dot notation"""
        try:
            parts = metric_path.split('.')
            value = metrics
            for part in parts:
                value = value[part]
            return float(value)
        except (KeyError, ValueError, TypeError):
            return 0.0

    def _handle_threshold_exceeded(
        self,
        metric: str,
        value: float,
        threshold: float
    ):
        """Handle threshold exceeded events"""
        message = (
            f"Threshold exceeded for {metric}: "
            f"{value:.2f} > {threshold:.2f}"
        )
        logger.warning(message)
        # Additional alert handling could be added here

    def get_metrics_summary(
        self,
        duration: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get summary statistics for the specified duration"""
        with self._lock:
            if not self.metrics_history:
                return {}
            
            if duration:
                cutoff = datetime.now() - duration
                metrics = [
                    m for m in self.metrics_history
                    if datetime.fromisoformat(m['timestamp']) > cutoff
                ]
            else:
                metrics = list(self.metrics_history)
            
            if not metrics:
                return {}
            
            return {
                'duration': str(duration) if duration else 'all',
                'samples': len(metrics),
                'cpu_percent': {
                    'avg': sum(m['cpu_percent'] for m in metrics) / len(metrics),
                    'max': max(m['cpu_percent'] for m in metrics),
                    'min': min(m['cpu_percent'] for m in metrics)
                },
                'memory_percent': {
                    'avg': sum(m['memory_percent'] for m in metrics) / len(metrics),
                    'max': max(m['memory_percent'] for m in metrics),
                    'min': min(m['memory_percent'] for m in metrics)
                },
                'disk_percent': {
                    'avg': sum(m['disk_percent'] for m in metrics) / len(metrics),
                    'max': max(m['disk_percent'] for m in metrics),
                    'min': min(m['disk_percent'] for m in metrics)
                }
            }

    @contextmanager
    def measure_latency(self, operation: str):
        """Context manager to measure operation latency"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.query_performance.observe(duration)
            
            if duration > self.config.alert_thresholds.get('latency_ms', 1000) / 1000:
                logger.warning(
                    f"Operation {operation} took {duration:.2f}s, "
                    "exceeding threshold"
                )

    async def get_performance_report(
        self,
        duration: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        summary = self.get_metrics_summary(duration)
        if not summary:
            return {}
        
        # Add query performance statistics
        summary['query_performance'] = {
            'count': self.request_count._value.get(),
            'latency': {
                'avg': self.query_performance.count._value.get(),
                'sum': self.query_performance.sum._value.get()
            }
        }
        
        # Add recommendations based on metrics
        summary['recommendations'] = self._generate_recommendations(summary)
        
        return summary

    def _generate_recommendations(
        self,
        summary: Dict[str, Any]
    ) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # CPU recommendations
        if summary['cpu_percent']['avg'] > 70:
            recommendations.append(
                "High CPU usage detected. Consider scaling compute resources "
                "or optimizing compute-intensive operations."
            )
        
        # Memory recommendations
        if summary['memory_percent']['avg'] > 70:
            recommendations.append(
                "High memory usage detected. Consider increasing memory "
                "or implementing memory optimization strategies."
            )
        
        # Disk recommendations
        if summary['disk_percent']['avg'] > 80:
            recommendations.append(
                "High disk usage detected. Consider cleaning up unused data "
                "or adding more storage capacity."
            )
        
        # Query performance recommendations
        if 'query_performance' in summary:
            avg_latency = (
                summary['query_performance']['latency']['sum'] /
                summary['query_performance']['latency']['count']
                if summary['query_performance']['latency']['count'] > 0
                else 0
            )
            if avg_latency > 1.0:
                recommendations.append(
                    "High query latency detected. Consider optimizing query "
                    "patterns or implementing additional caching."
                )
        
        return recommendations
