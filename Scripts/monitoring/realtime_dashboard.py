"""
Real-time Performance Monitoring Dashboard
Advanced monitoring system with real-time metrics, alerts, and analytics.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import websockets
import psutil
import torch
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    
    # Monitoring settings
    sampling_interval: float = 1.0
    history_size: int = 1000
    enable_real_time: bool = True
    enable_alerts: bool = True
    
    # Dashboard settings
    dashboard_port: int = 8050
    dashboard_host: str = "localhost"
    websocket_port: int = 8765
    
    # Alert thresholds
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    gpu_memory_threshold: float = 90.0
    disk_threshold: float = 90.0
    latency_threshold: float = 1000.0  # milliseconds
    
    # Data retention
    metrics_retention_hours: int = 24
    detailed_retention_hours: int = 6
    
    # Performance tracking
    track_gpu_metrics: bool = True
    track_process_metrics: bool = True
    track_network_metrics: bool = True

class MetricsCollector:
    """Advanced metrics collection system."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_history = deque(maxlen=config.history_size)
        self.alert_history = deque(maxlen=100)
        self.collection_active = False
        self.collection_thread = None
        
        # Performance counters
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        
        # Alert system
        self.alert_callbacks: List[Callable] = []
        
    def start_collection(self):
        """Start metrics collection."""
        if not self.collection_active:
            self.collection_active = True
            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                daemon=True
            )
            self.collection_thread.start()
            logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.collection_active:
            try:
                metrics = self._collect_comprehensive_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                if self.config.enable_alerts:
                    self._check_alerts(metrics)
                
                time.sleep(self.config.sampling_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(1)
    
    def _collect_comprehensive_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        timestamp = datetime.now()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'timestamp': timestamp.isoformat(),
            'unix_timestamp': timestamp.timestamp(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
            }
        }
        
        # Network metrics
        if self.config.track_network_metrics:
            metrics['network'] = self._collect_network_metrics()
        
        # Process metrics
        if self.config.track_process_metrics:
            metrics['process'] = self._collect_process_metrics()
        
        # GPU metrics
        if self.config.track_gpu_metrics and torch.cuda.is_available():
            metrics['gpu'] = self._collect_gpu_metrics()
        
        # Application metrics
        metrics['application'] = {
            'request_count': self.request_count,
            'avg_latency_ms': (
                self.total_latency / self.request_count 
                if self.request_count > 0 else 0
            ),
            'error_count': self.error_count,
            'error_rate': (
                self.error_count / self.request_count 
                if self.request_count > 0 else 0
            )
        }
        
        return metrics
    
    def _collect_network_metrics(self) -> Dict[str, Any]:
        """Collect network metrics."""
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout,
            }
        except Exception:
            return {}
    
    def _collect_process_metrics(self) -> Dict[str, Any]:
        """Collect current process metrics."""
        try:
            process = psutil.Process()
            with process.oneshot():
                return {
                    'cpu_percent': process.cpu_percent(),
                    'memory_rss_mb': process.memory_info().rss / (1024**2),
                    'memory_vms_mb': process.memory_info().vms / (1024**2),
                    'memory_percent': process.memory_percent(),
                    'num_threads': process.num_threads(),
                    'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0,
                }
        except Exception:
            return {}
    
    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU metrics."""
        try:
            return {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                'memory_cached_mb': torch.cuda.memory_cached() / (1024**2),
                'max_memory_allocated_mb': torch.cuda.max_memory_allocated() / (1024**2),
                'utilization_percent': self._get_gpu_utilization(),
            }
        except Exception:
            return {}
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return 0.0
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions."""
        alerts = []
        
        system = metrics.get('system', {})
        
        # CPU alert
        if system.get('cpu_percent', 0) > self.config.cpu_threshold:
            alerts.append({
                'type': 'cpu',
                'severity': 'warning',
                'message': f"High CPU usage: {system['cpu_percent']:.1f}%",
                'value': system['cpu_percent'],
                'threshold': self.config.cpu_threshold
            })
        
        # Memory alert
        if system.get('memory_percent', 0) > self.config.memory_threshold:
            alerts.append({
                'type': 'memory',
                'severity': 'warning',
                'message': f"High memory usage: {system['memory_percent']:.1f}%",
                'value': system['memory_percent'],
                'threshold': self.config.memory_threshold
            })
        
        # Disk alert
        if system.get('disk_percent', 0) > self.config.disk_threshold:
            alerts.append({
                'type': 'disk',
                'severity': 'warning',
                'message': f"High disk usage: {system['disk_percent']:.1f}%",
                'value': system['disk_percent'],
                'threshold': self.config.disk_threshold
            })
        
        # GPU memory alert
        gpu = metrics.get('gpu', {})
        if gpu:
            gpu_memory_percent = (
                gpu.get('memory_allocated_mb', 0) / 
                (gpu.get('memory_reserved_mb', 1) or 1) * 100
            )
            if gpu_memory_percent > self.config.gpu_memory_threshold:
                alerts.append({
                    'type': 'gpu_memory',
                    'severity': 'warning',
                    'message': f"High GPU memory usage: {gpu_memory_percent:.1f}%",
                    'value': gpu_memory_percent,
                    'threshold': self.config.gpu_memory_threshold
                })
        
        # Application latency alert
        app = metrics.get('application', {})
        avg_latency = app.get('avg_latency_ms', 0)
        if avg_latency > self.config.latency_threshold:
            alerts.append({
                'type': 'latency',
                'severity': 'critical',
                'message': f"High response latency: {avg_latency:.1f}ms",
                'value': avg_latency,
                'threshold': self.config.latency_threshold
            })
        
        # Process alerts
        for alert in alerts:
            alert['timestamp'] = metrics['timestamp']
            self.alert_history.append(alert)
            
            # Notify alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def record_request(self, latency_ms: float, success: bool = True):
        """Record application request metrics."""
        self.request_count += 1
        self.total_latency += latency_ms
        if not success:
            self.error_count += 1
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_range(
        self, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get metrics within time range."""
        if not start_time:
            start_time = datetime.now() - timedelta(hours=1)
        if not end_time:
            end_time = datetime.now()
        
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()
        
        return [
            metrics for metrics in self.metrics_history
            if start_ts <= metrics['unix_timestamp'] <= end_ts
        ]

class RealTimeDashboard:
    """Real-time monitoring dashboard using Dash."""
    
    def __init__(self, metrics_collector: MetricsCollector, config: MonitoringConfig):
        self.metrics_collector = metrics_collector
        self.config = config
        self.app = None
        self.server_thread = None
        
    def create_app(self):
        """Create Dash application."""
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        self.app.layout = self._create_layout()
        self._register_callbacks()
        
        return self.app
    
    def _create_layout(self):
        """Create dashboard layout."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("RAG System Performance Monitor", className="text-center mb-4"),
                    html.Hr(),
                ], width=12)
            ]),
            
            # Alert banner
            dbc.Row([
                dbc.Col([
                    html.Div(id="alert-banner")
                ], width=12)
            ]),
            
            # Key metrics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("CPU Usage", className="card-title"),
                            html.H2(id="cpu-metric", className="text-primary"),
                        ])
                    ], className="mb-3")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Memory Usage", className="card-title"),
                            html.H2(id="memory-metric", className="text-info"),
                        ])
                    ], className="mb-3")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("GPU Memory", className="card-title"),
                            html.H2(id="gpu-metric", className="text-warning"),
                        ])
                    ], className="mb-3")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Avg Latency", className="card-title"),
                            html.H2(id="latency-metric", className="text-success"),
                        ])
                    ], className="mb-3")
                ], width=3),
            ]),
            
            # Charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="system-metrics-chart")
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id="gpu-metrics-chart")
                ], width=6),
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="application-metrics-chart")
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id="network-metrics-chart")
                ], width=6),
            ]),
            
            # Recent alerts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Alerts"),
                        dbc.CardBody([
                            html.Div(id="alerts-list")
                        ])
                    ])
                ], width=12)
            ], className="mt-4"),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=2000,  # 2 seconds
                n_intervals=0
            ),
            
        ], fluid=True)
    
    def _register_callbacks(self):
        """Register dashboard callbacks."""
        
        @self.app.callback(
            [
                Output('cpu-metric', 'children'),
                Output('memory-metric', 'children'),
                Output('gpu-metric', 'children'),
                Output('latency-metric', 'children'),
                Output('system-metrics-chart', 'figure'),
                Output('gpu-metrics-chart', 'figure'),
                Output('application-metrics-chart', 'figure'),
                Output('network-metrics-chart', 'figure'),
                Output('alerts-list', 'children'),
                Output('alert-banner', 'children'),
            ],
            Input('interval-component', 'n_intervals')
        )
        def update_dashboard(n):
            latest = self.metrics_collector.get_latest_metrics()
            
            if not latest:
                # Return empty values
                return (
                    "N/A", "N/A", "N/A", "N/A",
                    {}, {}, {}, {}, [], []
                )
            
            # Extract metrics
            system = latest.get('system', {})
            gpu = latest.get('gpu', {})
            app = latest.get('application', {})
            
            # Update metric cards
            cpu_text = f"{system.get('cpu_percent', 0):.1f}%"
            memory_text = f"{system.get('memory_percent', 0):.1f}%"
            gpu_text = f"{gpu.get('memory_allocated_mb', 0):.0f}MB" if gpu else "N/A"
            latency_text = f"{app.get('avg_latency_ms', 0):.1f}ms"
            
            # Get recent metrics for charts
            recent_metrics = list(self.metrics_collector.metrics_history)[-50:]
            
            # Create charts
            system_chart = self._create_system_chart(recent_metrics)
            gpu_chart = self._create_gpu_chart(recent_metrics)
            app_chart = self._create_application_chart(recent_metrics)
            network_chart = self._create_network_chart(recent_metrics)
            
            # Create alerts
            alerts_display = self._create_alerts_display()
            alert_banner = self._create_alert_banner()
            
            return (
                cpu_text, memory_text, gpu_text, latency_text,
                system_chart, gpu_chart, app_chart, network_chart,
                alerts_display, alert_banner
            )
    
    def _create_system_chart(self, metrics: List[Dict]) -> Dict:
        """Create system metrics chart."""
        if not metrics:
            return {}
        
        timestamps = [m['timestamp'] for m in metrics]
        cpu_data = [m.get('system', {}).get('cpu_percent', 0) for m in metrics]
        memory_data = [m.get('system', {}).get('memory_percent', 0) for m in metrics]
        disk_data = [m.get('system', {}).get('disk_percent', 0) for m in metrics]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=cpu_data,
            mode='lines', name='CPU %',
            line=dict(color='#1f77b4')
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=memory_data,
            mode='lines', name='Memory %',
            line=dict(color='#ff7f0e')
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=disk_data,
            mode='lines', name='Disk %',
            line=dict(color='#2ca02c')
        ))
        
        fig.update_layout(
            title="System Metrics",
            xaxis_title="Time",
            yaxis_title="Percentage",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _create_gpu_chart(self, metrics: List[Dict]) -> Dict:
        """Create GPU metrics chart."""
        if not metrics:
            return {}
        
        timestamps = [m['timestamp'] for m in metrics]
        gpu_memory = [
            m.get('gpu', {}).get('memory_allocated_mb', 0) 
            for m in metrics
        ]
        gpu_util = [
            m.get('gpu', {}).get('utilization_percent', 0) 
            for m in metrics
        ]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('GPU Memory (MB)', 'GPU Utilization (%)'),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=gpu_memory,
                mode='lines', name='Memory MB',
                line=dict(color='#d62728')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=gpu_util,
                mode='lines', name='Utilization %',
                line=dict(color='#9467bd')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="GPU Metrics",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _create_application_chart(self, metrics: List[Dict]) -> Dict:
        """Create application metrics chart."""
        if not metrics:
            return {}
        
        timestamps = [m['timestamp'] for m in metrics]
        request_count = [
            m.get('application', {}).get('request_count', 0) 
            for m in metrics
        ]
        latency = [
            m.get('application', {}).get('avg_latency_ms', 0) 
            for m in metrics
        ]
        error_rate = [
            m.get('application', {}).get('error_rate', 0) * 100 
            for m in metrics
        ]
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Request Count', 'Avg Latency (ms)', 'Error Rate (%)'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=request_count,
                mode='lines', name='Requests',
                line=dict(color='#17becf')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=latency,
                mode='lines', name='Latency',
                line=dict(color='#bcbd22')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=error_rate,
                mode='lines', name='Error Rate',
                line=dict(color='#e377c2')
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title="Application Metrics",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def _create_network_chart(self, metrics: List[Dict]) -> Dict:
        """Create network metrics chart."""
        if not metrics:
            return {}
        
        timestamps = [m['timestamp'] for m in metrics]
        bytes_sent = [
            m.get('network', {}).get('bytes_sent', 0) / (1024**2)  # MB
            for m in metrics
        ]
        bytes_recv = [
            m.get('network', {}).get('bytes_recv', 0) / (1024**2)  # MB
            for m in metrics
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=bytes_sent,
            mode='lines', name='Bytes Sent (MB)',
            line=dict(color='#8c564b')
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=bytes_recv,
            mode='lines', name='Bytes Received (MB)',
            line=dict(color='#7f7f7f')
        ))
        
        fig.update_layout(
            title="Network Metrics",
            xaxis_title="Time",
            yaxis_title="MB",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _create_alerts_display(self) -> List:
        """Create alerts display."""
        recent_alerts = list(self.metrics_collector.alert_history)[-10:]
        
        if not recent_alerts:
            return [html.P("No recent alerts", className="text-muted")]
        
        alert_items = []
        for alert in reversed(recent_alerts):  # Most recent first
            severity_color = {
                'warning': 'warning',
                'critical': 'danger',
                'info': 'info'
            }.get(alert.get('severity', 'info'), 'info')
            
            alert_items.append(
                dbc.Alert([
                    html.Strong(f"{alert['type'].upper()}: "),
                    alert['message'],
                    html.Small(
                        f" at {alert['timestamp'][:19]}",
                        className="text-muted"
                    )
                ], color=severity_color, className="mb-2")
            )
        
        return alert_items
    
    def _create_alert_banner(self) -> List:
        """Create alert banner for critical alerts."""
        recent_alerts = list(self.metrics_collector.alert_history)[-5:]
        critical_alerts = [
            a for a in recent_alerts 
            if a.get('severity') == 'critical'
        ]
        
        if not critical_alerts:
            return []
        
        latest_critical = critical_alerts[-1]
        return [
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Strong("CRITICAL ALERT: "),
                latest_critical['message']
            ], color="danger", dismissable=True)
        ]
    
    def start_server(self):
        """Start dashboard server."""
        if not self.app:
            self.create_app()
        
        def run_server():
            self.app.run_server(
                host=self.config.dashboard_host,
                port=self.config.dashboard_port,
                debug=False
            )
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        logger.info(
            f"Dashboard started at http://{self.config.dashboard_host}:"
            f"{self.config.dashboard_port}"
        )
    
    def stop_server(self):
        """Stop dashboard server."""
        # Note: Dash doesn't have a built-in way to stop server
        # This would need to be implemented with a proper WSGI server
        logger.info("Dashboard server stopped")

class PerformanceMonitoringSystem:
    """Main performance monitoring system."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.metrics_collector = MetricsCollector(self.config)
        self.dashboard = RealTimeDashboard(self.metrics_collector, self.config)
        self.websocket_server = None
        
        # Setup alert logging
        self.metrics_collector.add_alert_callback(self._log_alert)
    
    def start(self):
        """Start the monitoring system."""
        self.metrics_collector.start_collection()
        
        if self.config.enable_real_time:
            self.dashboard.start_server()
        
        logger.info("Performance monitoring system started")
    
    def stop(self):
        """Stop the monitoring system."""
        self.metrics_collector.stop_collection()
        self.dashboard.stop_server()
        
        logger.info("Performance monitoring system stopped")
    
    def _log_alert(self, alert: Dict[str, Any]):
        """Log alert to file and console."""
        severity = alert.get('severity', 'info').upper()
        message = alert.get('message', 'Unknown alert')
        
        if severity == 'CRITICAL':
            logger.critical(f"ALERT: {message}")
        elif severity == 'WARNING':
            logger.warning(f"ALERT: {message}")
        else:
            logger.info(f"ALERT: {message}")
    
    def record_request_metrics(self, latency_ms: float, success: bool = True):
        """Record request metrics for monitoring."""
        self.metrics_collector.record_request(latency_ms, success)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        return self.metrics_collector.get_latest_metrics()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        recent_metrics = list(self.metrics_collector.metrics_history)[-100:]
        
        if not recent_metrics:
            return {'message': 'No metrics available'}
        
        # Calculate averages
        avg_cpu = np.mean([
            m.get('system', {}).get('cpu_percent', 0) 
            for m in recent_metrics
        ])
        
        avg_memory = np.mean([
            m.get('system', {}).get('memory_percent', 0) 
            for m in recent_metrics
        ])
        
        avg_latency = np.mean([
            m.get('application', {}).get('avg_latency_ms', 0) 
            for m in recent_metrics
        ])
        
        total_requests = recent_metrics[-1].get('application', {}).get('request_count', 0)
        total_errors = recent_metrics[-1].get('application', {}).get('error_count', 0)
        
        return {
            'monitoring_period': f"{len(recent_metrics)} samples",
            'avg_cpu_percent': f"{avg_cpu:.1f}%",
            'avg_memory_percent': f"{avg_memory:.1f}%",
            'avg_latency_ms': f"{avg_latency:.1f}ms",
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': f"{(total_errors/total_requests*100):.2f}%" if total_requests > 0 else "0%",
            'recent_alerts': len(self.metrics_collector.alert_history),
            'dashboard_url': f"http://{self.config.dashboard_host}:{self.config.dashboard_port}"
        }

# Global monitoring system instance
_global_monitor: Optional[PerformanceMonitoringSystem] = None

def get_global_monitor() -> PerformanceMonitoringSystem:
    """Get or create global monitoring system."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitoringSystem()
    return _global_monitor

def initialize_monitoring(config: Optional[MonitoringConfig] = None):
    """Initialize global monitoring system."""
    global _global_monitor
    if _global_monitor is not None:
        _global_monitor.stop()
    
    _global_monitor = PerformanceMonitoringSystem(config)
    _global_monitor.start()

def shutdown_monitoring():
    """Shutdown global monitoring system."""
    global _global_monitor
    if _global_monitor is not None:
        _global_monitor.stop()
        _global_monitor = None
