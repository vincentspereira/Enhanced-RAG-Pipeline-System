"""
Auto-scaling System for RAG Pipeline
Dynamically adjusts system resources based on load and performance metrics
"""

import asyncio
import logging
import threading
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import psutil
import subprocess
import os
import yaml

try:
    import torch
    import pynvml
    TORCH_AVAILABLE = True
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    GPU_MONITORING_AVAILABLE = False

class ScalingAction(Enum):
    """Types of scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    OPTIMIZE = "optimize"
    MAINTAIN = "maintain"

class ResourceType(Enum):
    """Types of resources that can be scaled"""
    CPU_WORKERS = "cpu_workers"
    BATCH_SIZE = "batch_size"
    MEMORY_CACHE = "memory_cache"
    GPU_MEMORY = "gpu_memory"
    CONNECTION_POOL = "connection_pool"
    EMBEDDING_MODEL = "embedding_model"

@dataclass
class ScalingThreshold:
    """Thresholds for scaling decisions"""
    metric_name: str
    scale_up_threshold: float
    scale_down_threshold: float
    resource_type: ResourceType
    action_cooldown: int = 300  # 5 minutes
    min_value: int = 1
    max_value: int = 100

@dataclass
class ScalingConfig:
    """Configuration for auto-scaling system"""
    # Enable/disable scaling
    enable_auto_scaling: bool = True
    enable_cpu_scaling: bool = True
    enable_memory_scaling: bool = True
    enable_gpu_scaling: bool = True
    enable_model_scaling: bool = True
    
    # Monitoring intervals
    monitoring_interval: int = 30  # seconds
    scaling_decision_interval: int = 60  # seconds
    metrics_retention_minutes: int = 60
    
    # Safety limits
    max_cpu_workers: int = 16
    min_cpu_workers: int = 2
    max_batch_size: int = 128
    min_batch_size: int = 8
    max_memory_cache_mb: int = 8192
    min_memory_cache_mb: int = 512
    
    # Performance targets
    target_cpu_utilization: float = 0.7  # 70%
    target_memory_utilization: float = 0.8  # 80%
    target_gpu_utilization: float = 0.75  # 75%
    target_response_time: float = 1.5  # 1.5 seconds
    
    # Advanced settings
    scaling_sensitivity: float = 1.0  # Scaling aggressiveness
    prediction_window_minutes: int = 15
    load_prediction_enabled: bool = True
    adaptive_thresholds: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_auto_scaling": self.enable_auto_scaling,
            "enable_cpu_scaling": self.enable_cpu_scaling,
            "enable_memory_scaling": self.enable_memory_scaling,
            "enable_gpu_scaling": self.enable_gpu_scaling,
            "enable_model_scaling": self.enable_model_scaling,
            "monitoring_interval": self.monitoring_interval,
            "scaling_decision_interval": self.scaling_decision_interval,
            "metrics_retention_minutes": self.metrics_retention_minutes,
            "max_cpu_workers": self.max_cpu_workers,
            "min_cpu_workers": self.min_cpu_workers,
            "max_batch_size": self.max_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_memory_cache_mb": self.max_memory_cache_mb,
            "min_memory_cache_mb": self.min_memory_cache_mb,
            "target_cpu_utilization": self.target_cpu_utilization,
            "target_memory_utilization": self.target_memory_utilization,
            "target_gpu_utilization": self.target_gpu_utilization,
            "target_response_time": self.target_response_time,
            "scaling_sensitivity": self.scaling_sensitivity,
            "prediction_window_minutes": self.prediction_window_minutes,
            "load_prediction_enabled": self.load_prediction_enabled,
            "adaptive_thresholds": self.adaptive_thresholds
        }

@dataclass
class SystemMetrics:
    """Current system metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    gpu_utilization: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    queue_size: int
    avg_response_time: float
    requests_per_second: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_available_mb": self.memory_available_mb,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "disk_io_read_mb": self.disk_io_read_mb,
            "disk_io_write_mb": self.disk_io_write_mb,
            "network_sent_mb": self.network_sent_mb,
            "network_recv_mb": self.network_recv_mb,
            "active_connections": self.active_connections,
            "queue_size": self.queue_size,
            "avg_response_time": self.avg_response_time,
            "requests_per_second": self.requests_per_second,
            "error_rate": self.error_rate
        }

@dataclass
class ScalingDecision:
    """Represents a scaling decision"""
    action: ScalingAction
    resource_type: ResourceType
    current_value: int
    target_value: int
    reason: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "resource_type": self.resource_type.value,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "reason": self.reason,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }

class ResourceManager:
    """Manages system resources that can be scaled"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_resources = {
            ResourceType.CPU_WORKERS: 4,
            ResourceType.BATCH_SIZE: 32,
            ResourceType.MEMORY_CACHE: 2048,  # MB
            ResourceType.GPU_MEMORY: 4096,   # MB
            ResourceType.CONNECTION_POOL: 10
        }
        self.resource_locks = {rt: threading.Lock() for rt in ResourceType}
    
    async def get_current_value(self, resource_type: ResourceType) -> int:
        """Get current value for a resource"""
        return self.current_resources.get(resource_type, 0)
    
    async def set_resource_value(self, resource_type: ResourceType, value: int) -> bool:
        """Set resource value"""
        try:
            with self.resource_locks[resource_type]:
                old_value = self.current_resources.get(resource_type, 0)
                
                if resource_type == ResourceType.CPU_WORKERS:
                    success = await self._scale_cpu_workers(value)
                elif resource_type == ResourceType.BATCH_SIZE:
                    success = await self._scale_batch_size(value)
                elif resource_type == ResourceType.MEMORY_CACHE:
                    success = await self._scale_memory_cache(value)
                elif resource_type == ResourceType.GPU_MEMORY:
                    success = await self._scale_gpu_memory(value)
                elif resource_type == ResourceType.CONNECTION_POOL:
                    success = await self._scale_connection_pool(value)
                else:
                    success = False
                
                if success:
                    self.current_resources[resource_type] = value
                    self.logger.info(f"Scaled {resource_type.value} from {old_value} to {value}")
                else:
                    self.logger.warning(f"Failed to scale {resource_type.value} to {value}")
                
                return success
                
        except Exception as e:
            self.logger.error(f"Error scaling {resource_type.value}: {e}")
            return False
    
    async def _scale_cpu_workers(self, workers: int) -> bool:
        """Scale CPU worker processes/threads"""
        try:
            # This would integrate with your specific worker management system
            # For now, we'll simulate by setting environment variables
            os.environ['RAG_CPU_WORKERS'] = str(workers)
            return True
        except Exception as e:
            self.logger.error(f"Error scaling CPU workers: {e}")
            return False
    
    async def _scale_batch_size(self, batch_size: int) -> bool:
        """Scale processing batch size"""
        try:
            os.environ['RAG_BATCH_SIZE'] = str(batch_size)
            return True
        except Exception as e:
            self.logger.error(f"Error scaling batch size: {e}")
            return False
    
    async def _scale_memory_cache(self, cache_mb: int) -> bool:
        """Scale memory cache size"""
        try:
            os.environ['RAG_MEMORY_CACHE_MB'] = str(cache_mb)
            return True
        except Exception as e:
            self.logger.error(f"Error scaling memory cache: {e}")
            return False
    
    async def _scale_gpu_memory(self, memory_mb: int) -> bool:
        """Scale GPU memory allocation"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # This would involve model reloading with different memory settings
                os.environ['RAG_GPU_MEMORY_MB'] = str(memory_mb)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error scaling GPU memory: {e}")
            return False
    
    async def _scale_connection_pool(self, pool_size: int) -> bool:
        """Scale database/API connection pool"""
        try:
            os.environ['RAG_CONNECTION_POOL_SIZE'] = str(pool_size)
            return True
        except Exception as e:
            self.logger.error(f"Error scaling connection pool: {e}")
            return False

class MetricsCollector:
    """Collects system metrics for auto-scaling decisions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._last_disk_io = None
        self._last_network_io = None
        
        # Initialize GPU monitoring if available
        self.gpu_available = False
        if GPU_MONITORING_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = self.gpu_count > 0
            except Exception as e:
                self.logger.warning(f"GPU monitoring not available: {e}")
    
    async def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            now = datetime.now()
            
            # CPU and Memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            disk_read_mb = 0.0
            disk_write_mb = 0.0
            
            if self._last_disk_io and disk_io:
                disk_read_mb = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024 * 1024)
                disk_write_mb = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024 * 1024)
            
            self._last_disk_io = disk_io
            
            # Network I/O metrics
            network_io = psutil.net_io_counters()
            network_sent_mb = 0.0
            network_recv_mb = 0.0
            
            if self._last_network_io and network_io:
                network_sent_mb = (network_io.bytes_sent - self._last_network_io.bytes_sent) / (1024 * 1024)
                network_recv_mb = (network_io.bytes_recv - self._last_network_io.bytes_recv) / (1024 * 1024)
            
            self._last_network_io = network_io
            
            # GPU metrics
            gpu_utilization = 0.0
            gpu_memory_used_mb = 0.0
            gpu_memory_total_mb = 0.0
            
            if self.gpu_available:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use first GPU
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = gpu_util.gpu
                    
                    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used_mb = gpu_memory.used / (1024 * 1024)
                    gpu_memory_total_mb = gpu_memory.total / (1024 * 1024)
                except Exception as e:
                    self.logger.debug(f"GPU metrics collection error: {e}")
            
            # Connection metrics (simulated - would integrate with actual connection pools)
            active_connections = len(psutil.net_connections())
            
            # Application metrics (would be provided by the application)
            queue_size = 0  # Would come from task queue
            avg_response_time = 0.5  # Would come from request metrics
            requests_per_second = 10.0  # Would come from request metrics
            error_rate = 0.01  # Would come from error tracking
            
            return SystemMetrics(
                timestamp=now,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / (1024 * 1024),
                gpu_utilization=gpu_utilization,
                gpu_memory_used_mb=gpu_memory_used_mb,
                gpu_memory_total_mb=gpu_memory_total_mb,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                active_connections=active_connections,
                queue_size=queue_size,
                avg_response_time=avg_response_time,
                requests_per_second=requests_per_second,
                error_rate=error_rate
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            # Return default metrics
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0, memory_percent=0.0, memory_available_mb=0.0,
                gpu_utilization=0.0, gpu_memory_used_mb=0.0, gpu_memory_total_mb=0.0,
                disk_io_read_mb=0.0, disk_io_write_mb=0.0,
                network_sent_mb=0.0, network_recv_mb=0.0,
                active_connections=0, queue_size=0,
                avg_response_time=0.0, requests_per_second=0.0, error_rate=0.0
            )

class LoadPredictor:
    """Predicts future load based on historical metrics"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.metrics_history: List[SystemMetrics] = []
        self.logger = logging.getLogger(__name__)
    
    def add_metrics(self, metrics: SystemMetrics):
        """Add metrics to history"""
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > self.window_size * 2:
            self.metrics_history = self.metrics_history[-self.window_size:]
    
    def predict_load(self, minutes_ahead: int = 5) -> Dict[str, float]:
        """Predict system load N minutes into the future"""
        if len(self.metrics_history) < 3:
            return {}
        
        try:
            # Simple linear prediction based on trend
            recent_metrics = self.metrics_history[-min(10, len(self.metrics_history)):]
            
            # Calculate trends
            cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
            gpu_trend = self._calculate_trend([m.gpu_utilization for m in recent_metrics])
            response_time_trend = self._calculate_trend([m.avg_response_time for m in recent_metrics])
            
            # Current values
            latest = self.metrics_history[-1]
            
            # Predict future values
            prediction_factor = minutes_ahead / 5.0  # Scale by time horizon
            
            predicted_cpu = max(0, min(100, latest.cpu_percent + (cpu_trend * prediction_factor)))
            predicted_memory = max(0, min(100, latest.memory_percent + (memory_trend * prediction_factor)))
            predicted_gpu = max(0, min(100, latest.gpu_utilization + (gpu_trend * prediction_factor)))
            predicted_response_time = max(0, latest.avg_response_time + (response_time_trend * prediction_factor))
            
            return {
                "cpu_percent": predicted_cpu,
                "memory_percent": predicted_memory,
                "gpu_utilization": predicted_gpu,
                "avg_response_time": predicted_response_time,
                "confidence": min(1.0, len(recent_metrics) / 10.0)  # Confidence based on data availability
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting load: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) from values"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / denominator
        return slope

class AutoScaler:
    """Main auto-scaling system"""
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.resource_manager = ResourceManager()
        self.metrics_collector = MetricsCollector()
        self.load_predictor = LoadPredictor()
        
        # State
        self.metrics_history: List[SystemMetrics] = []
        self.scaling_history: List[ScalingDecision] = []
        self.last_scaling_actions: Dict[ResourceType, datetime] = {}
        
        # Threading
        self.monitoring_thread: Optional[threading.Thread] = None
        self.scaling_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Statistics
        self.total_scaling_actions = 0
        self.successful_scaling_actions = 0
        
        # Initialize thresholds
        self.scaling_thresholds = self._initialize_thresholds()
    
    def _initialize_thresholds(self) -> List[ScalingThreshold]:
        """Initialize scaling thresholds"""
        return [
            ScalingThreshold(
                metric_name="cpu_percent",
                scale_up_threshold=self.config.target_cpu_utilization * 100 + 15,
                scale_down_threshold=self.config.target_cpu_utilization * 100 - 15,
                resource_type=ResourceType.CPU_WORKERS,
                min_value=self.config.min_cpu_workers,
                max_value=self.config.max_cpu_workers
            ),
            ScalingThreshold(
                metric_name="memory_percent",
                scale_up_threshold=self.config.target_memory_utilization * 100 + 10,
                scale_down_threshold=self.config.target_memory_utilization * 100 - 20,
                resource_type=ResourceType.MEMORY_CACHE,
                min_value=self.config.min_memory_cache_mb,
                max_value=self.config.max_memory_cache_mb
            ),
            ScalingThreshold(
                metric_name="gpu_utilization",
                scale_up_threshold=self.config.target_gpu_utilization * 100 + 15,
                scale_down_threshold=self.config.target_gpu_utilization * 100 - 15,
                resource_type=ResourceType.BATCH_SIZE,
                min_value=self.config.min_batch_size,
                max_value=self.config.max_batch_size
            ),
            ScalingThreshold(
                metric_name="avg_response_time",
                scale_up_threshold=self.config.target_response_time + 0.5,
                scale_down_threshold=self.config.target_response_time - 0.2,
                resource_type=ResourceType.CPU_WORKERS,
                min_value=self.config.min_cpu_workers,
                max_value=self.config.max_cpu_workers
            )
        ]
    
    async def start(self) -> bool:
        """Start the auto-scaling system"""
        try:
            if not self.config.enable_auto_scaling:
                self.logger.info("Auto-scaling is disabled")
                return False
            
            self.logger.info("Starting auto-scaling system...")
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="AutoScalerMonitoring"
            )
            self.monitoring_thread.start()
            
            # Start scaling thread
            self.scaling_thread = threading.Thread(
                target=self._scaling_loop,
                daemon=True,
                name="AutoScalerScaling"
            )
            self.scaling_thread.start()
            
            self.logger.info("Auto-scaling system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start auto-scaling system: {e}")
            return False
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = asyncio.run(self.metrics_collector.collect_metrics())
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Add to load predictor
                self.load_predictor.add_metrics(metrics)
                
                # Trim history
                retention_seconds = self.config.metrics_retention_minutes * 60
                cutoff_time = datetime.now() - timedelta(seconds=retention_seconds)
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp > cutoff_time
                ]
                
                # Wait for next monitoring cycle
                self._shutdown_event.wait(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._shutdown_event.wait(60)
    
    def _scaling_loop(self):
        """Background scaling decision loop"""
        while not self._shutdown_event.is_set():
            try:
                # Make scaling decisions
                asyncio.run(self._make_scaling_decisions())
                
                # Wait for next scaling cycle
                self._shutdown_event.wait(self.config.scaling_decision_interval)
                
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                self._shutdown_event.wait(60)
    
    async def _make_scaling_decisions(self):
        """Make scaling decisions based on current metrics"""
        if not self.metrics_history:
            return
        
        latest_metrics = self.metrics_history[-1]
        
        # Get load prediction if enabled
        prediction = {}
        if self.config.load_prediction_enabled:
            prediction = self.load_predictor.predict_load(self.config.prediction_window_minutes)
        
        # Evaluate each threshold
        for threshold in self.scaling_thresholds:
            try:
                decision = await self._evaluate_threshold(threshold, latest_metrics, prediction)
                
                if decision and decision.action != ScalingAction.MAINTAIN:
                    success = await self._execute_scaling_decision(decision)
                    
                    if success:
                        self.successful_scaling_actions += 1
                        self.scaling_history.append(decision)
                        self.last_scaling_actions[decision.resource_type] = datetime.now()
                    
                    self.total_scaling_actions += 1
                    
                    # Trim scaling history
                    if len(self.scaling_history) > 100:
                        self.scaling_history = self.scaling_history[-50:]
                
            except Exception as e:
                self.logger.error(f"Error evaluating threshold {threshold.metric_name}: {e}")
    
    async def _evaluate_threshold(
        self, 
        threshold: ScalingThreshold, 
        metrics: SystemMetrics, 
        prediction: Dict[str, float]
    ) -> Optional[ScalingDecision]:
        """Evaluate a scaling threshold and return decision"""
        
        # Check cooldown
        if threshold.resource_type in self.last_scaling_actions:
            last_action = self.last_scaling_actions[threshold.resource_type]
            if (datetime.now() - last_action).total_seconds() < threshold.action_cooldown:
                return None
        
        # Get current metric value
        current_value = getattr(metrics, threshold.metric_name, 0.0)
        
        # Get predicted value if available
        predicted_value = prediction.get(threshold.metric_name, current_value)
        
        # Use prediction or current value based on confidence
        metric_value = current_value
        if prediction.get('confidence', 0) > 0.7:
            metric_value = predicted_value
        
        # Get current resource value
        current_resource_value = await self.resource_manager.get_current_value(threshold.resource_type)
        
        # Determine scaling action
        action = ScalingAction.MAINTAIN
        target_value = current_resource_value
        reason = "Within thresholds"
        confidence = 0.5
        
        if metric_value > threshold.scale_up_threshold:
            if current_resource_value < threshold.max_value:
                action = ScalingAction.SCALE_UP
                # Calculate scaling factor based on how far above threshold
                scale_factor = min(2.0, 1 + ((metric_value - threshold.scale_up_threshold) / threshold.scale_up_threshold))
                target_value = min(threshold.max_value, int(current_resource_value * scale_factor))
                reason = f"{threshold.metric_name} ({metric_value:.1f}) above threshold ({threshold.scale_up_threshold:.1f})"
                confidence = min(1.0, (metric_value - threshold.scale_up_threshold) / (threshold.scale_up_threshold * 0.2))
        
        elif metric_value < threshold.scale_down_threshold:
            if current_resource_value > threshold.min_value:
                action = ScalingAction.SCALE_DOWN
                # Calculate scaling factor
                scale_factor = max(0.5, (metric_value / threshold.scale_down_threshold))
                target_value = max(threshold.min_value, int(current_resource_value * scale_factor))
                reason = f"{threshold.metric_name} ({metric_value:.1f}) below threshold ({threshold.scale_down_threshold:.1f})"
                confidence = min(1.0, (threshold.scale_down_threshold - metric_value) / (threshold.scale_down_threshold * 0.2))
        
        # Apply scaling sensitivity
        confidence *= self.config.scaling_sensitivity
        
        # Only proceed if confidence is high enough
        if confidence < 0.3:
            return None
        
        return ScalingDecision(
            action=action,
            resource_type=threshold.resource_type,
            current_value=current_resource_value,
            target_value=target_value,
            reason=reason,
            confidence=confidence
        )
    
    async def _execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision"""
        try:
            self.logger.info(f"Executing scaling decision: {decision.action.value} "
                           f"{decision.resource_type.value} from {decision.current_value} "
                           f"to {decision.target_value} (confidence: {decision.confidence:.2f})")
            
            success = await self.resource_manager.set_resource_value(
                decision.resource_type, 
                decision.target_value
            )
            
            if success:
                self.logger.info(f"Successfully scaled {decision.resource_type.value} to {decision.target_value}")
            else:
                self.logger.warning(f"Failed to scale {decision.resource_type.value}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing scaling decision: {e}")
            return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        success_rate = 0.0
        if self.total_scaling_actions > 0:
            success_rate = self.successful_scaling_actions / self.total_scaling_actions
        
        # Get latest metrics
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        # Get current resource values
        current_resources = {}
        for rt in ResourceType:
            try:
                current_resources[rt.value] = asyncio.run(self.resource_manager.get_current_value(rt))
            except:
                current_resources[rt.value] = 0
        
        return {
            "enabled": self.config.enable_auto_scaling,
            "total_scaling_actions": self.total_scaling_actions,
            "successful_scaling_actions": self.successful_scaling_actions,
            "success_rate": success_rate,
            "metrics_history_size": len(self.metrics_history),
            "scaling_history_size": len(self.scaling_history),
            "current_resources": current_resources,
            "latest_metrics": latest_metrics.to_dict() if latest_metrics else None,
            "prediction_enabled": self.config.load_prediction_enabled,
            "gpu_monitoring_available": GPU_MONITORING_AVAILABLE
        }
    
    def get_recent_decisions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent scaling decisions"""
        recent = self.scaling_history[-count:] if self.scaling_history else []
        return [d.to_dict() for d in recent]
    
    async def force_scaling_action(self, resource_type: ResourceType, target_value: int) -> bool:
        """Force a specific scaling action"""
        try:
            current_value = await self.resource_manager.get_current_value(resource_type)
            
            decision = ScalingDecision(
                action=ScalingAction.SCALE_UP if target_value > current_value else ScalingAction.SCALE_DOWN,
                resource_type=resource_type,
                current_value=current_value,
                target_value=target_value,
                reason="Manual scaling action",
                confidence=1.0
            )
            
            return await self._execute_scaling_decision(decision)
            
        except Exception as e:
            self.logger.error(f"Error in forced scaling action: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the auto-scaling system"""
        self.logger.info("Shutting down auto-scaling system...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for threads to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        if self.scaling_thread and self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=5)
        
        self.logger.info("Auto-scaling system shutdown complete")

# Global auto-scaler instance
_auto_scaler: Optional[AutoScaler] = None

async def get_auto_scaler() -> AutoScaler:
    """Get global auto-scaler instance"""
    global _auto_scaler
    
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
        await _auto_scaler.start()
    
    return _auto_scaler

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create auto-scaler
        config = ScalingConfig(
            enable_auto_scaling=True,
            monitoring_interval=10,
            scaling_decision_interval=20
        )
        
        scaler = AutoScaler(config)
        
        # Start auto-scaling
        success = await scaler.start()
        print(f"Auto-scaler started: {'Success' if success else 'Failed'}")
        
        if success:
            # Let it run for a bit
            await asyncio.sleep(30)
            
            # Get stats
            stats = scaler.get_scaling_stats()
            print(f"Scaling stats: {json.dumps(stats, indent=2)}")
            
            # Test forced scaling
            await scaler.force_scaling_action(ResourceType.CPU_WORKERS, 6)
            
            # Get recent decisions
            decisions = scaler.get_recent_decisions(5)
            print(f"Recent decisions: {json.dumps(decisions, indent=2)}")
        
        # Shutdown
        await scaler.shutdown()
    
    asyncio.run(main())
