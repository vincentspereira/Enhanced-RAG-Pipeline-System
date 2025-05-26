"""
Integration Manager for RAG System Enhancements
Coordinates all optimization components with the existing RAG pipeline
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import json
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancers.advanced_memory_optimizer import AdvancedMemoryOptimizer, AdvancedMemoryConfig
from enhancers.advanced_query_optimizer import AdvancedQueryOptimizer, OptimizationConfig, QueryAnalysis
from monitoring.realtime_dashboard import RealTimeDashboard, MonitoringConfig
try:
    from enhancers.query_cache import QueryCache
except ImportError:
    from caching.query_cache import QueryCache
from monitoring.performance_metrics import PerformanceMetrics

@dataclass
class IntegrationConfig:
    """Configuration for integration manager"""
    # Component enablement
    enable_memory_optimization: bool = True
    enable_query_optimization: bool = True
    enable_realtime_monitoring: bool = True
    enable_advanced_caching: bool = True
    
    # Performance thresholds
    memory_threshold: float = 0.85  # 85% memory usage threshold
    response_time_threshold: float = 2.0  # 2 seconds max response time
    cache_hit_ratio_threshold: float = 0.6  # 60% minimum cache hit ratio
    
    # Auto-optimization settings
    auto_optimization_interval: int = 300  # 5 minutes
    performance_window: int = 1800  # 30 minutes for performance analysis
    
    # Integration settings
    config_file: str = "config/integration_config.json"
    metrics_retention_days: int = 7
    optimization_history_size: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "enable_memory_optimization": self.enable_memory_optimization,
            "enable_query_optimization": self.enable_query_optimization,
            "enable_realtime_monitoring": self.enable_realtime_monitoring,
            "enable_advanced_caching": self.enable_advanced_caching,
            "memory_threshold": self.memory_threshold,
            "response_time_threshold": self.response_time_threshold,
            "cache_hit_ratio_threshold": self.cache_hit_ratio_threshold,
            "auto_optimization_interval": self.auto_optimization_interval,
            "performance_window": self.performance_window,
            "config_file": self.config_file,
            "metrics_retention_days": self.metrics_retention_days,
            "optimization_history_size": self.optimization_history_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntegrationConfig':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class SystemHealth:
    """System health metrics"""
    memory_usage: float
    cpu_usage: float
    gpu_usage: float
    cache_hit_ratio: float
    avg_response_time: float
    error_rate: float
    throughput: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_healthy(self, config: IntegrationConfig) -> bool:
        """Check if system is healthy based on thresholds"""
        return (
            self.memory_usage < config.memory_threshold and
            self.avg_response_time < config.response_time_threshold and
            self.cache_hit_ratio > config.cache_hit_ratio_threshold and
            self.error_rate < 0.05  # 5% error rate threshold
        )

class IntegrationManager:
    """
    Central manager for coordinating all RAG system enhancements
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Component instances
        self.memory_optimizer: Optional[AdvancedMemoryOptimizer] = None
        self.query_optimizer: Optional[AdvancedQueryOptimizer] = None
        self.dashboard: Optional[RealTimeDashboard] = None
        self.performance_metrics: Optional[PerformanceMetrics] = None
        self.query_cache: Optional[QueryCache] = None
        
        # Integration state
        self.is_initialized = False
        self.optimization_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Performance tracking
        self.system_health_history: List[SystemHealth] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Integration metrics
        self.integration_start_time = datetime.now()
        self.total_optimizations = 0
        self.successful_optimizations = 0
        
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            self.logger.info("Initializing Integration Manager...")
            
            # Initialize components based on configuration
            if self.config.enable_memory_optimization:
                await self._initialize_memory_optimizer()
            
            if self.config.enable_query_optimization:
                await self._initialize_query_optimizer()
            
            if self.config.enable_realtime_monitoring:
                await self._initialize_dashboard()
            
            if self.config.enable_advanced_caching:
                await self._initialize_query_cache()
            
            # Initialize performance metrics
            self.performance_metrics = PerformanceMetrics()
              # Start background optimization
            self._start_optimization_loop()
            
            # Start monitoring
            self._start_monitoring_loop()
            
            self.is_initialized = True
            self.logger.info("Integration Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Integration Manager: {e}")
            return False

    async def _initialize_memory_optimizer(self):
        """Initialize memory optimizer"""
        memory_config = AdvancedMemoryConfig(
            enable_gpu_memory_pool=True,
            enable_mixed_precision=True,
            enable_smart_caching=True,
            cache_compression_threshold=1024
        )
        self.memory_optimizer = AdvancedMemoryOptimizer(memory_config)
        await self.memory_optimizer.start()
        self.logger.info("Memory optimizer initialized")

    async def _initialize_query_optimizer(self):
        """Initialize query optimizer"""        query_config = OptimizationConfig(
            enable_intent_analysis=True,
            enable_semantic_expansion=True,
            enable_entity_extraction=True,
            enable_query_rewriting=True,
            enable_performance_optimization=True
        )
        self.query_optimizer = AdvancedQueryOptimizer(query_config)
        self.logger.info("Query optimizer initialized")
    
    async def _initialize_dashboard(self):
        """Initialize real-time dashboard"""
        from monitoring.realtime_dashboard import MonitoringConfig, MetricsCollector
        
        dashboard_config = MonitoringConfig(
            enable_gpu_monitoring=True,
            enable_network_monitoring=True,
            enable_alerts=True,
            dashboard_port=8051
        )
        
        # Create metrics collector first
        metrics_collector = MetricsCollector(dashboard_config)
        
        # Create dashboard with metrics collector
        self.dashboard = RealTimeDashboard(metrics_collector, dashboard_config)
        self.dashboard.start()
        self.logger.info("Real-time dashboard initialized")
    
    async def _initialize_query_cache(self):
        """Initialize enhanced query cache"""
        self.query_cache = QueryCache()
        self.logger.info("Query cache initialized")
    
    def _start_optimization_loop(self):
        """Start background optimization loop"""
        if self.optimization_thread and self.optimization_thread.is_alive():
            return
        
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True,
            name="OptimizationLoop"
        )
        self.optimization_thread.start()
        self.logger.info("Optimization loop started")
    
    def _start_monitoring_loop(self):
        """Start monitoring loop"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MonitoringLoop"
        )
        self.monitoring_thread.start()
        self.logger.info("Monitoring loop started")
    
    def _optimization_loop(self):
        """Background optimization loop"""
        while not self._shutdown_event.is_set():
            try:
                # Collect current metrics
                health = self._collect_system_health()
                
                # Analyze and optimize if needed
                if not health.is_healthy(self.config):
                    self._perform_optimization(health)
                
                # Wait for next optimization cycle
                self._shutdown_event.wait(self.config.auto_optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                self._shutdown_event.wait(60)  # Wait 1 minute before retry
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Collect and store health metrics
                health = self._collect_system_health()
                self.system_health_history.append(health)
                
                # Trim history to prevent memory bloat
                if len(self.system_health_history) > 1000:
                    self.system_health_history = self.system_health_history[-500:]
                
                # Update dashboard if available
                if self.dashboard:
                    asyncio.run_coroutine_threadsafe(
                        self.dashboard.update_metrics(health.__dict__),
                        asyncio.new_event_loop()
                    )
                
                # Wait for next monitoring cycle
                self._shutdown_event.wait(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._shutdown_event.wait(60)
    
    def _collect_system_health(self) -> SystemHealth:
        """Collect current system health metrics"""
        try:
            # Get basic system metrics
            if self.performance_metrics:
                metrics = self.performance_metrics.get_metrics()
            else:
                metrics = {}
            
            # Calculate cache hit ratio
            cache_hit_ratio = 0.0
            if self.query_cache:
                cache_stats = self.query_cache.get_stats()
                total_requests = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
                if total_requests > 0:
                    cache_hit_ratio = cache_stats.get('hits', 0) / total_requests
            
            return SystemHealth(
                memory_usage=metrics.get('memory_percent', 0.0) / 100.0,
                cpu_usage=metrics.get('cpu_percent', 0.0) / 100.0,
                gpu_usage=metrics.get('gpu_utilization', 0.0) / 100.0,
                cache_hit_ratio=cache_hit_ratio,
                avg_response_time=metrics.get('avg_response_time', 0.0),
                error_rate=metrics.get('error_rate', 0.0),
                throughput=metrics.get('requests_per_second', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system health: {e}")
            return SystemHealth(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _perform_optimization(self, health: SystemHealth):
        """Perform system optimization based on health metrics"""
        try:
            self.total_optimizations += 1
            optimization_start = time.time()
            
            optimizations_performed = []
            
            # Memory optimization
            if health.memory_usage > self.config.memory_threshold and self.memory_optimizer:
                self.memory_optimizer.optimize_memory()
                optimizations_performed.append("memory_cleanup")
            
            # Cache optimization
            if health.cache_hit_ratio < self.config.cache_hit_ratio_threshold and self.query_cache:
                self.query_cache.clear_expired()
                optimizations_performed.append("cache_cleanup")
            
            # GPU memory optimization
            if health.gpu_usage > 0.9 and self.memory_optimizer:
                self.memory_optimizer.optimize_gpu_memory()
                optimizations_performed.append("gpu_cleanup")
            
            optimization_time = time.time() - optimization_start
            
            # Record optimization
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "health_before": health.__dict__,
                "optimizations": optimizations_performed,
                "duration": optimization_time
            }
            
            self.optimization_history.append(optimization_record)
            
            # Trim optimization history
            if len(self.optimization_history) > self.config.optimization_history_size:
                self.optimization_history = self.optimization_history[-50:]
            
            self.successful_optimizations += 1
            self.logger.info(f"Optimization completed: {optimizations_performed}")
            
        except Exception as e:
            self.logger.error(f"Error performing optimization: {e}")
    
    async def optimize_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize a query using the integrated system"""
        try:
            optimization_start = time.time()
            
            # Use query optimizer if available
            if self.query_optimizer:
                analysis = await self.query_optimizer.analyze_query(query, context)
                optimized_query = analysis.optimized_query
                optimization_metadata = {
                    "intent": analysis.intent,
                    "query_type": analysis.query_type,
                    "complexity": analysis.complexity,
                    "entities": analysis.entities,
                    "expanded_terms": analysis.expanded_terms
                }
            else:
                optimized_query = query
                optimization_metadata = {}
            
            optimization_time = time.time() - optimization_start
            
            return {
                "original_query": query,
                "optimized_query": optimized_query,
                "optimization_time": optimization_time,
                "metadata": optimization_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing query: {e}")
            return {
                "original_query": query,
                "optimized_query": query,
                "optimization_time": 0.0,
                "metadata": {"error": str(e)}
            }
    
    async def get_cached_result(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available"""
        if self.query_cache:
            return await self.query_cache.get(query)
        return None
    
    async def cache_result(self, query: str, result: Dict[str, Any], ttl: Optional[int] = None):
        """Cache query result"""
        if self.query_cache:
            await self.query_cache.set(query, result, ttl)
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        uptime = datetime.now() - self.integration_start_time
        
        # Get latest health metrics
        latest_health = self.system_health_history[-1] if self.system_health_history else None
        
        # Calculate optimization success rate
        success_rate = 0.0
        if self.total_optimizations > 0:
            success_rate = self.successful_optimizations / self.total_optimizations
        
        return {
            "is_initialized": self.is_initialized,
            "uptime_seconds": uptime.total_seconds(),
            "total_optimizations": self.total_optimizations,
            "successful_optimizations": self.successful_optimizations,
            "optimization_success_rate": success_rate,
            "latest_health": latest_health.__dict__ if latest_health else None,
            "components_status": {
                "memory_optimizer": self.memory_optimizer is not None,
                "query_optimizer": self.query_optimizer is not None,
                "dashboard": self.dashboard is not None,
                "query_cache": self.query_cache is not None
            },
            "health_history_size": len(self.system_health_history),
            "optimization_history_size": len(self.optimization_history)
        }
    
    def get_health_trend(self, minutes: int = 30) -> Dict[str, List[float]]:
        """Get health trend for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_health = [h for h in self.system_health_history if h.timestamp > cutoff_time]
        
        if not recent_health:
            return {}
        
        return {
            "timestamps": [h.timestamp.isoformat() for h in recent_health],
            "memory_usage": [h.memory_usage for h in recent_health],
            "cpu_usage": [h.cpu_usage for h in recent_health],
            "gpu_usage": [h.gpu_usage for h in recent_health],
            "cache_hit_ratio": [h.cache_hit_ratio for h in recent_health],
            "response_time": [h.avg_response_time for h in recent_health],
            "error_rate": [h.error_rate for h in recent_health],
            "throughput": [h.throughput for h in recent_health]
        }
    
    async def save_config(self, filepath: Optional[str] = None):
        """Save current configuration"""
        config_path = filepath or self.config.config_file
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"Configuration saved to {config_path}")
    
    @classmethod
    async def load_config(cls, filepath: str) -> 'IntegrationManager':
        """Load configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            config = IntegrationConfig.from_dict(config_data)
            return cls(config)
            
        except FileNotFoundError:
            logging.warning(f"Config file {filepath} not found, using defaults")
            return cls()
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return cls()
    
    async def shutdown(self):
        """Shutdown the integration manager"""
        self.logger.info("Shutting down Integration Manager...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for threads to finish
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5)
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)        # Shutdown components
        if self.memory_optimizer:
            await self.memory_optimizer.stop()
        
        if self.dashboard:
            self.dashboard.stop()
        
        self.logger.info("Integration Manager shutdown complete")

# Global integration manager instance
_integration_manager: Optional[IntegrationManager] = None

async def get_integration_manager() -> IntegrationManager:
    """Get global integration manager instance"""
    global _integration_manager
    
    if _integration_manager is None:
        _integration_manager = await IntegrationManager.load_config("config/integration_config.json")
        await _integration_manager.initialize()
    
    return _integration_manager

async def initialize_integration() -> bool:
    """Initialize the integration system"""
    try:
        manager = await get_integration_manager()
        return manager.is_initialized
    except Exception as e:
        logging.error(f"Failed to initialize integration: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create integration manager
        manager = IntegrationManager()
        
        # Initialize
        success = await manager.initialize()
        print(f"Initialization: {'Success' if success else 'Failed'}")
        
        if success:
            # Test query optimization
            result = await manager.optimize_query("What are the latest trends in AI?")
            print(f"Query optimization result: {result}")
            
            # Get stats
            stats = manager.get_integration_stats()
            print(f"Integration stats: {stats}")
            
            # Wait a bit for monitoring
            await asyncio.sleep(5)
            
            # Get health trend
            trend = manager.get_health_trend(5)
            print(f"Health trend: {len(trend.get('timestamps', []))} data points")
        
        # Shutdown
        await manager.shutdown()
    
    asyncio.run(main())
