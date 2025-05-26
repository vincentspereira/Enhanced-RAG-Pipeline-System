"""
Advanced Memory Optimization System for RAG Pipeline
Implements intelligent memory management, GPU optimization, and resource allocation.
"""

import os
import gc
import psutil
import torch
import numpy as np
import asyncio
import threading
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import time
from pathlib import Path
import json
import pickle
import weakref
from contextlib import contextmanager
import mmap
import tempfile

logger = logging.getLogger(__name__)

@dataclass
class AdvancedMemoryConfig:
    """Configuration for advanced memory optimization."""
    
    # Memory thresholds
    max_system_memory_usage: float = 0.85  # Maximum system memory usage
    max_gpu_memory_usage: float = 0.80     # Maximum GPU memory usage
    memory_cleanup_threshold: float = 0.90  # Trigger cleanup when exceeded
    
    # GPU optimization
    enable_gpu_memory_pool: bool = True
    gpu_memory_fraction: float = 0.7
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    
    # Async processing
    max_concurrent_tasks: int = 8
    task_queue_size: int = 1000
    enable_background_optimization: bool = True
    
    # Caching
    enable_smart_caching: bool = True
    cache_compression_threshold: int = 1024  # bytes
    max_cache_memory_mb: int = 2048
    
    # Monitoring
    monitoring_interval: float = 1.0  # seconds
    enable_detailed_profiling: bool = True
    profile_save_interval: int = 300  # seconds

class GPUMemoryManager:
    """Advanced GPU memory management with optimization."""
    
    def __init__(self, config: AdvancedMemoryConfig):
        self.config = config
        self.cuda_available = torch.cuda.is_available()
        self.memory_pool = None
        self.streams = []
        
        if self.cuda_available:
            self._init_gpu_optimization()
    
    def _init_gpu_optimization(self):
        """Initialize GPU optimizations."""
        try:
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            
            # Enable optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Create CUDA streams for parallel processing
            self.streams = [torch.cuda.Stream() for _ in range(4)]
            
            # Initialize memory pool if enabled
            if self.config.enable_gpu_memory_pool:
                self._init_memory_pool()
            
            logger.info("GPU optimization initialized successfully")
        except Exception as e:
            logger.error(f"GPU optimization failed: {e}")
    
    def _init_memory_pool(self):
        """Initialize custom GPU memory pool."""
        try:
            # Custom memory allocator for better GPU memory management
            torch.cuda.empty_cache()
            self.memory_pool = torch.cuda.memory.MemoryPool()
            logger.info("GPU memory pool initialized")
        except Exception as e:
            logger.warning(f"Memory pool initialization failed: {e}")
    
    @contextmanager
    def gpu_memory_context(self, stream_idx: int = 0):
        """Context manager for GPU memory optimization."""
        if not self.cuda_available:
            yield
            return
        
        stream = self.streams[stream_idx] if self.streams else torch.cuda.current_stream()
        
        with torch.cuda.stream(stream):
            if self.config.enable_mixed_precision:
                with torch.cuda.amp.autocast():
                    yield
            else:
                yield
    
    def optimize_gpu_memory(self):
        """Optimize GPU memory usage."""
        if not self.cuda_available:
            return
        
        try:
            # Clear unused cache
            torch.cuda.empty_cache()
            
            # Get memory stats
            memory_stats = torch.cuda.memory_stats()
            allocated = memory_stats.get('allocated_bytes.all.current', 0)
            reserved = memory_stats.get('reserved_bytes.all.current', 0)
            
            # Calculate usage
            total_memory = torch.cuda.get_device_properties(0).total_memory
            usage_ratio = allocated / total_memory
            
            if usage_ratio > self.config.max_gpu_memory_usage:
                self._aggressive_gpu_cleanup()
            
            logger.debug(f"GPU memory usage: {usage_ratio:.2%}")
        except Exception as e:
            logger.error(f"GPU memory optimization failed: {e}")
    
    def _aggressive_gpu_cleanup(self):
        """Aggressive GPU memory cleanup."""
        try:
            # Clear all caches
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            logger.info("Aggressive GPU cleanup completed")
        except Exception as e:
            logger.error(f"Aggressive GPU cleanup failed: {e}")

class SystemMemoryManager:
    """Advanced system memory management."""
    
    def __init__(self, config: AdvancedMemoryConfig):
        self.config = config
        self.temp_files: List[Path] = []
        self.memory_mappings: Dict[str, Any] = {}
        self.weak_refs: weakref.WeakSet = weakref.WeakSet()
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics."""
        process = psutil.Process()
        system = psutil.virtual_memory()
        
        stats = {
            'system_total_gb': system.total / (1024**3),
            'system_used_gb': system.used / (1024**3),
            'system_available_gb': system.available / (1024**3),
            'system_usage_percent': system.percent,
            'process_rss_mb': process.memory_info().rss / (1024**2),
            'process_vms_mb': process.memory_info().vms / (1024**2),
            'process_percent': process.memory_percent(),
        }
        
        # Add swap information
        swap = psutil.swap_memory()
        stats.update({
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3),
            'swap_percent': swap.percent,
        })
        
        return stats
    
    def should_optimize_memory(self) -> bool:
        """Check if memory optimization is needed."""
        stats = self.get_memory_stats()
        return (
            stats['system_usage_percent'] > self.config.memory_cleanup_threshold * 100 or
            stats['process_percent'] > 50.0
        )
    
    def optimize_system_memory(self):
        """Optimize system memory usage."""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clean up temporary files
            self._cleanup_temp_files()
            
            # Clean up memory mappings
            self._cleanup_memory_mappings()
            
            # Platform-specific optimizations
            self._platform_specific_optimization()
            
            logger.info(f"Memory optimization: collected {collected} objects")
        except Exception as e:
            logger.error(f"System memory optimization failed: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files[:]:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    self.temp_files.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")
    
    def _cleanup_memory_mappings(self):
        """Clean up memory mappings."""
        for key, mapping in list(self.memory_mappings.items()):
            try:
                if hasattr(mapping, 'close'):
                    mapping.close()
                del self.memory_mappings[key]
            except Exception as e:
                logger.warning(f"Failed to cleanup memory mapping {key}: {e}")
    
    def _platform_specific_optimization(self):
        """Platform-specific memory optimization."""
        try:
            if os.name == 'nt':  # Windows
                import ctypes
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
            else:  # Unix-like
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (-1, -1))
        except Exception as e:
            logger.debug(f"Platform-specific optimization skipped: {e}")

class AsyncTaskManager:
    """Advanced async task management with optimization."""
    
    def __init__(self, config: AdvancedMemoryConfig):
        self.config = config
        self.task_queue = asyncio.Queue(maxsize=config.task_queue_size)
        self.active_tasks: List[asyncio.Task] = []
        self.task_semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self.running = False
        
    async def start(self):
        """Start the async task manager."""
        self.running = True
        # Start worker tasks
        for i in range(self.config.max_concurrent_tasks):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.active_tasks.append(task)
        
        logger.info(f"Started {len(self.active_tasks)} async workers")
    
    async def stop(self):
        """Stop the async task manager."""
        self.running = False
        
        # Cancel all active tasks
        for task in self.active_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.active_tasks, return_exceptions=True)
        self.active_tasks.clear()
        
        logger.info("Async task manager stopped")
    
    async def _worker(self, worker_name: str):
        """Worker coroutine for processing tasks."""
        while self.running:
            try:
                # Get task from queue with timeout
                task_func, args, kwargs, future = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )
                
                async with self.task_semaphore:
                    try:
                        result = await task_func(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                    finally:
                        self.task_queue.task_done()
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a task for async execution."""
        future = asyncio.Future()
        await self.task_queue.put((func, args, kwargs, future))
        return await future

class SmartCache:
    """Intelligent caching system with compression and optimization."""
    
    def __init__(self, config: AdvancedMemoryConfig):
        self.config = config
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self._decompress_if_needed(self.cache[key])
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with smart compression."""
        with self.lock:
            # Check if we need to evict items
            if self._should_evict():
                self._evict_lru_items()
            
            # Compress if needed
            compressed_value = self._compress_if_needed(value)
            
            self.cache[key] = compressed_value
            self.access_times[key] = time.time()
    
    def _should_evict(self) -> bool:
        """Check if cache eviction is needed."""
        if len(self.cache) < 1000:  # Keep reasonable size limit
            return False
        
        # Check memory usage
        estimated_size = sum(self._estimate_size(v) for v in self.cache.values())
        return estimated_size > self.config.max_cache_memory_mb * 1024 * 1024
    
    def _evict_lru_items(self):
        """Evict least recently used items."""
        # Sort by access time and remove oldest 20%
        sorted_items = sorted(
            self.access_times.items(),
            key=lambda x: x[1]
        )
        
        num_to_evict = max(1, len(sorted_items) // 5)
        for key, _ in sorted_items[:num_to_evict]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
    
    def _compress_if_needed(self, value: Any) -> Any:
        """Compress value if it's large enough."""
        try:
            serialized = pickle.dumps(value)
            if len(serialized) > self.config.cache_compression_threshold:
                import zlib
                return zlib.compress(serialized)
            return serialized
        except Exception:
            return value
    
    def _decompress_if_needed(self, value: Any) -> Any:
        """Decompress value if it was compressed."""
        try:
            if isinstance(value, bytes):
                try:
                    import zlib
                    decompressed = zlib.decompress(value)
                    return pickle.loads(decompressed)
                except:
                    return pickle.loads(value)
            return value
        except Exception:
            return value
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if isinstance(obj, bytes):
                return len(obj)
            return len(pickle.dumps(obj))
        except:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'estimated_memory_mb': sum(
                self._estimate_size(v) for v in self.cache.values()
            ) / (1024 * 1024)
        }

class AdvancedMemoryOptimizer:
    """Main advanced memory optimization system."""
    
    def __init__(self, config: Optional[AdvancedMemoryConfig] = None):
        self.config = config or AdvancedMemoryConfig()
        
        # Initialize components
        self.gpu_manager = GPUMemoryManager(self.config)
        self.system_manager = SystemMemoryManager(self.config)
        self.task_manager = AsyncTaskManager(self.config)
        self.cache = SmartCache(self.config)
        
        # Monitoring
        self.monitoring_active = False
        self.performance_history = deque(maxlen=1000)
        
        # Background optimization
        self.optimization_thread = None
        self.optimization_active = False
    
    async def start(self):
        """Start the memory optimization system."""
        await self.task_manager.start()
        
        if self.config.enable_background_optimization:
            self.start_background_optimization()
        
        if self.config.enable_detailed_profiling:
            self.start_monitoring()
        
        logger.info("Advanced memory optimizer started")
    
    async def stop(self):
        """Stop the memory optimization system."""
        await self.task_manager.stop()
        self.stop_background_optimization()
        self.stop_monitoring()
        
        logger.info("Advanced memory optimizer stopped")
    
    def start_background_optimization(self):
        """Start background optimization thread."""
        if self.optimization_thread is None:
            self.optimization_active = True
            self.optimization_thread = threading.Thread(
                target=self._background_optimization_loop,
                daemon=True
            )
            self.optimization_thread.start()
    
    def stop_background_optimization(self):
        """Stop background optimization."""
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
            self.optimization_thread = None
    
    def _background_optimization_loop(self):
        """Background optimization loop."""
        while self.optimization_active:
            try:
                # Check if optimization is needed
                if (self.system_manager.should_optimize_memory() or 
                    self._should_optimize_gpu()):
                    
                    self.optimize_all()
                
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                logger.error(f"Background optimization error: {e}")
    
    def _should_optimize_gpu(self) -> bool:
        """Check if GPU optimization is needed."""
        if not self.gpu_manager.cuda_available:
            return False
        
        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            usage = allocated / total
            return usage > self.config.max_gpu_memory_usage
        except:
            return False
    
    def optimize_all(self):
        """Optimize all memory systems."""
        try:
            # System memory optimization
            self.system_manager.optimize_system_memory()
            
            # GPU memory optimization
            self.gpu_manager.optimize_gpu_memory()
            
            # Record optimization
            stats = self.get_comprehensive_stats()
            self.performance_history.append({
                'timestamp': time.time(),
                'optimization': True,
                'stats': stats
            })
            
            logger.info("Comprehensive memory optimization completed")
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_active = True
        threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        ).start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
    
    def _monitoring_loop(self):
        """Performance monitoring loop."""
        while self.monitoring_active:
            try:
                stats = self.get_comprehensive_stats()
                self.performance_history.append({
                    'timestamp': time.time(),
                    'optimization': False,
                    'stats': stats
                })
                
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'memory': self.system_manager.get_memory_stats(),
            'cache': self.cache.get_stats(),
            'task_queue_size': self.task_manager.task_queue.qsize(),
            'active_tasks': len(self.task_manager.active_tasks),
        }
        
        # Add GPU stats if available
        if self.gpu_manager.cuda_available:
            try:
                stats['gpu'] = {
                    'allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                    'reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                    'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024**2),
                }
            except:
                pass
        
        return stats
    
    @contextmanager
    def optimized_context(self, task_name: str = "unknown"):
        """Context manager for optimized execution."""
        start_time = time.time()
        
        try:
            with self.gpu_manager.gpu_memory_context():
                yield
        finally:
            duration = time.time() - start_time
            
            # Log performance
            logger.debug(f"Task '{task_name}' completed in {duration:.3f}s")
            
            # Check if optimization is needed after task
            if duration > 5.0:  # Long-running task
                self.optimize_all()
    
    async def process_with_optimization(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Process function with automatic optimization."""
        # Submit to async task manager
        return await self.task_manager.submit_task(func, *args, **kwargs)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report."""
        if not self.performance_history:
            return {'message': 'No performance data available'}
        
        recent_stats = list(self.performance_history)[-10:]
        
        # Calculate averages
        avg_memory = np.mean([
            s['stats']['memory']['system_usage_percent'] 
            for s in recent_stats
        ])
        
        avg_cache_hit_rate = np.mean([
            s['stats']['cache']['hit_rate'] 
            for s in recent_stats
        ])
        
        report = {
            'monitoring_period': f"{len(self.performance_history)} samples",
            'average_memory_usage': f"{avg_memory:.1f}%",
            'cache_hit_rate': f"{avg_cache_hit_rate:.2%}",
            'optimizations_performed': sum(
                1 for s in recent_stats if s.get('optimization', False)
            ),
            'recommendations': self._generate_recommendations(recent_stats)
        }
        
        return report
    
    def _generate_recommendations(self, stats: List[Dict]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not stats:
            return recommendations
        
        # Memory usage analysis
        avg_memory = np.mean([
            s['stats']['memory']['system_usage_percent'] 
            for s in stats
        ])
        
        if avg_memory > 80:
            recommendations.append(
                "High memory usage detected. Consider increasing system memory "
                "or reducing batch sizes."
            )
        
        # Cache analysis
        avg_hit_rate = np.mean([
            s['stats']['cache']['hit_rate'] 
            for s in stats
        ])
        
        if avg_hit_rate < 0.5:
            recommendations.append(
                "Low cache hit rate. Consider increasing cache size or "
                "optimizing cache key strategies."
            )
        
        # GPU analysis
        gpu_stats = [s['stats'].get('gpu') for s in stats if s['stats'].get('gpu')]
        if gpu_stats:
            avg_gpu_memory = np.mean([
                g['allocated_mb'] for g in gpu_stats
            ])
            
            if avg_gpu_memory > 4000:  # > 4GB
                recommendations.append(
                    "High GPU memory usage. Consider reducing batch sizes "
                    "or enabling gradient checkpointing."
                )
        
        return recommendations

# Global instance for easy access
_global_optimizer: Optional[AdvancedMemoryOptimizer] = None

def get_global_optimizer() -> AdvancedMemoryOptimizer:
    """Get or create global memory optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = AdvancedMemoryOptimizer()
    return _global_optimizer

async def initialize_global_optimizer(config: Optional[AdvancedMemoryConfig] = None):
    """Initialize global memory optimizer."""
    global _global_optimizer
    if _global_optimizer is not None:
        await _global_optimizer.stop()
    
    _global_optimizer = AdvancedMemoryOptimizer(config)
    await _global_optimizer.start()

async def shutdown_global_optimizer():
    """Shutdown global memory optimizer."""
    global _global_optimizer
    if _global_optimizer is not None:
        await _global_optimizer.stop()
        _global_optimizer = None
