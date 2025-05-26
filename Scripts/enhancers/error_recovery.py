"""
Enhanced Error Recovery System for RAG Pipeline
Provides comprehensive error handling, recovery mechanisms, and system resilience
"""

import asyncio
import logging
import threading
import time
import json
import traceback
from typing import Dict, Any, Optional, List, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import functools
import inspect
from pathlib import Path
import yaml

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SYSTEM_RESTART = "system_restart"
    IGNORE = "ignore"

class ComponentState(Enum):
    """Component health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"

@dataclass
class ErrorPattern:
    """Pattern for matching and handling specific errors"""
    name: str
    error_types: List[Type[Exception]]
    error_messages: List[str]  # Regex patterns
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    max_retries: int = 3
    backoff_factor: float = 2.0
    timeout_seconds: float = 30.0
    fallback_function: Optional[str] = None
    custom_handler: Optional[str] = None

@dataclass
class RecoveryConfig:
    """Configuration for error recovery system"""
    # Global settings
    enable_error_recovery: bool = True
    enable_circuit_breaker: bool = True
    enable_health_monitoring: bool = True
    enable_automatic_restart: bool = True
    
    # Retry settings
    default_max_retries: int = 3
    default_backoff_factor: float = 2.0
    default_timeout: float = 30.0
    exponential_backoff: bool = True
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: int = 60
    circuit_breaker_reset_timeout: int = 300
    
    # Health monitoring
    health_check_interval: int = 30
    component_timeout: float = 10.0
    max_consecutive_failures: int = 3
    
    # Logging and alerting
    error_log_file: str = "logs/error_recovery.log"
    alert_on_critical_errors: bool = True
    max_error_history: int = 1000
    
    # Recovery strategies
    auto_fallback_enabled: bool = True
    graceful_degradation_enabled: bool = True
    system_restart_threshold: int = 10  # Critical errors before restart
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_error_recovery": self.enable_error_recovery,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "enable_health_monitoring": self.enable_health_monitoring,
            "enable_automatic_restart": self.enable_automatic_restart,
            "default_max_retries": self.default_max_retries,
            "default_backoff_factor": self.default_backoff_factor,
            "default_timeout": self.default_timeout,
            "exponential_backoff": self.exponential_backoff,
            "circuit_breaker_failure_threshold": self.circuit_breaker_failure_threshold,
            "circuit_breaker_timeout": self.circuit_breaker_timeout,
            "circuit_breaker_reset_timeout": self.circuit_breaker_reset_timeout,
            "health_check_interval": self.health_check_interval,
            "component_timeout": self.component_timeout,
            "max_consecutive_failures": self.max_consecutive_failures,
            "error_log_file": self.error_log_file,
            "alert_on_critical_errors": self.alert_on_critical_errors,
            "max_error_history": self.max_error_history,
            "auto_fallback_enabled": self.auto_fallback_enabled,
            "graceful_degradation_enabled": self.graceful_degradation_enabled,
            "system_restart_threshold": self.system_restart_threshold
        }

@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    timestamp: datetime
    component: str
    function_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    recovery_success: bool
    retry_count: int
    duration_ms: float
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "function_name": self.function_name,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "recovery_strategy": self.recovery_strategy.value,
            "recovery_success": self.recovery_success,
            "retry_count": self.retry_count,
            "duration_ms": self.duration_ms,
            "stack_trace": self.stack_trace,
            "context": self.context
        }

@dataclass
class ComponentHealth:
    """Health status of a system component"""
    name: str
    state: ComponentState
    last_check: datetime
    consecutive_failures: int
    total_failures: int
    total_successes: int
    avg_response_time: float
    last_error: Optional[str] = None
    recovery_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "last_check": self.last_check.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "avg_response_time": self.avg_response_time,
            "last_error": self.last_error,
            "recovery_count": self.recovery_count
        }

class CircuitBreaker:
    """Circuit breaker for preventing cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, reset_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.reset_timeout = reset_timeout
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.logger.info("Circuit breaker half-open, attempting reset")
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                self.logger.info("Circuit breaker reset to closed")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self.last_failure_time and 
            time.time() - self.last_failure_time > self.reset_timeout
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time
        }

class HealthMonitor:
    """Monitors component health and triggers recovery"""
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Component health tracking
        self.components: Dict[str, ComponentHealth] = {}
        self.health_checks: Dict[str, Callable] = {}
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
    
    def register_component(self, name: str, health_check: Callable):
        """Register a component for health monitoring"""
        self.components[name] = ComponentHealth(
            name=name,
            state=ComponentState.HEALTHY,
            last_check=datetime.now(),
            consecutive_failures=0,
            total_failures=0,
            total_successes=0,
            avg_response_time=0.0
        )
        self.health_checks[name] = health_check
        self.logger.info(f"Registered component for monitoring: {name}")
    
    def start_monitoring(self):
        """Start health monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="HealthMonitoring"
        )
        self.monitoring_thread.start()
        self.logger.info("Health monitoring started")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                for component_name in list(self.components.keys()):
                    asyncio.run(self._check_component_health(component_name))
                
                self._shutdown_event.wait(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._shutdown_event.wait(60)
    
    async def _check_component_health(self, component_name: str):
        """Check health of a specific component"""
        try:
            component = self.components[component_name]
            health_check = self.health_checks[component_name]
            
            start_time = time.time()
            
            # Execute health check with timeout
            try:
                if asyncio.iscoroutinefunction(health_check):
                    result = await asyncio.wait_for(
                        health_check(), 
                        timeout=self.config.component_timeout
                    )
                else:
                    result = health_check()
                
                # Health check passed
                response_time = (time.time() - start_time) * 1000
                component.avg_response_time = (
                    (component.avg_response_time * component.total_successes + response_time) /
                    (component.total_successes + 1)
                )
                component.total_successes += 1
                component.consecutive_failures = 0
                component.last_check = datetime.now()
                
                # Update state based on response time and other factors
                if component.state != ComponentState.HEALTHY:
                    component.state = ComponentState.HEALTHY
                    self.logger.info(f"Component {component_name} recovered to healthy state")
                
            except asyncio.TimeoutError:
                self._handle_component_failure(component, "Health check timeout")
            except Exception as e:
                self._handle_component_failure(component, str(e))
            
        except Exception as e:
            self.logger.error(f"Error checking component {component_name}: {e}")
    
    def _handle_component_failure(self, component: ComponentHealth, error: str):
        """Handle component failure"""
        component.total_failures += 1
        component.consecutive_failures += 1
        component.last_error = error
        component.last_check = datetime.now()
        
        # Update component state based on failure count
        if component.consecutive_failures >= self.config.max_consecutive_failures:
            component.state = ComponentState.FAILED
            self.logger.error(f"Component {component.name} marked as failed: {error}")
        elif component.consecutive_failures >= 2:
            component.state = ComponentState.FAILING
            self.logger.warning(f"Component {component.name} is failing: {error}")
        else:
            component.state = ComponentState.DEGRADED
            self.logger.warning(f"Component {component.name} is degraded: {error}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        healthy_count = sum(1 for c in self.components.values() if c.state == ComponentState.HEALTHY)
        total_count = len(self.components)
        
        overall_state = "healthy"
        if healthy_count == 0:
            overall_state = "critical"
        elif healthy_count < total_count * 0.5:
            overall_state = "degraded"
        elif healthy_count < total_count:
            overall_state = "warning"
        
        return {
            "overall_state": overall_state,
            "healthy_components": healthy_count,
            "total_components": total_count,
            "components": {name: comp.to_dict() for name, comp in self.components.items()}
        }
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self._shutdown_event.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

class ErrorRecoverySystem:
    """Main error recovery system"""
    
    def __init__(self, config: Optional[RecoveryConfig] = None):
        self.config = config or RecoveryConfig()
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history: List[ErrorRecord] = []
        self.error_patterns: List[ErrorPattern] = []
        self.component_circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Health monitoring
        self.health_monitor = HealthMonitor(self.config)
        
        # Recovery functions registry
        self.fallback_functions: Dict[str, Callable] = {}
        self.custom_handlers: Dict[str, Callable] = {}
        
        # Statistics
        self.total_errors = 0
        self.recovered_errors = 0
        self.critical_errors = 0
        
        # Initialize default error patterns
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """Initialize default error patterns"""
        # GPU/CUDA errors
        self.error_patterns.append(ErrorPattern(
            name="gpu_memory_error",
            error_types=[RuntimeError],
            error_messages=["CUDA out of memory", "GPU memory", "device-side assert"],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            max_retries=2,
            fallback_function="cpu_fallback"
        ))
        
        # Network/API errors
        self.error_patterns.append(ErrorPattern(
            name="network_error",
            error_types=[ConnectionError, TimeoutError],
            error_messages=["Connection failed", "Timeout", "Network unreachable"],
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            max_retries=3,
            backoff_factor=2.0
        ))
        
        # Memory errors
        self.error_patterns.append(ErrorPattern(
            name="memory_error",
            error_types=[MemoryError],
            error_messages=["Out of memory", "Cannot allocate memory"],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            fallback_function="reduce_batch_size"
        ))
        
        # File system errors
        self.error_patterns.append(ErrorPattern(
            name="filesystem_error",
            error_types=[IOError, FileNotFoundError, PermissionError],
            error_messages=["No such file", "Permission denied", "Disk full"],
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            max_retries=2
        ))
        
        # Model loading errors
        self.error_patterns.append(ErrorPattern(
            name="model_error",
            error_types=[RuntimeError, ValueError],
            error_messages=["model", "checkpoint", "transformer"],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            fallback_function="load_backup_model"
        ))
    
    def register_fallback_function(self, name: str, func: Callable):
        """Register a fallback function"""
        self.fallback_functions[name] = func
        self.logger.info(f"Registered fallback function: {name}")
    
    def register_custom_handler(self, name: str, handler: Callable):
        """Register a custom error handler"""
        self.custom_handlers[name] = handler
        self.logger.info(f"Registered custom handler: {name}")
    
    def with_recovery(
        self,
        component_name: str = "unknown",
        max_retries: Optional[int] = None,
        fallback_func: Optional[Callable] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ):
        """Decorator for adding error recovery to functions"""
        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_recovery(
                    func, args, kwargs, component_name, max_retries, fallback_func, severity
                )
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._execute_with_recovery(
                    func, args, kwargs, component_name, max_retries, fallback_func, severity
                ))
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    async def _execute_with_recovery(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        component_name: str,
        max_retries: Optional[int],
        fallback_func: Optional[Callable],
        severity: ErrorSeverity
    ):
        """Execute function with error recovery"""
        start_time = time.time()
        retry_count = 0
        last_error = None
        
        # Get or create circuit breaker for component
        if component_name not in self.component_circuit_breakers and self.config.enable_circuit_breaker:
            self.component_circuit_breakers[component_name] = CircuitBreaker(
                self.config.circuit_breaker_failure_threshold,
                self.config.circuit_breaker_timeout,
                self.config.circuit_breaker_reset_timeout
            )
        
        circuit_breaker = self.component_circuit_breakers.get(component_name)
        
        # Determine max retries
        effective_max_retries = max_retries or self.config.default_max_retries
        
        while retry_count <= effective_max_retries:
            try:
                # Execute through circuit breaker if available
                if circuit_breaker and self.config.enable_circuit_breaker:
                    result = await circuit_breaker.call(func, *args, **kwargs)
                else:
                    result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Success - record if there were previous failures
                if retry_count > 0:
                    duration_ms = (time.time() - start_time) * 1000
                    self._record_recovery_success(
                        component_name, func.__name__, retry_count, duration_ms
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                self.total_errors += 1
                
                # Find matching error pattern
                pattern = self._find_error_pattern(e)
                
                # Record error
                duration_ms = (time.time() - start_time) * 1000
                error_record = self._record_error(
                    component_name, func.__name__, e, pattern, retry_count, duration_ms
                )
                
                # Check if we should retry
                if retry_count < effective_max_retries and pattern and pattern.recovery_strategy == RecoveryStrategy.RETRY:
                    retry_count += 1
                    
                    # Calculate backoff delay
                    if self.config.exponential_backoff:
                        delay = (pattern.backoff_factor ** retry_count) * 0.1
                    else:
                        delay = pattern.backoff_factor * 0.1
                    
                    self.logger.warning(f"Retrying {func.__name__} (attempt {retry_count}/{effective_max_retries}) after {delay:.2f}s")
                    await asyncio.sleep(delay)
                    continue
                
                # Try recovery strategy
                recovery_result = await self._attempt_recovery(
                    e, pattern, func, args, kwargs, fallback_func
                )
                
                if recovery_result is not None:
                    self.recovered_errors += 1
                    error_record.recovery_success = True
                    return recovery_result
                
                # Recovery failed, re-raise the last error
                break
        
        # All retries exhausted, record final error
        duration_ms = (time.time() - start_time) * 1000
        self._record_error(component_name, func.__name__, last_error, None, retry_count, duration_ms)
        
        # Check for critical error threshold
        if severity == ErrorSeverity.CRITICAL:
            self.critical_errors += 1
            if self.critical_errors >= self.config.system_restart_threshold:
                await self._trigger_system_restart()
        
        raise last_error
    
    def _find_error_pattern(self, error: Exception) -> Optional[ErrorPattern]:
        """Find matching error pattern for an exception"""
        import re
        
        error_type = type(error)
        error_message = str(error)
        
        for pattern in self.error_patterns:
            # Check error type
            if error_type in pattern.error_types:
                # Check error message patterns
                for msg_pattern in pattern.error_messages:
                    if re.search(msg_pattern, error_message, re.IGNORECASE):
                        return pattern
        
        return None
    
    async def _attempt_recovery(
        self,
        error: Exception,
        pattern: Optional[ErrorPattern],
        func: Callable,
        args: tuple,
        kwargs: dict,
        fallback_func: Optional[Callable]
    ) -> Any:
        """Attempt error recovery based on strategy"""
        try:
            if pattern:
                strategy = pattern.recovery_strategy
                
                if strategy == RecoveryStrategy.FALLBACK:
                    # Try pattern fallback first, then provided fallback
                    if pattern.fallback_function and pattern.fallback_function in self.fallback_functions:
                        fallback = self.fallback_functions[pattern.fallback_function]
                        return await fallback(*args, **kwargs) if asyncio.iscoroutinefunction(fallback) else fallback(*args, **kwargs)
                    elif fallback_func:
                        return await fallback_func(*args, **kwargs) if asyncio.iscoroutinefunction(fallback_func) else fallback_func(*args, **kwargs)
                
                elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    return await self._graceful_degradation_fallback(*args, **kwargs)
                
                elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    # Circuit breaker is handled in the main execution loop
                    pass
                
                # Try custom handler if available
                if pattern.custom_handler and pattern.custom_handler in self.custom_handlers:
                    handler = self.custom_handlers[pattern.custom_handler]
                    return await handler(error, func, *args, **kwargs)
            
            # Default fallback if provided
            if fallback_func:
                return await fallback_func(*args, **kwargs) if asyncio.iscoroutinefunction(fallback_func) else fallback_func(*args, **kwargs)
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
        
        return None
    
    async def _graceful_degradation_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Default graceful degradation response"""
        return {
            "status": "degraded",
            "message": "Service operating in degraded mode due to error",
            "timestamp": datetime.now().isoformat(),
            "partial_result": True
        }
    
    def _record_error(
        self,
        component: str,
        function_name: str,
        error: Exception,
        pattern: Optional[ErrorPattern],
        retry_count: int,
        duration_ms: float
    ) -> ErrorRecord:
        """Record an error occurrence"""
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            component=component,
            function_name=function_name,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=pattern.severity if pattern else ErrorSeverity.MEDIUM,
            recovery_strategy=pattern.recovery_strategy if pattern else RecoveryStrategy.IGNORE,
            recovery_success=False,
            retry_count=retry_count,
            duration_ms=duration_ms,
            stack_trace=traceback.format_exc()
        )
        
        self.error_history.append(error_record)
        
        # Trim error history
        if len(self.error_history) > self.config.max_error_history:
            self.error_history = self.error_history[-self.config.max_error_history // 2:]
        
        # Log error
        if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"Error in {component}.{function_name}: {error}")
        else:
            self.logger.warning(f"Error in {component}.{function_name}: {error}")
        
        return error_record
    
    def _record_recovery_success(self, component: str, function_name: str, retry_count: int, duration_ms: float):
        """Record successful recovery"""
        self.logger.info(f"Recovery successful for {component}.{function_name} after {retry_count} retries ({duration_ms:.1f}ms)")
    
    async def _trigger_system_restart(self):
        """Trigger system restart due to critical errors"""
        self.logger.critical("Critical error threshold reached, triggering system restart")
        
        if self.config.enable_automatic_restart:
            # This would integrate with your system restart mechanism
            # For now, we'll just log the action
            self.logger.critical("Automatic restart would be triggered here")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error and recovery statistics"""
        recovery_rate = 0.0
        if self.total_errors > 0:
            recovery_rate = self.recovered_errors / self.total_errors
        
        # Error breakdown by severity
        severity_counts = {severity.value: 0 for severity in ErrorSeverity}
        for error in self.error_history:
            severity_counts[error.severity.value] += 1
        
        # Recent errors (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_errors = [e for e in self.error_history if e.timestamp > recent_cutoff]
        
        return {
            "total_errors": self.total_errors,
            "recovered_errors": self.recovered_errors,
            "critical_errors": self.critical_errors,
            "recovery_rate": recovery_rate,
            "error_history_size": len(self.error_history),
            "recent_errors_count": len(recent_errors),
            "severity_breakdown": severity_counts,
            "circuit_breakers": {
                name: cb.get_state() 
                for name, cb in self.component_circuit_breakers.items()
            },
            "patterns_registered": len(self.error_patterns),
            "fallback_functions": list(self.fallback_functions.keys()),
            "custom_handlers": list(self.custom_handlers.keys())
        }
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent error records"""
        recent = sorted(self.error_history, key=lambda x: x.timestamp, reverse=True)[:count]
        return [error.to_dict() for error in recent]
    
    def start_health_monitoring(self):
        """Start health monitoring"""
        if self.config.enable_health_monitoring:
            self.health_monitor.start_monitoring()
    
    def register_health_check(self, component_name: str, health_check: Callable):
        """Register a component for health monitoring"""
        self.health_monitor.register_component(component_name, health_check)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        return self.health_monitor.get_system_health()
    
    async def shutdown(self):
        """Shutdown the error recovery system"""
        self.logger.info("Shutting down error recovery system...")
        self.health_monitor.stop_monitoring()

# Global error recovery system
_error_recovery_system: Optional[ErrorRecoverySystem] = None

def get_error_recovery_system() -> ErrorRecoverySystem:
    """Get global error recovery system instance"""
    global _error_recovery_system
    
    if _error_recovery_system is None:
        _error_recovery_system = ErrorRecoverySystem()
        _error_recovery_system.start_health_monitoring()
    
    return _error_recovery_system

# Convenience decorators
def with_recovery(component_name: str = "unknown", **kwargs):
    """Decorator for adding error recovery to functions"""
    recovery_system = get_error_recovery_system()
    return recovery_system.with_recovery(component_name=component_name, **kwargs)

def health_check(component_name: str):
    """Decorator for registering health check functions"""
    def decorator(func: Callable):
        recovery_system = get_error_recovery_system()
        recovery_system.register_health_check(component_name, func)
        return func
    return decorator

if __name__ == "__main__":
    # Example usage
    async def main():
        recovery_system = ErrorRecoverySystem()
        
        # Register a fallback function
        def cpu_fallback(*args, **kwargs):
            return {"status": "success", "mode": "cpu_fallback"}
        
        recovery_system.register_fallback_function("cpu_fallback", cpu_fallback)
        
        # Example function with recovery
        @recovery_system.with_recovery(component_name="test_component", max_retries=2)
        async def test_function(should_fail: bool = False):
            if should_fail:
                raise RuntimeError("CUDA out of memory")
            return {"status": "success", "data": "test_data"}
        
        # Test normal execution
        result1 = await test_function(False)
        print(f"Normal execution: {result1}")
        
        # Test error recovery
        try:
            result2 = await test_function(True)
            print(f"Error recovery: {result2}")
        except Exception as e:
            print(f"Recovery failed: {e}")
        
        # Get statistics
        stats = recovery_system.get_error_statistics()
        print(f"Error statistics: {json.dumps(stats, indent=2)}")
        
        await recovery_system.shutdown()
    
    asyncio.run(main())
