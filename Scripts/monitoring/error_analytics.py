import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import traceback
import sys
from dataclasses import dataclass, asdict
import asyncio
import aiohttp
from prometheus_client import Counter, Histogram
import hashlib
from collections import defaultdict
import threading
from queue import Queue

@dataclass
class ErrorEvent:
    error_type: str
    message: str
    stack_trace: str
    component: str
    severity: str
    timestamp: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.metadata:
            self.metadata = {}

class ErrorReporter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_logging()
        self.setup_metrics()
        
        # Initialize error queue for async processing
        self.error_queue = Queue()
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.worker.start()
        
        # Error aggregation
        self.error_counts = defaultdict(int)
        self.last_errors = {}
        
    def _setup_logging(self):
        self.logger = logging.getLogger("error_reporter")
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Error log handler
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(error_handler)
        self.logger.setLevel(logging.ERROR)
        
    def setup_metrics(self):
        """Initialize Prometheus metrics."""
        self.error_counter = Counter(
            'rag_errors_total',
            'Total number of errors',
            ['type', 'component', 'severity']
        )
        self.error_processing_time = Histogram(
            'rag_error_processing_seconds',
            'Time spent processing errors'
        )
        
    def _get_error_fingerprint(self, error: ErrorEvent) -> str:
        """Generate a unique fingerprint for error deduplication."""
        content = f"{error.error_type}:{error.component}:{error.message}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def _should_notify(self, error: ErrorEvent, fingerprint: str) -> bool:
        """Determine if an error should trigger a notification."""
        now = datetime.utcnow()
        
        # Check if we've seen this error recently
        if fingerprint in self.last_errors:
            last_time = self.last_errors[fingerprint]
            if (now - last_time).seconds < self.config.get('notification_cooldown', 3600):
                return False
                
        # Check error count threshold
        threshold = self.config.get('error_threshold', 10)
        if self.error_counts[fingerprint] >= threshold:
            return True
            
        # Always notify for critical errors
        if error.severity == 'critical':
            return True
            
        return False
        
    async def _notify_error(self, error: ErrorEvent):
        """Send error notifications through configured channels."""
        if 'slack_webhook' in self.config:
            await self._notify_slack(error)
            
        if 'email_config' in self.config:
            await self._notify_email(error)
            
    async def _notify_slack(self, error: ErrorEvent):
        """Send error notification to Slack."""
        webhook_url = self.config['slack_webhook']
        
        message = {
            "text": f"*Error Alert*\nType: {error.error_type}\nComponent: {error.component}\nSeverity: {error.severity}\nMessage: {error.message}"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=message) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to send Slack notification: {await response.text()}")
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {str(e)}")
            
    async def _notify_email(self, error: ErrorEvent):
        """Send error notification via email."""
        # Implement email notification logic here
        pass
        
    def _process_queue(self):
        """Process errors from the queue."""
        while True:
            try:
                error = self.error_queue.get()
                if error is None:
                    break
                    
                # Process the error
                fingerprint = self._get_error_fingerprint(error)
                self.error_counts[fingerprint] += 1
                self.last_errors[fingerprint] = datetime.utcnow()
                
                # Update metrics
                self.error_counter.labels(
                    type=error.error_type,
                    component=error.component,
                    severity=error.severity
                ).inc()
                
                # Log the error
                self.logger.error(
                    f"Error in {error.component}: {error.message}\n{error.stack_trace}",
                    extra=error.metadata
                )
                
                # Check if we should notify
                if self._should_notify(error, fingerprint):
                    asyncio.run(self._notify_error(error))
                
                self.error_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing error event: {str(e)}")
                
    def report_error(self, error: Exception, component: str, severity: str = "error", metadata: Dict[str, Any] = None):
        """Report an error for processing."""
        error_event = ErrorEvent(
            error_type=error.__class__.__name__,
            message=str(error),
            stack_trace=''.join(traceback.format_tb(error.__traceback__)),
            component=component,
            severity=severity,
            metadata=metadata
        )
        
        self.error_queue.put(error_event)
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "errors_by_type": dict(self.error_counts),
            "last_errors": {k: v.isoformat() for k, v in self.last_errors.items()}
        }
        
    def __del__(self):
        """Cleanup when destroyed."""
        self.error_queue.put(None)
        self.worker.join()

class Analytics:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Initialize analytics metrics."""
        # User activity metrics
        self.user_requests = Counter(
            'rag_user_requests_total',
            'Total user requests',
            ['endpoint', 'user_id']
        )
        self.user_sessions = Counter(
            'rag_user_sessions_total',
            'Total user sessions',
            ['user_id']
        )
        
        # Search metrics
        self.search_latency = Histogram(
            'rag_search_latency_seconds',
            'Search request latency'
        )
        self.search_results = Histogram(
            'rag_search_results_count',
            'Number of search results returned'
        )
        
        # Document metrics
        self.doc_processing_time = Histogram(
            'rag_document_processing_seconds',
            'Document processing time',
            ['file_type']
        )
        self.doc_size = Histogram(
            'rag_document_size_bytes',
            'Document size in bytes',
            ['file_type']
        )
        
    def track_request(self, endpoint: str, user_id: str):
        """Track an API request."""
        self.user_requests.labels(endpoint=endpoint, user_id=user_id).inc()
        
    def track_session(self, user_id: str):
        """Track a user session."""
        self.user_sessions.labels(user_id=user_id).inc()
        
    def track_search(self, latency: float, results_count: int):
        """Track search performance."""
        self.search_latency.observe(latency)
        self.search_results.observe(results_count)
        
    def track_document(self, processing_time: float, size: int, file_type: str):
        """Track document processing."""
        self.doc_processing_time.labels(file_type=file_type).observe(processing_time)
        self.doc_size.labels(file_type=file_type).observe(size)
        
    async def generate_analytics_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analytics report."""
        # Implement analytics aggregation and reporting logic
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                # Add collected metrics here
            }
        }
