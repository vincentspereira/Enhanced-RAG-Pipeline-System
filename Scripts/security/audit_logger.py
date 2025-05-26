import logging
from datetime import datetime
import json
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass, asdict
import uuid
from elasticsearch import Elasticsearch
import threading
from queue import Queue
import time

@dataclass
class AuditEvent:
    event_type: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    timestamp: str = None
    metadata: Dict[str, Any] = None
    status: str = "success"
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.metadata:
            self.metadata = {}

class AuditLogger:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("audit")
        self._setup_logging()
        
        # Initialize Elasticsearch if configured
        self.es = None
        if config.get("use_elasticsearch"):
            self.es = Elasticsearch(config["elasticsearch_url"])
            
        # Setup async logging queue
        self.queue = Queue()
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.worker.start()
        
    def _setup_logging(self):
        """Configure logging handlers."""
        log_dir = self.config.get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler for audit logs
        audit_handler = logging.FileHandler(
            os.path.join(log_dir, "audit.log")
        )
        audit_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        )
        self.logger.addHandler(audit_handler)
        self.logger.setLevel(logging.INFO)
        
    def _process_queue(self):
        """Process events from the queue."""
        while True:
            try:
                event = self.queue.get()
                if event is None:
                    break
                    
                self._write_event(event)
                self.queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing audit event: {str(e)}")
                
    def _write_event(self, event: AuditEvent):
        """Write event to configured outputs."""
        event_dict = asdict(event)
        
        # Log to file
        self.logger.info(json.dumps(event_dict))
        
        # Store in Elasticsearch if configured
        if self.es:
            try:
                self.es.index(
                    index=f"audit-{datetime.now():%Y-%m}",
                    document=event_dict
                )
            except Exception as e:
                self.logger.error(f"Failed to store audit event in Elasticsearch: {str(e)}")
                
    def log_event(self, event: AuditEvent):
        """Queue an audit event for processing."""
        self.queue.put(event)
        
    def search_events(self, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     filters: Optional[Dict[str, Any]] = None) -> list:
        """Search audit events."""
        if not self.es:
            raise Exception("Elasticsearch not configured for audit log searching")
            
        query = {"bool": {"must": []}}
        
        # Add date range
        if start_date or end_date:
            date_range = {}
            if start_date:
                date_range["gte"] = start_date.isoformat()
            if end_date:
                date_range["lte"] = end_date.isoformat()
            query["bool"]["must"].append({"range": {"timestamp": date_range}})
            
        # Add filters
        if filters:
            for key, value in filters.items():
                query["bool"]["must"].append({"match": {key: value}})
                
        try:
            result = self.es.search(
                index="audit-*",
                query=query,
                size=100  # Configurable
            )
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except Exception as e:
            self.logger.error(f"Failed to search audit events: {str(e)}")
            return []
            
    def __del__(self):
        """Cleanup when destroyed."""
        self.queue.put(None)  # Signal worker to stop
        self.worker.join()
        if self.es:
            self.es.close()
