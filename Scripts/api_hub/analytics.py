"""
API hub usage analytics module.

This module provides functionality for tracking and analyzing
usage of the API Integration Hub.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import time
import uuid
import sqlite3
from pathlib import Path
import threading
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ApiUsageAnalytics:
    """API usage analytics tracking and reporting."""
    
    def __init__(self, db_path: str = "data/api_hub/analytics.db"):
        """Initialize the analytics tracker.
        
        Args:
            db_path: Path to SQLite database for storing analytics
        """
        self.db_path = db_path
        self._ensure_db_exists()
        self._lock = threading.Lock()
    
    def _ensure_db_exists(self):
        """Ensure the analytics database exists and has the correct schema."""
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_requests (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                service_id TEXT,
                endpoint TEXT,
                method TEXT,
                client_id TEXT,
                status_code INTEGER,
                response_time_ms REAL,
                token_type TEXT,
                error TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS rate_limit_events (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                client_id TEXT,
                limit_name TEXT,
                limit_value INTEGER,
                window_seconds INTEGER,
                current_count INTEGER
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS oauth_events (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                event_type TEXT,
                provider TEXT,
                client_id TEXT,
                success INTEGER,
                error TEXT
            )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON api_requests(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_requests_service ON api_requests(service_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rate_limits_timestamp ON rate_limit_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_oauth_timestamp ON oauth_events(timestamp)')
            
            conn.commit()
    
    def track_request(
        self,
        service_id: str,
        endpoint: str,
        method: str,
        client_id: str,
        status_code: int,
        response_time_ms: float,
        token_type: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Track an API request.
        
        Args:
            service_id: Service ID
            endpoint: API endpoint
            method: HTTP method
            client_id: Client ID
            status_code: HTTP status code
            response_time_ms: Response time in milliseconds
            token_type: Authentication token type
            error: Error message if request failed
        """
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        '''
                        INSERT INTO api_requests (
                            id, timestamp, service_id, endpoint, method, 
                            client_id, status_code, response_time_ms, token_type, error
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            str(uuid.uuid4()),
                            datetime.now().isoformat(),
                            service_id,
                            endpoint,
                            method,
                            client_id,
                            status_code,
                            response_time_ms,
                            token_type,
                            error
                        )
                    )
                    
                    conn.commit()
            except Exception as e:
                logger.error(f"Error tracking API request: {e}")
    
    def track_rate_limit_event(
        self,
        client_id: str,
        limit_name: str,
        limit_value: int,
        window_seconds: int,
        current_count: int
    ):
        """Track a rate limit event.
        
        Args:
            client_id: Client ID
            limit_name: Rate limit name
            limit_value: Rate limit value
            window_seconds: Rate limit window in seconds
            current_count: Current count
        """
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        '''
                        INSERT INTO rate_limit_events (
                            id, timestamp, client_id, limit_name, 
                            limit_value, window_seconds, current_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            str(uuid.uuid4()),
                            datetime.now().isoformat(),
                            client_id,
                            limit_name,
                            limit_value,
                            window_seconds,
                            current_count
                        )
                    )
                    
                    conn.commit()
            except Exception as e:
                logger.error(f"Error tracking rate limit event: {e}")
    
    def track_oauth_event(
        self,
        event_type: str,
        provider: str,
        client_id: str,
        success: bool,
        error: Optional[str] = None
    ):
        """Track an OAuth event.
        
        Args:
            event_type: Event type (authorize, token, refresh)
            provider: OAuth provider
            client_id: Client ID
            success: Whether the event was successful
            error: Error message if event failed
        """
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        '''
                        INSERT INTO oauth_events (
                            id, timestamp, event_type, provider, 
                            client_id, success, error
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            str(uuid.uuid4()),
                            datetime.now().isoformat(),
                            event_type,
                            provider,
                            client_id,
                            1 if success else 0,
                            error
                        )
                    )
                    
                    conn.commit()
            except Exception as e:
                logger.error(f"Error tracking OAuth event: {e}")
    
    def get_request_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        service_id: Optional[str] = None,
        client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get API request statistics.
        
        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            service_id: Service ID for filtering
            client_id: Client ID for filtering
            
        Returns:
            Dictionary with request statistics
        """
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # Build query
                    query = "SELECT * FROM api_requests WHERE 1=1"
                    params = []
                    
                    if start_time:
                        query += " AND timestamp >= ?"
                        params.append(start_time.isoformat())
                    
                    if end_time:
                        query += " AND timestamp <= ?"
                        params.append(end_time.isoformat())
                    
                    if service_id:
                        query += " AND service_id = ?"
                        params.append(service_id)
                    
                    if client_id:
                        query += " AND client_id = ?"
                        params.append(client_id)
                    
                    # Execute query
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Convert to dictionaries
                    requests = [dict(row) for row in rows]
                    
                    # Calculate statistics
                    total_requests = len(requests)
                    successful_requests = sum(1 for r in requests if 200 <= r["status_code"] < 300)
                    failed_requests = total_requests - successful_requests
                    avg_response_time = sum(r["response_time_ms"] for r in requests) / total_requests if total_requests else 0
                    
                    # Count by service
                    service_counts = {}
                    for r in requests:
                        service_id = r["service_id"]
                        service_counts[service_id] = service_counts.get(service_id, 0) + 1
                    
                    # Count by endpoint
                    endpoint_counts = {}
                    for r in requests:
                        endpoint = r["endpoint"]
                        endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
                    
                    # Count by method
                    method_counts = {}
                    for r in requests:
                        method = r["method"]
                        method_counts[method] = method_counts.get(method, 0) + 1
                    
                    # Count by status code
                    status_code_counts = {}
                    for r in requests:
                        status_code = r["status_code"]
                        status_code_counts[status_code] = status_code_counts.get(status_code, 0) + 1
                    
                    return {
                        "total_requests": total_requests,
                        "successful_requests": successful_requests,
                        "failed_requests": failed_requests,
                        "success_rate": successful_requests / total_requests if total_requests else 0,
                        "avg_response_time_ms": avg_response_time,
                        "service_counts": service_counts,
                        "endpoint_counts": endpoint_counts,
                        "method_counts": method_counts,
                        "status_code_counts": status_code_counts
                    }
            except Exception as e:
                logger.error(f"Error getting request stats: {e}")
                return {
                    "error": str(e),
                    "total_requests": 0
                }
    
    def get_rate_limit_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get rate limit statistics.
        
        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            client_id: Client ID for filtering
            
        Returns:
            Dictionary with rate limit statistics
        """
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # Build query
                    query = "SELECT * FROM rate_limit_events WHERE 1=1"
                    params = []
                    
                    if start_time:
                        query += " AND timestamp >= ?"
                        params.append(start_time.isoformat())
                    
                    if end_time:
                        query += " AND timestamp <= ?"
                        params.append(end_time.isoformat())
                    
                    if client_id:
                        query += " AND client_id = ?"
                        params.append(client_id)
                    
                    # Execute query
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Convert to dictionaries
                    events = [dict(row) for row in rows]
                    
                    # Calculate statistics
                    total_events = len(events)
                    
                    # Group by limit name
                    limit_counts = {}
                    for e in events:
                        limit_name = e["limit_name"]
                        limit_counts[limit_name] = limit_counts.get(limit_name, 0) + 1
                    
                    # Group by client
                    client_counts = {}
                    for e in events:
                        client_id = e["client_id"]
                        client_counts[client_id] = client_counts.get(client_id, 0) + 1
                    
                    # Calculate rate limit utilization
                    utilization = {}
                    for e in events:
                        limit_name = e["limit_name"]
                        if limit_name not in utilization:
                            utilization[limit_name] = {
                                "count": 0,
                                "total": 0,
                                "max_utilization": 0
                            }
                        
                        utilization[limit_name]["count"] += 1
                        utilization[limit_name]["total"] += e["current_count"] / e["limit_value"]
                        utilization[limit_name]["max_utilization"] = max(
                            utilization[limit_name]["max_utilization"],
                            e["current_count"] / e["limit_value"]
                        )
                    
                    # Calculate average utilization
                    for limit_name, stats in utilization.items():
                        stats["avg_utilization"] = stats["total"] / stats["count"] if stats["count"] else 0
                    
                    return {
                        "total_events": total_events,
                        "limit_counts": limit_counts,
                        "client_counts": client_counts,
                        "utilization": utilization
                    }
            except Exception as e:
                logger.error(f"Error getting rate limit stats: {e}")
                return {
                    "error": str(e),
                    "total_events": 0
                }
    
    def get_daily_request_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        service_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get daily API request statistics.
        
        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            service_id: Service ID for filtering
            
        Returns:
            Dictionary with daily request counts and performance metrics
        """
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # Build query
                    query = """
                    SELECT 
                        date(timestamp) as request_date, 
                        COUNT(*) as request_count,
                        AVG(response_time_ms) as avg_response_time,
                        SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) as success_count,
                        SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count
                    FROM api_requests 
                    WHERE 1=1
                    """
                    params = []
                    
                    if start_time:
                        query += " AND timestamp >= ?"
                        params.append(start_time.isoformat())
                    
                    if end_time:
                        query += " AND timestamp <= ?"
                        params.append(end_time.isoformat())
                    
                    if service_id:
                        query += " AND service_id = ?"
                        params.append(service_id)
                    
                    query += " GROUP BY date(timestamp) ORDER BY date(timestamp)"
                    
                    # Execute query
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Convert to list of dictionaries
                    daily_stats = []
                    for row in rows:
                        daily_stats.append({
                            "date": row["request_date"],
                            "request_count": row["request_count"],
                            "avg_response_time": row["avg_response_time"],
                            "success_count": row["success_count"],
                            "error_count": row["error_count"],
                            "success_rate": row["success_count"] / row["request_count"] if row["request_count"] > 0 else 0
                        })
                    
                    # If using pandas, we could create charts here
                    
                    return {
                        "daily_stats": daily_stats,
                        "total_days": len(daily_stats)
                    }
            except Exception as e:
                logger.error(f"Error getting daily request stats: {e}")
                return {
                    "error": str(e),
                    "daily_stats": []
                }
    
    def get_oauth_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        provider: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get OAuth statistics.
        
        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            provider: OAuth provider for filtering
            event_type: Event type for filtering
            
        Returns:
            Dictionary with OAuth statistics
        """
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # Build query
                    query = "SELECT * FROM oauth_events WHERE 1=1"
                    params = []
                    
                    if start_time:
                        query += " AND timestamp >= ?"
                        params.append(start_time.isoformat())
                    
                    if end_time:
                        query += " AND timestamp <= ?"
                        params.append(end_time.isoformat())
                    
                    if provider:
                        query += " AND provider = ?"
                        params.append(provider)
                    
                    if event_type:
                        query += " AND event_type = ?"
                        params.append(event_type)
                    
                    # Execute query
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Convert to dictionaries
                    events = [dict(row) for row in rows]
                    
                    # Calculate statistics
                    total_events = len(events)
                    successful_events = sum(1 for e in events if e["success"])
                    failed_events = total_events - successful_events
                    
                    # Group by provider
                    provider_counts = {}
                    for e in events:
                        provider = e["provider"]
                        provider_counts[provider] = provider_counts.get(provider, 0) + 1
                    
                    # Group by event type
                    event_type_counts = {}
                    for e in events:
                        event_type = e["event_type"]
                        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
                    
                    # Group by client
                    client_counts = {}
                    for e in events:
                        client_id = e["client_id"]
                        client_counts[client_id] = client_counts.get(client_id, 0) + 1
                    
                    # Success rates by provider
                    provider_success_rates = {}
                    for provider in provider_counts.keys():
                        provider_events = [e for e in events if e["provider"] == provider]
                        successful_provider_events = sum(1 for e in provider_events if e["success"])
                        provider_success_rates[provider] = successful_provider_events / len(provider_events) if provider_events else 0
                    
                    # Success rates by event type
                    event_type_success_rates = {}
                    for event_type in event_type_counts.keys():
                        event_type_events = [e for e in events if e["event_type"] == event_type]
                        successful_event_type_events = sum(1 for e in event_type_events if e["success"])
                        event_type_success_rates[event_type] = successful_event_type_events / len(event_type_events) if event_type_events else 0
                    
                    return {
                        "total_events": total_events,
                        "successful_events": successful_events,
                        "failed_events": failed_events,
                        "success_rate": successful_events / total_events if total_events else 0,
                        "provider_counts": provider_counts,
                        "event_type_counts": event_type_counts,
                        "client_counts": client_counts,
                        "provider_success_rates": provider_success_rates,
                        "event_type_success_rates": event_type_success_rates
                    }
            except Exception as e:
                logger.error(f"Error getting OAuth stats: {e}")
                return {
                    "error": str(e),
                    "total_events": 0
                }
    
    def generate_analytics_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive analytics report.
        
        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            output_dir: Directory to save report visualizations
            
        Returns:
            Dictionary with report data
        """
        # Default time range to last 7 days if not specified
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        # Get statistics
        request_stats = self.get_request_stats(start_time, end_time)
        rate_limit_stats = self.get_rate_limit_stats(start_time, end_time)
        oauth_stats = self.get_oauth_stats(start_time, end_time)
        
        # Generate visualizations if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._generate_request_visualizations(request_stats, start_time, end_time, output_dir)
            self._generate_rate_limit_visualizations(rate_limit_stats, start_time, end_time, output_dir)
            self._generate_oauth_visualizations(oauth_stats, start_time, end_time, output_dir)
        
        # Return combined report
        return {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "request_stats": request_stats,
            "rate_limit_stats": rate_limit_stats,
            "oauth_stats": oauth_stats,
            "summary": {
                "total_requests": request_stats.get("total_requests", 0),
                "success_rate": request_stats.get("success_rate", 0),
                "avg_response_time_ms": request_stats.get("avg_response_time_ms", 0),
                "total_rate_limit_events": rate_limit_stats.get("total_events", 0),
                "total_oauth_events": oauth_stats.get("total_events", 0),
                "oauth_success_rate": oauth_stats.get("success_rate", 0)
            }
        }
    
    def _generate_request_visualizations(
        self,
        stats: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
        output_dir: str
    ):
        """Generate visualizations for request statistics.
        
        Args:
            stats: Request statistics
            start_time: Start time
            end_time: End time
            output_dir: Output directory
        """
        # This would create visualizations using matplotlib or similar
        # For now, just log that this would be implemented
        logger.info("Request visualization generation would be implemented here")
    
    def _generate_rate_limit_visualizations(
        self,
        stats: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
        output_dir: str
    ):
        """Generate visualizations for rate limit statistics.
        
        Args:
            stats: Rate limit statistics
            start_time: Start time
            end_time: End time
            output_dir: Output directory
        """
        # This would create visualizations using matplotlib or similar
        # For now, just log that this would be implemented
        logger.info("Rate limit visualization generation would be implemented here")
    
    def _generate_oauth_visualizations(
        self,
        stats: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
        output_dir: str
    ):
        """Generate visualizations for OAuth statistics.
        
        Args:
            stats: OAuth statistics
            start_time: Start time
            end_time: End time
            output_dir: Output directory
        """
        # This would create visualizations using matplotlib or similar
        # For now, just log that this would be implemented
        logger.info("OAuth visualization generation would be implemented here")
