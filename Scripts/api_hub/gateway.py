"""
API Gateway for external service integration.

This module provides a centralized gateway for integrating with external APIs
and services, with OAuth2 support, rate limiting, and usage analytics.
"""
import os
import json
import logging
import time
import requests
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import yaml
import jwt
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from fastapi.security.api_key import APIKeyHeader, APIKey
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

class APIGateway:
    """Central gateway for external API integration."""
    
    def __init__(
        self, 
        config_path: str = "config.yaml",
        cache_dir: str = "cache/api_responses",
        token_dir: str = "data/api_tokens"
    ):
        """Initialize the API gateway.
        
        Args:
            config_path: Path to configuration file
            cache_dir: Directory for caching API responses
            token_dir: Directory for storing OAuth tokens
        """
        self.config_path = config_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token_dir = Path(token_dir)
        self.token_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self._load_config()
        
        # Initialize services
        self.services = {}
        self._initialize_services()
        
        # Analytics tracker
        self.analytics = APIAnalytics()
    
    def _load_config(self):
        """Load gateway configuration."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Extract API hub config if it exists
            self.config = config.get("api_hub", {})
            
            # If no API hub config exists, create default
            if not self.config:
                self.config = {
                    "services": {},
                    "rate_limits": {
                        "default": "100/day"
                    },
                    "cache_ttl": 3600  # Default 1 hour cache
                }
                
                # Update config file with default API hub section
                config["api_hub"] = self.config
                with open(self.config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
        
        except Exception as e:
            logger.error(f"Failed to load API gateway config: {e}")
            # Create default config
            self.config = {
                "services": {},
                "rate_limits": {
                    "default": "100/day"
                },
                "cache_ttl": 3600  # Default 1 hour cache
            }
    
    def _initialize_services(self):
        """Initialize configured services."""
        for service_name, service_config in self.config.get("services", {}).items():
            service_type = service_config.get("type", "rest")
            
            if service_type == "rest":
                self.services[service_name] = RESTService(
                    name=service_name,
                    config=service_config,
                    gateway=self
                )
            elif service_type == "graphql":
                self.services[service_name] = GraphQLService(
                    name=service_name,
                    config=service_config,
                    gateway=self
                )
            elif service_type == "oauth":
                self.services[service_name] = OAuthService(
                    name=service_name,
                    config=service_config,
                    gateway=self,
                    token_dir=self.token_dir
                )
            else:
                logger.warning(f"Unknown service type '{service_type}' for service '{service_name}'")
    
    def register_service(
        self, 
        name: str,
        service_type: str,
        base_url: str,
        auth_type: str = None,
        auth_params: Dict[str, Any] = None,
        rate_limit: str = None,
        endpoints: Dict[str, Dict[str, Any]] = None
    ):
        """Register a new service.
        
        Args:
            name: Service name
            service_type: Service type (rest, graphql, oauth)
            base_url: Base URL for the service
            auth_type: Authentication type (api_key, oauth2, basic)
            auth_params: Authentication parameters
            rate_limit: Rate limit for the service
            endpoints: Endpoint configurations
        """
        # Create service config
        service_config = {
            "type": service_type,
            "base_url": base_url,
            "auth": {
                "type": auth_type,
                "params": auth_params or {}
            } if auth_type else None,
            "rate_limit": rate_limit,
            "endpoints": endpoints or {}
        }
        
        # Add to configuration
        self.config["services"][name] = service_config
        
        # Initialize service
        if service_type == "rest":
            self.services[name] = RESTService(
                name=name,
                config=service_config,
                gateway=self
            )
        elif service_type == "graphql":
            self.services[name] = GraphQLService(
                name=name,
                config=service_config,
                gateway=self
            )
        elif service_type == "oauth":
            self.services[name] = OAuthService(
                name=name,
                config=service_config,
                gateway=self,
                token_dir=self.token_dir
            )
        
        # Save updated configuration
        self._save_config()
        
        logger.info(f"Registered service '{name}' of type '{service_type}'")
        return self.services[name]
    
    def _save_config(self):
        """Save current configuration."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Update API hub section
            config["api_hub"] = self.config
            
            with open(self.config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        
        except Exception as e:
            logger.error(f"Failed to save API gateway config: {e}")
    
    def get_service(self, name: str):
        """Get a service by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance or None if not found
        """
        return self.services.get(name)
    
    def list_services(self):
        """List all registered services.
        
        Returns:
            List of service names
        """
        return list(self.services.keys())
    
    def call_endpoint(
        self,
        service_name: str,
        endpoint_name: str,
        params: Dict[str, Any] = None,
        data: Any = None,
        headers: Dict[str, str] = None,
        use_cache: bool = True,
        cache_ttl: int = None
    ):
        """Call a service endpoint.
        
        Args:
            service_name: Name of the service
            endpoint_name: Name of the endpoint
            params: Query parameters
            data: Request data/payload
            headers: Additional headers
            use_cache: Whether to use caching
            cache_ttl: Cache TTL in seconds
            
        Returns:
            Response from the service
        """
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service '{service_name}' not found")
        
        # Track API usage
        self.analytics.track_request(service_name, endpoint_name)
        
        # Compute cache key if using cache
        cache_key = None
        if use_cache:
            cache_key = self._compute_cache_key(
                service_name=service_name,
                endpoint_name=endpoint_name,
                params=params,
                data=data
            )
            
            # Check cache
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for {service_name}.{endpoint_name}")
                return cached_response
        
        # Call service endpoint
        response = service.call_endpoint(
            endpoint_name=endpoint_name,
            params=params,
            data=data,
            headers=headers
        )
        
        # Cache response if requested
        if use_cache and cache_key:
            self._cache_response(
                cache_key=cache_key,
                response=response,
                ttl=cache_ttl or self.config.get("cache_ttl", 3600)
            )
        
        return response
    
    def _compute_cache_key(
        self,
        service_name: str,
        endpoint_name: str,
        params: Dict[str, Any] = None,
        data: Any = None
    ):
        """Compute a cache key for a request.
        
        Args:
            service_name: Service name
            endpoint_name: Endpoint name
            params: Query parameters
            data: Request data
            
        Returns:
            Cache key
        """
        # Create a string representation of the request
        key_parts = [service_name, endpoint_name]
        
        if params:
            # Sort parameters for consistent keys
            param_str = json.dumps(params, sort_keys=True)
            key_parts.append(param_str)
        
        if data:
            if isinstance(data, dict):
                data_str = json.dumps(data, sort_keys=True)
            else:
                data_str = str(data)
            key_parts.append(data_str)
        
        # Join parts and compute hash
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str):
        """Get a response from cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached response or None
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            
            # Check if cache has expired
            expires = datetime.fromisoformat(cache_data["expires"])
            if expires < datetime.now():
                # Cache expired
                cache_file.unlink()
                return None
            
            return cache_data["response"]
        
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None
    
    def _cache_response(self, cache_key: str, response: Any, ttl: int):
        """Cache a response.
        
        Args:
            cache_key: Cache key
            response: Response to cache
            ttl: Cache TTL in seconds
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            expires = datetime.now() + timedelta(seconds=ttl)
            cache_data = {
                "response": response,
                "expires": expires.isoformat()
            }
            
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
        
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def clear_cache(self, service_name: str = None, endpoint_name: str = None):
        """Clear API response cache.
        
        Args:
            service_name: Optional service name to clear cache for
            endpoint_name: Optional endpoint name to clear cache for
        """
        # If no filters, clear all cache
        if not service_name and not endpoint_name:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared all API cache")
            return
        
        # Clear cache for specific service/endpoint
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                
                cached_service = cache_data.get("service")
                cached_endpoint = cache_data.get("endpoint")
                
                if service_name and cached_service != service_name:
                    continue
                
                if endpoint_name and cached_endpoint != endpoint_name:
                    continue
                
                cache_file.unlink()
            
            except Exception as e:
                logger.warning(f"Failed to process cache file {cache_file}: {e}")
        
        if endpoint_name:
            logger.info(f"Cleared cache for {service_name}.{endpoint_name}")
        else:
            logger.info(f"Cleared cache for {service_name}")


class APIService:
    """Base class for API services."""
    
    def __init__(
        self, 
        name: str,
        config: Dict[str, Any],
        gateway: APIGateway
    ):
        """Initialize the service.
        
        Args:
            name: Service name
            config: Service configuration
            gateway: API gateway instance
        """
        self.name = name
        self.config = config
        self.gateway = gateway
        self.base_url = config.get("base_url", "")
    
    def call_endpoint(
        self,
        endpoint_name: str,
        params: Dict[str, Any] = None,
        data: Any = None,
        headers: Dict[str, str] = None
    ):
        """Call an endpoint.
        
        Args:
            endpoint_name: Endpoint name
            params: Query parameters
            data: Request data
            headers: Additional headers
            
        Returns:
            Response from the endpoint
        """
        raise NotImplementedError("Subclasses must implement call_endpoint")


class RESTService(APIService):
    """REST API service."""
    
    def call_endpoint(
        self,
        endpoint_name: str,
        params: Dict[str, Any] = None,
        data: Any = None,
        headers: Dict[str, str] = None
    ):
        """Call a REST endpoint.
        
        Args:
            endpoint_name: Endpoint name
            params: Query parameters
            data: Request data
            headers: Additional headers
            
        Returns:
            Response from the endpoint
        """
        # Get endpoint configuration
        endpoints = self.config.get("endpoints", {})
        if endpoint_name not in endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not defined for service '{self.name}'")
        
        endpoint_config = endpoints[endpoint_name]
        
        # Build URL
        path = endpoint_config.get("path", "")
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        
        # Get HTTP method
        method = endpoint_config.get("method", "GET").upper()
        
        # Prepare headers
        request_headers = {}
        if headers:
            request_headers.update(headers)
        
        # Add authentication if configured
        auth_config = self.config.get("auth")
        if auth_config:
            auth_type = auth_config.get("type")
            auth_params = auth_config.get("params", {})
            
            if auth_type == "api_key":
                # Add API key to headers or params
                key_name = auth_params.get("key_name", "api_key")
                key_value = auth_params.get("key_value", "")
                key_in = auth_params.get("in", "header")
                
                if key_in == "header":
                    request_headers[key_name] = key_value
                elif key_in == "query":
                    if params is None:
                        params = {}
                    params[key_name] = key_value
            
            elif auth_type == "basic":
                # Add basic authentication
                import base64
                username = auth_params.get("username", "")
                password = auth_params.get("password", "")
                auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
                request_headers["Authorization"] = f"Basic {auth_string}"
            
            elif auth_type == "bearer":
                # Add bearer token
                token = auth_params.get("token", "")
                request_headers["Authorization"] = f"Bearer {token}"
        
        # Make HTTP request
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                json=data if isinstance(data, dict) else None,
                data=data if not isinstance(data, dict) else None,
                headers=request_headers,
                timeout=endpoint_config.get("timeout", 30)
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            else:
                return response.text
        
        except requests.RequestException as e:
            logger.error(f"Error calling {self.name}.{endpoint_name}: {e}")
            raise


class GraphQLService(APIService):
    """GraphQL API service."""
    
    def call_endpoint(
        self,
        endpoint_name: str,
        params: Dict[str, Any] = None,
        data: Any = None,
        headers: Dict[str, str] = None
    ):
        """Call a GraphQL endpoint.
        
        Args:
            endpoint_name: Endpoint name (query/mutation name)
            params: Variables for the query
            data: Not used for GraphQL
            headers: Additional headers
            
        Returns:
            Response from the endpoint
        """
        # Get endpoint configuration
        endpoints = self.config.get("endpoints", {})
        if endpoint_name not in endpoints:
            raise ValueError(f"Query/mutation '{endpoint_name}' not defined for service '{self.name}'")
        
        endpoint_config = endpoints[endpoint_name]
        
        # Get query/mutation
        query = endpoint_config.get("query", "")
        if not query:
            raise ValueError(f"No query defined for endpoint '{endpoint_name}'")
        
        # Prepare headers
        request_headers = {
            "Content-Type": "application/json"
        }
        if headers:
            request_headers.update(headers)
        
        # Add authentication if configured
        auth_config = self.config.get("auth")
        if auth_config:
            auth_type = auth_config.get("type")
            auth_params = auth_config.get("params", {})
            
            if auth_type == "api_key":
                # Add API key to headers
                key_name = auth_params.get("key_name", "api_key")
                key_value = auth_params.get("key_value", "")
                request_headers[key_name] = key_value
            
            elif auth_type == "bearer":
                # Add bearer token
                token = auth_params.get("token", "")
                request_headers["Authorization"] = f"Bearer {token}"
        
        # Prepare request payload
        request_data = {
            "query": query,
            "variables": params or {}
        }
        
        # Make HTTP request
        try:
            response = requests.post(
                url=self.base_url,
                json=request_data,
                headers=request_headers,
                timeout=endpoint_config.get("timeout", 30)
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Check for GraphQL errors
            if "errors" in result:
                logger.error(f"GraphQL errors: {result['errors']}")
                raise ValueError(f"GraphQL errors: {result['errors']}")
            
            return result.get("data", {})
        
        except requests.RequestException as e:
            logger.error(f"Error calling {self.name}.{endpoint_name}: {e}")
            raise


class OAuthService(APIService):
    """OAuth2 API service."""
    
    def __init__(
        self, 
        name: str,
        config: Dict[str, Any],
        gateway: APIGateway,
        token_dir: Path
    ):
        """Initialize the OAuth service.
        
        Args:
            name: Service name
            config: Service configuration
            gateway: API gateway instance
            token_dir: Directory for storing OAuth tokens
        """
        super().__init__(name, config, gateway)
        self.token_dir = token_dir
        self.token_file = token_dir / f"{name}_token.json"
        
        # Load existing token if available
        self.token = self._load_token()
    
    def _load_token(self):
        """Load OAuth token from file.
        
        Returns:
            Token data or None
        """
        if not self.token_file.exists():
            return None
        
        try:
            with open(self.token_file, "r") as f:
                token_data = json.load(f)
            
            # Check if token has expired
            expires_at = token_data.get("expires_at")
            if expires_at and expires_at < time.time():
                # Try to refresh token
                refresh_token = token_data.get("refresh_token")
                if refresh_token:
                    return self._refresh_token(refresh_token)
                else:
                    return None
            
            return token_data
        
        except Exception as e:
            logger.warning(f"Failed to load token: {e}")
            return None
    
    def _save_token(self, token_data):
        """Save OAuth token to file.
        
        Args:
            token_data: Token data to save
        """
        try:
            with open(self.token_file, "w") as f:
                json.dump(token_data, f)
        
        except Exception as e:
            logger.warning(f"Failed to save token: {e}")
    
    def _refresh_token(self, refresh_token):
        """Refresh OAuth token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New token data or None
        """
        auth_config = self.config.get("auth", {})
        if auth_config.get("type") != "oauth2":
            return None
        
        auth_params = auth_config.get("params", {})
        token_url = auth_params.get("token_url", "")
        client_id = auth_params.get("client_id", "")
        client_secret = auth_params.get("client_secret", "")
        
        if not token_url or not client_id:
            return None
        
        try:
            response = requests.post(
                token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            )
            
            response.raise_for_status()
            new_token = response.json()
            
            # Add expiration time if not provided
            if "expires_in" in new_token and "expires_at" not in new_token:
                new_token["expires_at"] = time.time() + new_token["expires_in"]
            
            # Save new token
            self._save_token(new_token)
            
            return new_token
        
        except requests.RequestException as e:
            logger.error(f"Error refreshing token: {e}")
            return None
    
    def call_endpoint(
        self,
        endpoint_name: str,
        params: Dict[str, Any] = None,
        data: Any = None,
        headers: Dict[str, str] = None
    ):
        """Call an OAuth protected endpoint.
        
        Args:
            endpoint_name: Endpoint name
            params: Query parameters
            data: Request data
            headers: Additional headers
            
        Returns:
            Response from the endpoint
        """
        # Get endpoint configuration
        endpoints = self.config.get("endpoints", {})
        if endpoint_name not in endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not defined for service '{self.name}'")
        
        endpoint_config = endpoints[endpoint_name]
        
        # Build URL
        path = endpoint_config.get("path", "")
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        
        # Get HTTP method
        method = endpoint_config.get("method", "GET").upper()
        
        # Prepare headers
        request_headers = {}
        if headers:
            request_headers.update(headers)
        
        # Add OAuth token
        if not self.token:
            raise ValueError(f"No valid OAuth token available for service '{self.name}'")
        
        access_token = self.token.get("access_token")
        if not access_token:
            raise ValueError(f"Invalid OAuth token for service '{self.name}'")
        
        request_headers["Authorization"] = f"Bearer {access_token}"
        
        # Make HTTP request
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                json=data if isinstance(data, dict) else None,
                data=data if not isinstance(data, dict) else None,
                headers=request_headers,
                timeout=endpoint_config.get("timeout", 30)
            )
            
            # Check for unauthorized error (token might have expired)
            if response.status_code == 401:
                # Try to refresh token
                refresh_token = self.token.get("refresh_token")
                if refresh_token:
                    new_token = self._refresh_token(refresh_token)
                    if new_token:
                        # Update authorization header with new token
                        access_token = new_token.get("access_token")
                        request_headers["Authorization"] = f"Bearer {access_token}"
                        
                        # Retry request
                        response = requests.request(
                            method=method,
                            url=url,
                            params=params,
                            json=data if isinstance(data, dict) else None,
                            data=data if not isinstance(data, dict) else None,
                            headers=request_headers,
                            timeout=endpoint_config.get("timeout", 30)
                        )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            else:
                return response.text
        
        except requests.RequestException as e:
            logger.error(f"Error calling {self.name}.{endpoint_name}: {e}")
            raise


class APIAnalytics:
    """Analytics tracker for API usage."""
    
    def __init__(self, storage_dir: str = "data/api_analytics"):
        """Initialize the analytics tracker.
        
        Args:
            storage_dir: Directory for storing analytics data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Daily usage file
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_file = self.storage_dir / f"usage_{today}.json"
        
        # Initialize daily usage data
        self.today_usage = self._load_daily_usage()
    
    def _load_daily_usage(self):
        """Load daily usage data.
        
        Returns:
            Usage data or empty dict
        """
        if not self.daily_file.exists():
            return {"date": datetime.now().strftime("%Y-%m-%d"), "services": {}}
        
        try:
            with open(self.daily_file, "r") as f:
                return json.load(f)
        
        except Exception as e:
            logger.warning(f"Failed to load daily usage data: {e}")
            return {"date": datetime.now().strftime("%Y-%m-%d"), "services": {}}
    
    def _save_daily_usage(self):
        """Save daily usage data."""
        try:
            with open(self.daily_file, "w") as f:
                json.dump(self.today_usage, f, indent=2)
        
        except Exception as e:
            logger.warning(f"Failed to save daily usage data: {e}")
    
    def track_request(self, service_name: str, endpoint_name: str):
        """Track an API request.
        
        Args:
            service_name: Service name
            endpoint_name: Endpoint name
        """
        # Check if a new day has started
        today = datetime.now().strftime("%Y-%m-%d")
        if self.today_usage["date"] != today:
            # New day, create new file
            self.today_usage = {"date": today, "services": {}}
            self.daily_file = self.storage_dir / f"usage_{today}.json"
        
        # Update usage data
        services = self.today_usage["services"]
        if service_name not in services:
            services[service_name] = {"total": 0, "endpoints": {}}
        
        services[service_name]["total"] += 1
        
        endpoints = services[service_name]["endpoints"]
        if endpoint_name not in endpoints:
            endpoints[endpoint_name] = 0
        
        endpoints[endpoint_name] += 1
        
        # Save updated data
        self._save_daily_usage()
    
    def get_usage(self, days: int = 7):
        """Get API usage statistics.
        
        Args:
            days: Number of days to include
            
        Returns:
            Usage statistics
        """
        # Get date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Find usage files in range
        usage_files = []
        for i in range(days):
            date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            file_path = self.storage_dir / f"usage_{date}.json"
            if file_path.exists():
                usage_files.append(file_path)
        
        # Load and merge usage data
        usage_data = {
            "period": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d")
            },
            "services": {},
            "daily": []
        }
        
        for file_path in usage_files:
            try:
                with open(file_path, "r") as f:
                    daily_data = json.load(f)
                
                # Add to daily data
                usage_data["daily"].append(daily_data)
                
                # Merge service data
                for service_name, service_data in daily_data["services"].items():
                    if service_name not in usage_data["services"]:
                        usage_data["services"][service_name] = {"total": 0, "endpoints": {}}
                    
                    usage_data["services"][service_name]["total"] += service_data["total"]
                    
                    for endpoint_name, count in service_data["endpoints"].items():
                        endpoints = usage_data["services"][service_name]["endpoints"]
                        if endpoint_name not in endpoints:
                            endpoints[endpoint_name] = 0
                        
                        endpoints[endpoint_name] += count
            
            except Exception as e:
                logger.warning(f"Failed to load usage data from {file_path}: {e}")
        
        return usage_data


# FastAPI application for API Gateway
app = FastAPI(title="API Gateway", description="External API Integration Hub")

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# OAuth2 security scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    scopes={
        "admin": "Full access to API Gateway",
        "read": "Read-only access to API Gateway",
        "service": "Access to specific services"
    }
)

# User model for authentication
class User(BaseModel):
    username: str
    disabled: bool = False
    scopes: List[str] = []

# Token model
class Token(BaseModel):
    access_token: str
    token_type: str

# OAuth2 DTO
class OAuth2RequestForm(BaseModel):
    grant_type: str = Field(...)
    username: str = Field(...)
    password: str = Field(...)
    scope: str = Field(default="")
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

# Gateway request model
class GatewayRequest(BaseModel):
    service: str = Field(..., description="Service name")
    endpoint: str = Field(..., description="Endpoint name")
    params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
    data: Optional[Any] = Field(None, description="Request data")
    headers: Optional[Dict[str, str]] = Field(None, description="Additional headers")
    use_cache: bool = Field(True, description="Whether to use caching")

# Middleware for API request logging
class APILoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Record request time
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate request duration
        duration = time.time() - start_time
        
        # Log request details
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code}, "
            f"Duration: {duration:.4f}s"
        )
        
        return response

# Add middleware
app.add_middleware(APILoggingMiddleware)

# Dependency for API key validation
async def get_api_key(api_key: str = Security(api_key_header)):
    # In a real app, this would validate against a database
    if api_key != "test_api_key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Dependency for OAuth token validation
async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
):
    # Define required scopes
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        # Decode JWT token (in a real app, this would use a proper secret key)
        payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # Get token scopes
        token_scopes = payload.get("scopes", [])
    except jwt.PyJWTError:
        raise credentials_exception
    
    # Create user
    user = User(username=username, scopes=token_scopes)
    
    # Validate scopes
    for scope in security_scopes.scopes:
        if scope not in token_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required: {scope}",
                headers={"WWW-Authenticate": authenticate_value},
            )
    
    return user

# Endpoint for OAuth token
@app.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2RequestForm = Depends()):
    # In a real app, this would validate against a database
    if form_data.username != "admin" or form_data.password != "password":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Parse requested scopes
    scopes = form_data.scope.split() if form_data.scope else ["read"]
    
    # Create JWT token
    token_data = {
        "sub": form_data.username,
        "scopes": scopes,
        "exp": datetime.utcnow() + timedelta(minutes=30)
    }
    
    # Sign token
    access_token = jwt.encode(token_data, "secret_key", algorithm="HS256")
    
    return {"access_token": access_token, "token_type": "bearer"}

# Gateway endpoint with API key auth
@app.post("/api/gateway/key")
@limiter.limit("100/minute")
async def gateway_api_key(
    request: GatewayRequest,
    api_key: str = Depends(get_api_key)
):
    # Initialize gateway
    gateway = APIGateway()
    
    try:
        response = gateway.call_endpoint(
            service_name=request.service,
            endpoint_name=request.endpoint,
            params=request.params,
            data=request.data,
            headers=request.headers,
            use_cache=request.use_cache
        )
        
        return {"success": True, "data": response}
    
    except Exception as e:
        logger.error(f"Gateway error: {e}")
        return {"success": False, "error": str(e)}

# Gateway endpoint with OAuth auth
@app.post("/api/gateway/oauth")
@limiter.limit("100/minute")
async def gateway_oauth(
    request: GatewayRequest,
    user: User = Security(get_current_user, scopes=["read"])
):
    # Initialize gateway
    gateway = APIGateway()
    
    try:
        response = gateway.call_endpoint(
            service_name=request.service,
            endpoint_name=request.endpoint,
            params=request.params,
            data=request.data,
            headers=request.headers,
            use_cache=request.use_cache
        )
        
        return {"success": True, "data": response}
    
    except Exception as e:
        logger.error(f"Gateway error: {e}")
        return {"success": False, "error": str(e)}

# Analytics endpoint (admin only)
@app.get("/api/analytics")
async def get_analytics(
    days: int = 7,
    user: User = Security(get_current_user, scopes=["admin"])
):
    analytics = APIAnalytics()
    usage_data = analytics.get_usage(days=days)
    
    return usage_data

# Service management endpoints
@app.get("/api/services")
async def list_services(
    user: User = Security(get_current_user, scopes=["read"])
):
    gateway = APIGateway()
    services = gateway.list_services()
    
    return {"services": services}

@app.get("/api/services/{service_name}")
async def get_service(
    service_name: str,
    user: User = Security(get_current_user, scopes=["read"])
):
    gateway = APIGateway()
    service = gateway.get_service(service_name)
    
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{service_name}' not found"
        )
    
    return {"service": service.config}

# Cache management endpoint
@app.post("/api/cache/clear")
async def clear_cache(
    service_name: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    user: User = Security(get_current_user, scopes=["admin"])
):
    gateway = APIGateway()
    gateway.clear_cache(service_name=service_name, endpoint_name=endpoint_name)
    
    return {"success": True, "message": "Cache cleared"}
