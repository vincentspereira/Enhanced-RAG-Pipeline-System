"""
Command-line interface for the API Gateway.

This module provides a CLI for managing the API Gateway and its services.
"""
import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from tabulate import tabulate
import requests
from datetime import datetime, timedelta

from .gateway import APIGateway, APIAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="API Gateway CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Service commands
    service_parser = subparsers.add_parser("service", help="Manage API services")
    service_subparsers = service_parser.add_subparsers(dest="service_command", help="Service command to run")
    
    # List services
    list_parser = service_subparsers.add_parser("list", help="List registered services")
    
    # Show service info
    info_parser = service_subparsers.add_parser("info", help="Show service information")
    info_parser.add_argument("--name", required=True, help="Service name")
    
    # Register REST service
    register_rest_parser = service_subparsers.add_parser("register-rest", help="Register a REST service")
    register_rest_parser.add_argument("--name", required=True, help="Service name")
    register_rest_parser.add_argument("--url", required=True, help="Base URL for the service")
    register_rest_parser.add_argument("--auth-type", choices=["api_key", "basic", "bearer"], help="Authentication type")
    register_rest_parser.add_argument("--auth-params", help="Authentication parameters as JSON")
    register_rest_parser.add_argument("--rate-limit", help="Rate limit for the service")
    
    # Register GraphQL service
    register_graphql_parser = service_subparsers.add_parser("register-graphql", help="Register a GraphQL service")
    register_graphql_parser.add_argument("--name", required=True, help="Service name")
    register_graphql_parser.add_argument("--url", required=True, help="GraphQL endpoint URL")
    register_graphql_parser.add_argument("--auth-type", choices=["api_key", "bearer"], help="Authentication type")
    register_graphql_parser.add_argument("--auth-params", help="Authentication parameters as JSON")
    register_graphql_parser.add_argument("--rate-limit", help="Rate limit for the service")
    
    # Register OAuth service
    register_oauth_parser = service_subparsers.add_parser("register-oauth", help="Register an OAuth service")
    register_oauth_parser.add_argument("--name", required=True, help="Service name")
    register_oauth_parser.add_argument("--url", required=True, help="Base URL for the service")
    register_oauth_parser.add_argument("--token-url", required=True, help="OAuth token URL")
    register_oauth_parser.add_argument("--client-id", required=True, help="OAuth client ID")
    register_oauth_parser.add_argument("--client-secret", help="OAuth client secret")
    register_oauth_parser.add_argument("--rate-limit", help="Rate limit for the service")
    
    # Add endpoint
    add_endpoint_parser = service_subparsers.add_parser("add-endpoint", help="Add an endpoint to a service")
    add_endpoint_parser.add_argument("--service", required=True, help="Service name")
    add_endpoint_parser.add_argument("--name", required=True, help="Endpoint name")
    add_endpoint_parser.add_argument("--path", required=True, help="Endpoint path")
    add_endpoint_parser.add_argument("--method", default="GET", help="HTTP method")
    add_endpoint_parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    
    # Add GraphQL query
    add_query_parser = service_subparsers.add_parser("add-query", help="Add a GraphQL query to a service")
    add_query_parser.add_argument("--service", required=True, help="Service name")
    add_query_parser.add_argument("--name", required=True, help="Query name")
    add_query_parser.add_argument("--query", required=True, help="GraphQL query")
    add_query_parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    
    # Call endpoint
    call_parser = service_subparsers.add_parser("call", help="Call a service endpoint")
    call_parser.add_argument("--service", required=True, help="Service name")
    call_parser.add_argument("--endpoint", required=True, help="Endpoint name")
    call_parser.add_argument("--params", help="Query parameters as JSON")
    call_parser.add_argument("--data", help="Request data as JSON")
    call_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    # Cache commands
    cache_parser = subparsers.add_parser("cache", help="Manage API response cache")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", help="Cache command to run")
    
    # Clear cache
    clear_parser = cache_subparsers.add_parser("clear", help="Clear API response cache")
    clear_parser.add_argument("--service", help="Service name to clear cache for")
    clear_parser.add_argument("--endpoint", help="Endpoint name to clear cache for")
    
    # Analytics commands
    analytics_parser = subparsers.add_parser("analytics", help="API usage analytics")
    analytics_subparsers = analytics_parser.add_subparsers(dest="analytics_command", help="Analytics command to run")
    
    # Show analytics
    show_parser = analytics_subparsers.add_parser("show", help="Show API usage analytics")
    show_parser.add_argument("--days", type=int, default=7, help="Number of days to include")
    show_parser.add_argument("--service", help="Filter by service name")
    show_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Initialize gateway
    gateway = APIGateway()
    
    # Handle service commands
    if args.command == "service":
        if args.service_command == "list":
            services = gateway.list_services()
            
            if not services:
                print("No services registered")
                return
            
            print(f"Registered services ({len(services)}):")
            for service_name in services:
                service = gateway.get_service(service_name)
                service_type = service.config.get("type", "unknown")
                base_url = service.config.get("base_url", "")
                print(f"  {service_name} ({service_type}): {base_url}")
        
        elif args.service_command == "info":
            service = gateway.get_service(args.name)
            
            if not service:
                print(f"Service '{args.name}' not found")
                return
            
            print(f"Service: {args.name}")
            print(f"  Type: {service.config.get('type', 'unknown')}")
            print(f"  Base URL: {service.config.get('base_url', '')}")
            
            # Auth info
            auth_config = service.config.get("auth")
            if auth_config:
                auth_type = auth_config.get("type")
                print(f"  Authentication: {auth_type}")
            else:
                print("  Authentication: None")
            
            # Endpoints
            endpoints = service.config.get("endpoints", {})
            if endpoints:
                print(f"  Endpoints ({len(endpoints)}):")
                for endpoint_name, endpoint_config in endpoints.items():
                    if service.config.get("type") == "rest":
                        method = endpoint_config.get("method", "GET")
                        path = endpoint_config.get("path", "")
                        print(f"    {endpoint_name}: {method} {path}")
                    elif service.config.get("type") == "graphql":
                        print(f"    {endpoint_name}: GraphQL query")
            else:
                print("  No endpoints defined")
        
        elif args.service_command == "register-rest":
            # Parse auth params
            auth_params = None
            if args.auth_params:
                try:
                    auth_params = json.loads(args.auth_params)
                except json.JSONDecodeError:
                    print("Error: auth-params must be valid JSON")
                    return
            
            # Register service
            service = gateway.register_service(
                name=args.name,
                service_type="rest",
                base_url=args.url,
                auth_type=args.auth_type,
                auth_params=auth_params,
                rate_limit=args.rate_limit
            )
            
            print(f"Registered REST service '{args.name}'")
            print(f"  Base URL: {args.url}")
            if args.auth_type:
                print(f"  Authentication: {args.auth_type}")
            if args.rate_limit:
                print(f"  Rate limit: {args.rate_limit}")
        
        elif args.service_command == "register-graphql":
            # Parse auth params
            auth_params = None
            if args.auth_params:
                try:
                    auth_params = json.loads(args.auth_params)
                except json.JSONDecodeError:
                    print("Error: auth-params must be valid JSON")
                    return
            
            # Register service
            service = gateway.register_service(
                name=args.name,
                service_type="graphql",
                base_url=args.url,
                auth_type=args.auth_type,
                auth_params=auth_params,
                rate_limit=args.rate_limit
            )
            
            print(f"Registered GraphQL service '{args.name}'")
            print(f"  Endpoint URL: {args.url}")
            if args.auth_type:
                print(f"  Authentication: {args.auth_type}")
            if args.rate_limit:
                print(f"  Rate limit: {args.rate_limit}")
        
        elif args.service_command == "register-oauth":
            # Create auth params
            auth_params = {
                "token_url": args.token_url,
                "client_id": args.client_id
            }
            
            if args.client_secret:
                auth_params["client_secret"] = args.client_secret
            
            # Register service
            service = gateway.register_service(
                name=args.name,
                service_type="oauth",
                base_url=args.url,
                auth_type="oauth2",
                auth_params=auth_params,
                rate_limit=args.rate_limit
            )
            
            print(f"Registered OAuth service '{args.name}'")
            print(f"  Base URL: {args.url}")
            print(f"  Token URL: {args.token_url}")
            print(f"  Client ID: {args.client_id}")
            if args.rate_limit:
                print(f"  Rate limit: {args.rate_limit}")
        
        elif args.service_command == "add-endpoint":
            service = gateway.get_service(args.service)
            
            if not service:
                print(f"Service '{args.service}' not found")
                return
            
            if service.config.get("type") != "rest" and service.config.get("type") != "oauth":
                print(f"Service '{args.service}' is not a REST or OAuth service")
                return
            
            # Add endpoint to service config
            endpoints = service.config.get("endpoints", {})
            endpoints[args.name] = {
                "path": args.path,
                "method": args.method,
                "timeout": args.timeout
            }
            
            service.config["endpoints"] = endpoints
            
            # Save updated config
            gateway._save_config()
            
            print(f"Added endpoint '{args.name}' to service '{args.service}'")
            print(f"  Path: {args.path}")
            print(f"  Method: {args.method}")
            print(f"  Timeout: {args.timeout} seconds")
        
        elif args.service_command == "add-query":
            service = gateway.get_service(args.service)
            
            if not service:
                print(f"Service '{args.service}' not found")
                return
            
            if service.config.get("type") != "graphql":
                print(f"Service '{args.service}' is not a GraphQL service")
                return
            
            # Add query to service config
            endpoints = service.config.get("endpoints", {})
            endpoints[args.name] = {
                "query": args.query,
                "timeout": args.timeout
            }
            
            service.config["endpoints"] = endpoints
            
            # Save updated config
            gateway._save_config()
            
            print(f"Added GraphQL query '{args.name}' to service '{args.service}'")
            print(f"  Query: {args.query[:50]}...")
            print(f"  Timeout: {args.timeout} seconds")
        
        elif args.service_command == "call":
            # Parse params and data
            params = None
            if args.params:
                try:
                    params = json.loads(args.params)
                except json.JSONDecodeError:
                    print("Error: params must be valid JSON")
                    return
            
            data = None
            if args.data:
                try:
                    data = json.loads(args.data)
                except json.JSONDecodeError:
                    print("Error: data must be valid JSON")
                    return
            
            # Call endpoint
            try:
                response = gateway.call_endpoint(
                    service_name=args.service,
                    endpoint_name=args.endpoint,
                    params=params,
                    data=data,
                    use_cache=not args.no_cache
                )
                
                print("Response:")
                if isinstance(response, dict) or isinstance(response, list):
                    print(json.dumps(response, indent=2))
                else:
                    print(response)
            
            except Exception as e:
                print(f"Error calling endpoint: {e}")
    
    # Handle cache commands
    elif args.command == "cache":
        if args.cache_command == "clear":
            gateway.clear_cache(service_name=args.service, endpoint_name=args.endpoint)
            
            if args.service and args.endpoint:
                print(f"Cleared cache for {args.service}.{args.endpoint}")
            elif args.service:
                print(f"Cleared cache for {args.service}")
            else:
                print("Cleared all API cache")
    
    # Handle analytics commands
    elif args.command == "analytics":
        if args.analytics_command == "show":
            analytics = APIAnalytics()
            usage_data = analytics.get_usage(days=args.days)
            
            if args.format == "json":
                # Filter by service if requested
                if args.service:
                    if args.service in usage_data["services"]:
                        usage_data["services"] = {args.service: usage_data["services"][args.service]}
                    else:
                        usage_data["services"] = {}
                
                print(json.dumps(usage_data, indent=2))
                return
            
            # Table format
            print(f"API Usage - {usage_data['period']['start']} to {usage_data['period']['end']}")
            
            # Service summary
            service_rows = []
            for service_name, service_data in usage_data["services"].items():
                if args.service and args.service != service_name:
                    continue
                
                service_rows.append([service_name, service_data["total"]])
            
            if service_rows:
                print("\nService Usage:")
                print(tabulate(service_rows, headers=["Service", "Requests"], tablefmt="grid"))
            else:
                print("\nNo service usage data available")
            
            # Endpoint details (if filtering by service)
            if args.service and args.service in usage_data["services"]:
                service_data = usage_data["services"][args.service]
                
                endpoint_rows = []
                for endpoint_name, count in service_data["endpoints"].items():
                    endpoint_rows.append([endpoint_name, count])
                
                if endpoint_rows:
                    print(f"\nEndpoint Usage for {args.service}:")
                    print(tabulate(endpoint_rows, headers=["Endpoint", "Requests"], tablefmt="grid"))
                else:
                    print(f"\nNo endpoint usage data available for {args.service}")
            
            # Daily breakdown
            daily_rows = []
            for day_data in sorted(usage_data["daily"], key=lambda x: x["date"]):
                total = sum(service["total"] for service in day_data["services"].values())
                daily_rows.append([day_data["date"], total])
            
            if daily_rows:
                print("\nDaily Usage:")
                print(tabulate(daily_rows, headers=["Date", "Requests"], tablefmt="grid"))
    
    # No command or invalid command
    else:
        print("Please specify a valid command")
        print("Available commands: service, cache, analytics")


if __name__ == "__main__":
    main()
